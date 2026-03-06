import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scheduler.primative_nn import *

# from placement_rl.rl_agent import SoftmaxActor
# from env.latency import *

torch.autograd.set_detect_anomaly(True)


class Device_SoftmaxActor(nn.Module):
    def __init__(
        self,
        input_dim,
        n_device,
        device,
        hidden_dim=32,
    ):
        super(Device_SoftmaxActor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_device = n_device
        self.nn = FNN(input_dim, [hidden_dim], n_device).to(device)

    def forward(self, x):
        x = self.nn(x)
        return F.softmax(x, dim=-1)


class Aggregator(nn.Module):
    def __init__(self, node_dim, emb_dim, out_dim, reverse, device):
        super(Aggregator, self).__init__()

        self.reverse = reverse

        self.pre_layers = FNN(node_dim, [node_dim], emb_dim).to(
            device
        )  # preprocess input features of the source nodes before message-passing
        self.update_layers = FNN(emb_dim, [emb_dim], out_dim).to(device)

    def msg_func(self, edges):  # compute messages from source to destination nodes
        msg = F.relu(self.pre_layers(edges.src["y"]))
        return {"m": msg}

    def node_update(
        self, nodes
    ):  # update the node features based on aggregated messages 'z'
        h = F.relu(self.update_layers(nodes.data["z"]))
        return {"h": h}

    def forward(self, g):
        if self.reverse:
            g = dgl.reverse(g)
        g.update_all(
            self.msg_func, fn.sum("m", "z"), self.node_update
        )  # fn.sum('m', 'z'): aggregates the messages 'm' from all neighbors into a single value 'z'
        h = g.ndata.pop("h")
        return h


class MP(nn.Module):
    def __init__(self, emb_dim, k, device):
        super(MP, self).__init__()

        self.emb_dim = emb_dim  # embedding dimension
        self.k = k  # number of message-passing iterations
        self.device = device

        self.fpa = Aggregator(emb_dim, emb_dim, emb_dim, False, device).to(
            device
        )  # forward message passing
        self.bpa = Aggregator(emb_dim, emb_dim, emb_dim, True, device).to(
            device
        )  # backward message passing

        self.node_transform = FNN(emb_dim, [emb_dim], emb_dim).to(
            device
        )  # a multi-layer feedforward neural network

    def forward(self, g):
        self_trans = self.node_transform(g.ndata["feat"])  # "x" stores original data

        def message_pass(agg, sink):
            g.ndata["y"] = self_trans.clone()  # y stores transformed data
            for i in range(self.k):
                h = agg(g).clone()  # apply aggregator agg on graph g
                h[sink, :] = (
                    0  # set features of sink node to zero to exclude from further update
                )
                h = self_trans + h  # current features + update features
                g.ndata["y"] = h  # update "y" which stores transformed data
            return g.ndata["y"]

        out_fpa = message_pass(
            self.fpa, len(g.nodes()) - 1
        )  # might change to list of nodes for sink nodes here
        out_bpa = message_pass(
            self.bpa, 0
        )  # might change to list of nodes for sink nodes here
        return torch.cat([out_fpa, out_bpa], dim=1)


class PlaceToEmbedding(nn.Module):
    def __init__(self, emb_size, k, device):
        super(PlaceToEmbedding, self).__init__()
        self.mp = MP(emb_size, k, device).to(device)
        self.agg_p = Aggregator(
            emb_size * 2, emb_size * 2, emb_size * 2, False, device
        ).to(device)  # mean and standard deviation
        self.agg_c = Aggregator(
            emb_size * 2, emb_size * 2, emb_size * 2, True, device
        ).to(device)
        self.agg_r = Aggregator(
            emb_size * 2, emb_size * 2, emb_size * 2, False, device
        ).to(device)
        self.device = device

    def forward(self, g):
        return self.mp(g)


class PlacetoAgent:
    def __init__(
        self,
        node_dim,
        out_dim,
        n_device,
        device,
        topo_sort,
        lr,
        lr_decay,
        op_parents_dict,
        op_children_dict,
        op_parallel_dict,
        k=3,
        hidden_dim=32,
    ):

        self.node_dim = node_dim
        self.out_dim = out_dim
        self.n_device = n_device
        self.hidden_dim = hidden_dim
        self.k = k

        self.lr = lr
        self.lr_decay = lr_decay
        self.device = device

        self.topo_sort = topo_sort
        self.op_parents = op_parents_dict
        self.op_children = op_children_dict
        self.op_parallel = op_parallel_dict

        self.device_policy = Device_SoftmaxActor(
            node_dim * 8, n_device, device, hidden_dim
        ).to(device)
        self.embedding = PlaceToEmbedding(node_dim, k, device).to(device)
        self.optim = torch.optim.Adam(
            list(self.embedding.parameters()) + list(self.device_policy.parameters()),
            lr=lr,
        )

    def reset(self):
        self.log_probs = []
        self.entropy = []

    def device_selection(
        self,
        g,
        op,  # The opeartion for which a device is being selected
        device_epsilon,
        curr_step,
        device_feat_state=None,
        next_device_assign_state=None,
    ):

        g = g.to(self.device)
        emb = self.embedding(g)  # compute embedding for the entire graph
        g.ndata["y"] = emb

        if len(self.op_parents[op]):
            p_g = dgl.node_subgraph(g, [op] + self.op_parents[op]).to(
                self.device
            )  # extract subgraph containing target op + parents of target op
            pred_embeddings = self.embedding.agg_p(p_g)[
                0, :
            ]  # extracts the embedding of the target op
        else:
            pred_embeddings = torch.zeros(self.node_dim * 2).to(
                self.device
            )  # zeros holder for mean and std embedding

        if len(self.op_children[op]):  # same as parents
            c_g = dgl.node_subgraph(g, [op] + self.op_children[op]).to(self.device)
            desc_embeddings = self.embedding.agg_c(c_g)[0, :]
        else:
            desc_embeddings = torch.zeros(self.node_dim * 2).to(self.device)

        # Parallels
        r = self.op_parallel[
            op
        ]  # retreive ops that are parallel to the current operation
        if len(r):
            r_g = dgl.graph(([0] * len(r), range(1, len(r) + 1))).to(
                self.device
            )  # create a custom graph with target op connect to all parallel ops
            idx = [op] + r  # list of parallel op + target op index
            r_g.ndata["y"] = emb[
                idx, :
            ]  # assign embeddings from original graph to custom graph
            parallel_embeddings = self.embedding.agg_r(r_g)[
                0, :
            ]  # do aggregatioin on custom graph and extract target op embeddings
        else:
            parallel_embeddings = torch.zeros(self.node_dim * 2).to(self.device)

        node_embedding = emb[op, :]
        embedding = torch.cat(
            (pred_embeddings, desc_embeddings, parallel_embeddings, node_embedding),
            dim=-1,
        )

        probs = self.device_policy(embedding)
        device_distribution = torch.distributions.Categorical(probs=probs)

        random_number = torch.rand(1).item()
        if random_number > device_epsilon:
            # exploit
            action = probs.argmax()
        else:
            # explore
            action = torch.randint(0, probs.size(0), (1,))[0]

        self.log_probs.append(device_distribution.log_prob(action.to(self.device)))
        self.entropy.append(
            torch.sum(F.softmax(probs, dim=0) * F.log_softmax(probs, dim=-1))
        )

        return action.item()

    def node_selection(self, g, node_epsilon, legal_actions, curr_step):

        return self.topo_sort[curr_step]

    def finish_episode(
        self,
        rewards,
        dones,
        use_wandb,
        pg_weight=None,
        imitation_weight=None,
        entropy_weight=None,
        update_node=False,
        update_device=True,
        gamma=0.99,
    ):  # use_baseline: use baseline in reward function
        self.lr -= self.lr_decay
        self.optim.lr = self.lr
        R = 0
        pg_loss = 0
        entropy_loss = 0
        policy_loss = 0

        returns = []

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                continue

            rewards[t] += gamma * rewards[t + 1] * (1.0 - dones[t])
        returns = torch.tensor(rewards, device=self.device)

        if update_device:
            self.optim.zero_grad()
            for log_prob, R in zip(self.log_probs, returns):
                pg_loss = pg_loss - pg_weight * log_prob * R

            for entropy, R in zip(self.entropy, returns):
                entropy_loss = entropy_loss + entropy_weight * entropy

            policy_loss = pg_loss + entropy_loss
            print("policy_loss: ", policy_loss)
            policy_loss.backward()
            self.optim.step()

            if use_wandb:
                wandb.log({"policy_loss": policy_loss.item()})
                wandb.log({"entropy_loss": entropy_loss.item()})
                wandb.log({"pg_loss": pg_loss.item()})

        del self.log_probs[:]
        del self.entropy[:]
