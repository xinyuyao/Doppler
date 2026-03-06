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


class GraphSAGEEncoder(nn.Module):
    def __init__(self, device, input_dim, hidden_dim=32):
        """
        GraphSAGE-based graph encoder using DGL graph and g.ndata["feat"] as input.
        Args:
            input_dim (int): dimension of node input features.
            hidden_dim (int): output embedding dimension.
        """
        super(GraphSAGEEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        # First layer
        self.W_neighbor1 = nn.Linear(hidden_dim, hidden_dim)
        self.b_neighbor1 = nn.Parameter(torch.zeros(hidden_dim))
        self.ffnn = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        # Projection for input_dim ≠ hidden_dim
        self.input_project = None
        if input_dim != hidden_dim:
            self.input_project = nn.Linear(input_dim, hidden_dim)

        self.act = nn.Sigmoid()

    def forward(self, g):
        h = g.ndata["feat"]

        if self.input_project is not None:
            h = self.input_project(h)

        # First layer
        h1 = torch.zeros_like(h)
        for v in range(g.num_nodes()):
            neighbors = torch.stack(list(g.predecessors(v)) + list(g.successors(v)))
            if len(neighbors) > 1:
                nbr_feats = h[neighbors]
                nbr_transformed = self.act(
                    self.W_neighbor1(nbr_feats) + self.b_neighbor1
                )
                if len(neighbors) > 1:
                    nbr_agg = torch.max(nbr_transformed, dim=0).values
                else:
                    nbr_agg = nbr_transformed
            else:
                nbr_agg = torch.zeros(self.hidden_dim, device=self.device)
            combined = torch.cat([h[v], nbr_agg.squeeze(0)], dim=0)
            h1[v] = self.ffnn(combined)

        return h1


class TransformerXLPolicy(nn.Module):
    def __init__(
        self,
        device,
        num_devices,
        num_node_feat,
        hidden_dim=64,
        num_heads=4,
        mem_length=8,
    ):
        super(TransformerXLPolicy, self).__init__()
        self.device = device
        self.num_devices = num_devices
        self.hidden_dim = hidden_dim
        self.mem_length = mem_length
        self.graph_encoder = GraphSAGEEncoder(device, num_node_feat, hidden_dim).to(
            self.device
        )

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads).to(
            self.device
        )
        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_dim, hidden_dim * 4
            ),  # expand (feed-forward inner dimension)
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        ).to(self.device)
        self.layernorm1 = nn.LayerNorm(hidden_dim).to(self.device)
        self.layernorm2 = nn.LayerNorm(hidden_dim).to(self.device)
        # Output layer to get logits for device placements
        self.out_proj = nn.Linear(hidden_dim, num_devices).to(self.device)
        self.embed = None

    def forward(self, g):
        g = g.to(self.device)
        self.embed = self.graph_encoder(g)

        x = self.embed  # shape (N, hidden_dim)

        N = x.shape[0]
        outputs = []
        memory = torch.zeros(
            0, self.hidden_dim, device=self.device
        )  # (mem_len, hidden_dim)

        segment_length = self.mem_length
        for start in range(0, N, segment_length):
            end = min(start + segment_length, N)
            segment = x[start:end]  # (seg_len, hidden_dim)
            if memory.shape[0] > 0:
                kv = torch.cat([memory, segment], dim=0)  # (mem + seg, hidden_dim)
            else:
                kv = segment

            # Add batch dim: (seq_len, 1, hidden_dim)
            query = segment.unsqueeze(1).to(self.device)
            key_value = kv.unsqueeze(1).to(self.device)

            # Apply attention
            attn_output, _ = self.attn(
                query, key_value, key_value
            )  # shape: (seg_len, 1, hidden_dim)
            attn_output = attn_output.squeeze(1)  # (seg_len, hidden_dim)

            # Residual + norm
            segment = self.layernorm1(segment + attn_output)

            # FFN
            ff_output = self.ffn(segment)
            ff_output = ff_output.squeeze(1)  # (seg_len, hidden_dim)
            segment_out = self.layernorm2(segment + ff_output)
            outputs.append(segment_out)

            # Update memory (detach!)
            if segment_out.shape[0] >= self.mem_length:
                memory = segment_out[-self.mem_length :].clone().detach()
            else:
                memory = segment_out.clone().detach()

        full_out = torch.cat(outputs, dim=0)  # (N, hidden_dim)
        logits = self.out_proj(full_out)  # (N, num_devices)
        return logits


class GDPAgent:
    def __init__(
        self,
        device,
        num_devices,
        lr,
        lr_decay,
        num_nodes,
        num_node_feat,
        op_embedding_dim=128,
        hidden_dim=64,
        num_heads=4,
        mem_length=8,
    ):
        self.device_policy = TransformerXLPolicy(
            device, num_devices, num_node_feat, hidden_dim, num_heads, mem_length
        )
        self.device = device
        self.num_devices = num_devices
        self.num_nodes = num_nodes
        self.lr = lr
        self.lr_decay = lr_decay
        self.optim = torch.optim.Adam(self.device_policy.parameters(), lr=self.lr)

    def reset(self):
        self.log_probs = []
        self.entropy = []

    def device_selection(self, g, device_epsilon):  # logits: (N, num_devices)
        logits = self.device_policy(g)
        probs = F.softmax(logits, dim=-1)  # (N, num_devices)

        device_action_list = []
        for prob in probs:
            device_distribution = torch.distributions.Categorical(probs=prob)

            random_number = torch.rand(1).item()
            if random_number > device_epsilon:
                # exploit
                action = prob.argmax()
            else:
                # explore
                action = torch.randint(0, prob.size(0), (1,))[0]

            device_action_list.append(action.item())
            self.log_probs.append(device_distribution.log_prob(action.to(self.device)))
            self.entropy.append(
                torch.sum(F.softmax(prob, dim=0) * F.log_softmax(prob, dim=-1))
            )

        return device_action_list

    def node_selection(self):
        return torch.arange(self.num_nodes, dtype=torch.int32)

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
            policy_loss.backward()
            self.optim.step()

            if use_wandb:
                wandb.log({"policy_loss": policy_loss.item()})
                wandb.log({"entropy_loss": entropy_loss.item()})
                wandb.log({"pg_loss": pg_loss.item()})

        del self.log_probs[:]
        del self.entropy[:]
