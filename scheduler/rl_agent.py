import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import MPNNGNN
import numpy as np
from torch.distributions.categorical import Categorical
import wandb

class Device_Policy(nn.Module):
    
    def __init__(self, num_device, num_feat, num_device_feat, device, hidden1=128, hidden2=64):
        super(Device_Policy, self).__init__()

        self.ndevices = num_device
        self.device = device  # run on cpu/gpu
        # compute embedding for actor
        self.mpnn_layer = MPNNGNN(
            node_in_feats=num_feat,
            node_out_feats=hidden1,
            edge_in_feats=1,
            edge_hidden_feats=hidden1,
            num_step_message_passing=6,
        )
        self.ffnn_layer1 = layer_init(
            nn.Linear(num_device_feat, hidden1)
        )  # extract features from device embedding
        self.ffnn_layer2 = layer_init(nn.Linear(num_feat, hidden1))
        self.ffnn_layer3 = layer_init(nn.Linear(hidden1 * 4, hidden2))
        self.ffnn_layer4 = layer_init(nn.Linear(hidden2, 1))
    
    def forward(self, g, pred_node, device_feat_state, device_assign_state, curr_step, state=None):
        if state is None:
            state = g.ndata["feat"]

        m = nn.LeakyReLU(0.1)

        if curr_step == 0:
            norm_state = normalize_feat_state(state)
            norm_edge = normalize_feat_state(g.edata["feat"])
            self.mpnn_forward = m(
                self.mpnn_layer.forward(g, norm_state, norm_edge)
            )  # can input only candidate embedding
            self.node_latent_feat = m(self.ffnn_layer2(norm_state))

        device_feat_embedding = m(
            self.ffnn_layer1(normalize_feat_state(device_feat_state))
        )

        target_node_latent_feat = self.node_latent_feat[pred_node]
        repeat_target_node_latent_feat = target_node_latent_feat.repeat(
            self.ndevices, 1
        )

        device_sum_embedding = calculate_device_embedding_and_cnt(
            device_assign_state, self.mpnn_forward, self.device
        )
        device_sum_embedding = normalize_feat_state(device_sum_embedding)

        target_node_embedding = self.mpnn_forward[pred_node]
        repeat_target_node_embedding = target_node_embedding.repeat(self.ndevices, 1)

        concat_device_embedding = torch.cat(
            (
                device_feat_embedding,
                repeat_target_node_latent_feat,
                repeat_target_node_embedding,
                device_sum_embedding,
            ),
            dim=1,
        )

        h = m(self.ffnn_layer3(concat_device_embedding))
        output = self.ffnn_layer4(h)

        return output.view(-1)

class Node_Policy(nn.Module):
    
    def __init__(self, num_device, num_feat, device, b_level_dict, t_level_dict, hidden1=128, hidden2=64):
        super(Node_Policy, self).__init__()
        self.mpnn_layer = MPNNGNN(
            node_in_feats=num_feat,
            node_out_feats=hidden1,
            edge_in_feats=1,
            edge_hidden_feats=hidden1,
            num_step_message_passing=6,
        )

        # choosing location for actor
        self.ffnn_layer1 = layer_init(nn.Linear(num_feat, hidden1))
        self.ffnn_layer2 = layer_init(nn.Linear(hidden1 * 4, hidden2))
        self.ffnn_layer3 = layer_init(nn.Linear(hidden2, 1))
        self.ndevices = num_device
        self.device = device  # run on cpu/gpu
        self.b_level_nodes_dict = b_level_dict
        self.t_level_nodes_dict = t_level_dict
    
    def forward(self, g, legal_action, curr_step, state=None):
        if state is None:
            state = g.ndata["feat"]

        m = nn.LeakyReLU(0.1)
        if curr_step == 0:
            norm_state = normalize_feat_state(state)
            norm_edge_weight = normalize_feat_state(g.edata["feat"])
            self.mpnn_forward = m(
                self.mpnn_layer.forward(g, norm_state, norm_edge_weight)
            )  # can input only candidate embedding
            self.b_level_embedding = compute_level_embedding(
                self.mpnn_forward, self.b_level_nodes_dict
            )
            self.t_level_embedding = compute_level_embedding(
                self.mpnn_forward, self.t_level_nodes_dict
            )

        legal_action_norm_state = normalize_feat_state((state[legal_action]))
        latent_feat = m(self.ffnn_layer1(legal_action_norm_state))

        norm_b_level_embedding = normalize_feat_state(
            self.b_level_embedding[legal_action]
        )
        norm_t_level_embedding = normalize_feat_state(
            self.t_level_embedding[legal_action]
        )
        norm_mpnn_embedding = self.mpnn_forward[legal_action]

        feat_representation = torch.cat(
            (
                latent_feat,
                norm_mpnn_embedding,
                norm_b_level_embedding,
                norm_t_level_embedding,
            ),
            dim=1,
        )
        h = m(self.ffnn_layer2(feat_representation))
        output = self.ffnn_layer3(h)
        return output.view(-1)
    


class RLAgent:
    def __init__(self, 
                 num_device, 
                 num_node_feat, 
                 num_device_feat, 
                 b_level_dict,
                 t_level_dict,
                 b_level_cost,
                 lr,
                 lr_decay,
                 device,
                 pretrain_node_policy_path=None,
                 pretrain_device_policy_path=None):
        
        self.num_device = num_device
        self.num_node_feat = num_node_feat
        self.num_device_feat = num_device_feat
        self.device = device
        
        self.b_level_dict = b_level_dict
        self.t_level_dict = t_level_dict
        self.b_level_cost = b_level_cost
        
        self.lr = lr
        self.lr_decay = lr_decay

        self.node_policy = Node_Policy(self.num_device, self.num_node_feat, self.device, self.b_level_dict, self.t_level_dict,)
        self.device_policy = Device_Policy(self.num_device, self.num_node_feat, self.num_device_feat, self.device)
        self.node_optim = torch.optim.Adam(list(self.node_policy.parameters()), lr=lr)
        self.device_optim = torch.optim.Adam(list(self.device_policy.parameters()), lr=lr)
        
        if pretrain_node_policy_path:
            node_policy_state_dict = torch.load(
                                        os.path.join("pretrain_models/", pretrain_node_policy_path),
                                        map_location=device,
                                        weights_only=True,
                                        )
            self.node_policy.load_state_dict(node_policy_state_dict)
            
        if pretrain_device_policy_path:
            device_policy_state_dict = torch.load(
                                        os.path.join("pretrain_models/", pretrain_device_policy_path),
                                        map_location=device,
                                        weights_only=True,
                                        )
            self.device_policy.load_state_dict(device_policy_state_dict)
    
    def reset(self):
        self.node_log_probs = []
        self.device_log_probs = []
        
        self.cp_node_log_probs = []
        self.cp_device_log_probs = []
        
        self.node_entropy = []
        self.device_entropy = []
        
        self.node_match_cp = 0
        self.device_match_cp = 0
    
    def node_selection(self, 
                       g, 
                       node_epsilon, 
                       legal_nodes,
                       curr_step,
                       ):
        
        output = self.node_policy(g, legal_nodes, curr_step)
        
        output_softmax = F.softmax(output, dim=0)
        node_prob = Categorical(output_softmax)
        
        random_number = torch.rand(1).item()
        if random_number > node_epsilon:
            # exploit
            action_id = output.argmax()
        else:
            # explore
            action_id = torch.randint(0, output_softmax.size(0), (1,))
        action = convert_to_node_id(legal_nodes, action_id)
        self.node_log_probs.append(node_prob.log_prob(action_id))
        self.node_entropy.append(torch.sum(output_softmax * F.log_softmax(output, dim=-1)))
        
        # collect heuristic log prob
        cp_node_log_probs = torch.zeros(1)
        cp_nodes = get_cp_node_list(legal_nodes, self.b_level_cost)
        for cn in cp_nodes:
            cp_node_log_probs += node_prob.log_prob(cn)
        self.cp_node_log_probs.append(cp_node_log_probs)
        
        if action_id in cp_nodes:
            self.node_match_cp += 1
                
        return action.item()
            
        
    def device_selection(self,
                      g,
                      chosen_op,# The opeartion for which a device is being selected
                      device_epsilon,
                      curr_step,
                      device_feat_state=None,
                      next_device_assign_state=None,
                      ):
        output = self.device_policy(g, chosen_op, device_feat_state, next_device_assign_state, curr_step)
        output_softmax = F.softmax(output, dim=0)
        device_probs = Categorical(output_softmax)
        
        random_number = torch.rand(1).item()
        if random_number > device_epsilon:
            # exploit
            action = output.argmax()
        else:
            # explore
            action = torch.randint(0, output_softmax.size(0), (1,))[0]
            
        
        self.device_log_probs.append(device_probs.log_prob(action))
        self.device_entropy.append(torch.sum(output_softmax * F.log_softmax(output, dim=-1)))
        
        # collect heuristic log prob
        cp_device_log_probs = torch.zeros(1)
        cp_device = get_cp_device_list(device_feat_state)
        for cd in cp_device:
            cp_device_log_probs += device_probs.log_prob(cd)
        self.cp_device_log_probs.append(cp_device_log_probs)
        
        if action in cp_device:
            self.device_match_cp += 1
        
        return action.item()
        
    
    def cp_node_selection(self, legal_nodes):
        cp_nodes = get_cp_node_list(legal_nodes, self.b_level_cost)
        
        chosen_index = torch.randint(0, cp_nodes.size(0), (1,)).item()
        return legal_nodes[chosen_index]
    
    def cp_device_selection(self, device_feat_state):
        cp_devices = get_cp_device_list(device_feat_state)
        chosen_index = torch.randint(0, cp_devices.size(0), (1,)).item()
        return cp_devices[chosen_index].item()
        
    def finish_episode(self, rewards, dones, use_wandb=False, pg_weight=None, imitation_weight=None, entropy_weight=None, update_node=False, update_device=False, gamma=0.99):
        self.lr -= self.lr_decay
        self.node_optim.lr = self.lr
        self.device_optim.lr = self.lr
    
        R = 0
        node_policy_loss = 0
        node_pg_loss = 0
        node_imitation_loss = 0
        node_entropy_loss = 0
    
        device_policy_loss = 0
        device_pg_loss = 0
        device_imitation_loss = 0
        device_entropy_loss = 0
        returns = []

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                continue
            
            rewards[t] += gamma * rewards[t + 1] * (1.0 - dones[t])
            
        returns = torch.tensor(rewards, device=self.device)
        
        if update_node:
            self.node_optim.zero_grad()

            for node_log_prob, R in zip(self.node_log_probs, returns):
                node_pg_loss = node_pg_loss - pg_weight * node_log_prob.to(self.device) * R 
        
            for cp_node_log_prob, R in zip(self.cp_node_log_probs, returns):
                node_imitation_loss = node_imitation_loss - imitation_weight * cp_node_log_prob.to(self.device) 
                
            for node_entropy, R in zip(self.node_entropy, returns):
                node_entropy_loss = node_entropy_loss + entropy_weight * node_entropy.to(self.device)
                
            
            node_policy_loss = node_pg_loss + node_imitation_loss + node_entropy_loss
            
            node_policy_loss.backward()
            self.node_optim.step()
            print("node_policy_loss: ", node_policy_loss)
            if use_wandb:
                wandb.log({"node_pg_loss": node_pg_loss.item()})
                wandb.log({"node_imitation_loss": node_imitation_loss.item()})
                wandb.log({"node_entropy_loss": node_entropy_loss.item()})
                wandb.log({"node_match_cp": self.node_match_cp})
        
        if update_device:
            self.device_optim.zero_grad()
            
            for device_log_prob, R in zip(self.device_log_probs, returns):
                device_pg_loss = device_pg_loss - pg_weight * device_log_prob.to(self.device) * R
        
            for cp_device_log_prob, R in zip(self.cp_device_log_probs, returns):
                device_imitation_loss = device_imitation_loss - imitation_weight * cp_device_log_prob.to(self.device) 
            
            for device_entropy, R in zip(self.device_entropy, returns):
                device_entropy_loss = device_entropy_loss + entropy_weight * device_entropy.to(self.device)
                
            device_policy_loss = device_pg_loss + device_imitation_loss + device_entropy_loss
            device_policy_loss.backward()
            self.device_optim.step()
            print("device_policy_loss: ", device_policy_loss)
            
            
            if use_wandb:
                wandb.log({"device_pg_loss": device_pg_loss.item()})
                wandb.log({"device_imitation_loss": device_imitation_loss.item()})
                wandb.log({"device_entropy_loss": device_entropy_loss.item()})
                wandb.log({"device_match_cp": self.device_match_cp})
                    
        del self.node_log_probs[:]
        del self.node_entropy[:]
        del self.device_log_probs[:]
        del self.device_entropy[:]
    
    
def convert_to_node_id(legal_action, action_idx):
    idx = 0
    for v in legal_action:
        if v == -1:
            continue
        if action_idx != idx:
            idx += 1
        else:
            return torch.tensor(v)
    print("Wrong action\n")


def normalize_feat_state(feat_state):
    # Column-wise normalization for the first two columns
    mean = feat_state.mean(dim=0)
    std = feat_state.std(dim=0, unbiased=False)
    normalized_feat_state = (feat_state - mean) / (std + 1e-6)
    return normalized_feat_state

def calculate_device_embedding_and_cnt(device_state, node_embeddings, device):
    device_embedding_list = []
    count_list = []

    for i in range(device_state.size(0)):
        # Find column indices where the value is one
        ones_indices = torch.nonzero(device_state[i] == 1).squeeze(1)

        # Count the number of ones in the current row
        count_ones = ones_indices.size(0)
        count_list.append(count_ones + 1e-6)

        # Select the corresponding rows from the second tensor
        selected_rows = node_embeddings[ones_indices]

        # Sum up the selected rows
        row_sum = selected_rows.sum(dim=0)

        # Append the result to the list
        device_embedding_list.append(row_sum)

    # Stack the list into a 2D tensor
    device_embedding_sum = torch.stack(device_embedding_list)

    return device_embedding_sum

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # if hasattr(layer, 'weight'):
    torch.nn.init.orthogonal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def compute_level_embedding(mpnn_forward, level_nodes_dict):
    # Initialize lists to collect the concatenated embeddings and node indices
    sum_embeddings = []

    # Iterate over each key in b_level_nodes_dict
    for key in level_nodes_dict:
        # Get the list of indices along the path
        path_indices = level_nodes_dict[key]

        # Sum over the embeddings at these indices
        path_embeddings = mpnn_forward[
            path_indices, :
        ]  # Shape: (path_length, embedding_dim)
        sum_embedding = path_embeddings.sum(dim=0)  # Shape: (embedding_dim,)

        # Append the result and the key to their respective lists
        sum_embeddings.append(sum_embedding)

    # Stack the list into a tensor
    sum_embeddings = torch.stack(sum_embeddings)  # Shape: (num_keys, 2 * embedding_dim)

    # Normalize the embeddings along the columns
    # Compute the mean and std across the nodes dimension (dim=0)
    mean = sum_embeddings.mean(dim=0, keepdim=True)
    std = (
        sum_embeddings.std(dim=0, keepdim=True) + 1e-8
    )  # Add epsilon to avoid division by zero

    # Perform the normalization
    normalized_embeddings = (sum_embeddings - mean) / std
    return normalized_embeddings

def get_cp_device_list(device_feat_state):
    # Extract the last column (start time column)
    start_time = device_feat_state[:, -1].view(-1)

    # Find the minimum start time value
    min_start_time = torch.min(start_time)

    min_indices = torch.nonzero(start_time == min_start_time).squeeze(1)
    
    return min_indices

def get_cp_node_list(legal_nodes, b_level_cost):
    legal_indices = [value for value in legal_nodes if value != -1]

    b_level_cost_for_legal_nodes = b_level_cost[legal_indices]

    # Find the maximum value
    max_b_level_cost_for_legal_nodes = torch.max(b_level_cost_for_legal_nodes)

    # Find all indices with the maximum value
    max_indices = torch.nonzero(
        b_level_cost_for_legal_nodes == max_b_level_cost_for_legal_nodes
    ).squeeze(1)
    
    return max_indices