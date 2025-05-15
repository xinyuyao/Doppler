import re

import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt

from cp_utils import compute_critical_paths

def generate_cost_features(file_name):
    
    in_list, out_list, num_node, node_info = _read_compute_graph(file_name)
    g = dgl.graph(
        (torch.tensor(in_list), torch.tensor(out_list)), num_nodes=num_node
    )
    comm_cost = torch.squeeze(node_info[:, 1:2])
    comp_cost = torch.squeeze(node_info[:, :1])
    
    return g, comm_cost, comp_cost 

def generate_placeto_features(file_name):
    
    g, comm_cost, comp_cost = generate_cost_features(file_name)
    
    # get topological order
    nx_graph = g.to_networkx().to_directed()
    topo_order = list(nx.topological_sort(nx_graph))
    
    op_parents = [list(nx_graph.predecessors(n)) for n in range(g.number_of_nodes())]
    op_children = [list(nx_graph.successors(n)) for n in range(g.number_of_nodes())]
    op_parallel = []
    for n in range(g.number_of_nodes()):
        parallel_group = []
        for m in range(g.number_of_nodes()):
            if m == n:
                continue
            if not nx.has_path(nx_graph, n, m) and not nx.has_path(nx_graph,  m, n):
                parallel_group.append(m)
        op_parallel.append(parallel_group)
    
    return topo_order, op_parents, op_children, op_parallel


def get_placeto_graph(file_name, num_device):
    # todo
    in_list, out_list, num_node, node_info = _read_compute_graph(file_name)
    
    # comput edge features
    dir_g = dgl.graph(
        (torch.tensor(in_list), torch.tensor(out_list)), num_nodes=num_node
    )
    comm_cost = torch.squeeze(node_info[:, 1:2])
    edge_features = _compute_edge_features(dir_g, comm_cost)
    
    # compute node features
    # 1. total runtime
    comp_cost = node_info[:, :1]
    
    # 2. output tensor size
    output_tensor_size = _compute_output_tensor_size(dir_g, comp_cost, num_node)
    
    # 3. [one-hot encoding of device, binary encoding for current node, binary encoding for visted/unvisited]
    one_hot_device = torch.zeros(num_node, num_device + 2)
    
    node_features = torch.cat(
        (
            comp_cost,
            output_tensor_size,
            one_hot_device,
        ),
        dim=1,
    )

    dir_g.ndata["feat"] = node_features
    dir_g.edata["feat"] = edge_features
    
    print("graph node features: ", dir_g.ndata["feat"][:10,:])
    print("graph edge features: ", dir_g.edata["feat"][:10,:])
    
    return dir_g
    
def get_rl_graph(file_name):
    in_list, out_list, num_node, node_info = _read_compute_graph(file_name)
    comm_cost = torch.squeeze(node_info[:, 1:2])
    
    
    # start to compute node features
    dir_g = dgl.graph(
        (torch.tensor(in_list), torch.tensor(out_list)), num_nodes=num_node
    )
    edge_data = _compute_edge_features(dir_g, comm_cost)
    dir_g.edata["feat"] = edge_data
    
    g = dgl.to_bidirected(dir_g, copy_ndata=True) # this is the graph used for updae
    
    # compute edge features
    edge_features = _compute_edge_features_for_undirected(g, comm_cost)
    
    # 0. Computation costs (d=1)\\
    comp_cost = torch.squeeze(node_info[:, :1])
    
    # 1. Bottom-level cost. (d=1)\\
    b_level_cost, b_level_nodes_dict = compute_critical_paths(
        num_node, comp_cost, comm_cost, in_list, out_list
    )
    # 2. Top-level cost. (d=1)\\
    t_level_cost, t_level_nodes_dict = compute_critical_paths(
        num_node, comp_cost, comm_cost, out_list, in_list
    )

    # 3. Input Communication cost. (d=1)\\
    in_edge_weight = _compute_input_communication_cost(num_node, dir_g)
    
    # 4. Output Communication Cost. (d=1)\\
    reverse_direction_g = dgl.reverse(dir_g, copy_edata=True)
    out_edge_weight = _compute_input_communication_cost(num_node, reverse_direction_g)

    node_features = torch.cat(
        (
            comp_cost.unsqueeze(1),
            b_level_cost.unsqueeze(1),
            t_level_cost.unsqueeze(1),
            in_edge_weight,
            out_edge_weight,
        ),
        dim=1,
    )
    
    g.ndata["feat"] = node_features
    g.edata["feat"] = edge_features
    
    print("graph node features: ", g.ndata["feat"][:10,:])
    print("graph edge features: ", g.edata["feat"][:10,:])
    
    return g, b_level_cost, b_level_nodes_dict, t_level_nodes_dict


def _read_compute_graph(file_name):
    data = []
    
    with open(file_name, "r") as f:
        data = f.readlines()

    input_node_list = list(map(int, re.findall(r"\d+", data[1]))) # not used in RL
    in_list = list(map(int, re.findall(r"\d+", data[2])))
    out_list = list(map(int, re.findall(r"\d+", data[3])))
    num_node = int(data[0]) - len(input_node_list)
    
    # construct the compute graph 
    g = dgl.graph(
        (torch.tensor(in_list), torch.tensor(out_list)), num_nodes=num_node
    )
    
    print("# total nodes (including inpute nodes): ", int(data[0]))
    print("# node in the graph: ", num_node)
    print("# input ndoes: ", len(input_node_list))
    
    node_info = []
    for i in range(num_node):
        node_info.append(list(map(int, re.findall(r"\d+", data[4 + i]))))
    node_info = torch.tensor(node_info)
    
    return in_list, out_list, num_node, node_info


def _compute_edge_features(g, comm_cost):
    edges = g.edges()
    src = edges[0]
    dst = edges[1]
    edge_dict = {}
    edge_feat = torch.zeros(len(g.edges()[0]), 1)
    edge_feat = torch.zeros(len(src), 1)

    for idx in range(len(src)):
        s = src[idx].item()
        d = dst[idx].item()
        if d in edge_dict and (s in edge_dict[d]):
            edge_feat[idx][0] = comm_cost[(dst[idx].item())]
        else:
            edge_feat[idx][0] = comm_cost[src[idx].item()]
    return edge_feat


def _compute_input_communication_cost(num_node, dir_g):
    in_edge_weight = torch.zeros(num_node, 1)
    for node_id in dir_g.nodes():
        # Get in-edges of the node
        in_edges = dir_g.in_edges(node_id, form="all")
        in_edge_ids = in_edges[
            2
        ]  # 'all' form returns (src, dst, eid), we need eid to index weights
        in_edge_weights = dir_g.edata["feat"][in_edge_ids]
        sum_in_edge_weights = torch.sum(in_edge_weights)
        in_edge_weight[node_id][0] = sum_in_edge_weights

    return in_edge_weight


def _compute_edge_features(dir_g, comm_cost):
    edges = dir_g.edges()
    src = edges[0]
    dst = edges[1]
    edge_dict = {}
    edge_feat = torch.zeros(len(src), 1)

    for idx in range(len(src)):
        s = src[idx].item()
        d = dst[idx].item()
        if s in edge_dict:
            edge_dict[s].append(d)
        else:
            edge_dict[s] = [d]
        edge_feat[idx][0] = comm_cost[src[idx].item()]

    return edge_feat


def _compute_edge_features_for_undirected(g, comm_cost):
    edges = g.edges()
    src = edges[0]
    dst = edges[1]
    edge_dict = {}
    edge_feat = torch.zeros(len(g.edges()[0]), 1)
    edge_feat = torch.zeros(len(src), 1)

    for idx in range(len(src)):
        s = src[idx].item()
        d = dst[idx].item()
        if d in edge_dict and (s in edge_dict[d]):
            edge_feat[idx][0] = comm_cost[(dst[idx].item())]
        else:
            edge_feat[idx][0] = comm_cost[src[idx].item()]
    return edge_feat


def _compute_output_tensor_size(dir_g, comp_cost, num_node):
    output_tensor_size_list = torch.zeros(num_node,1)
    # Iterate over all nodes in the graph
    for node in range(dir_g.num_nodes()):
        # Get the successors of the node
        output_nodes = dir_g.successors(node)
        for out_node in output_nodes:
            output_tensor_size_list[node][0] += comp_cost[out_node][0]
    
    return output_tensor_size_list