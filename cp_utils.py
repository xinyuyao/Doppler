import torch


class ComputeNode:
    def __init__(self, idx, compute_value, communicate_value):
        self.idx = idx
        self.compute_value = compute_value
        self.communicate_value = communicate_value
        self.longest_path_value = 0
        self.children = []
        self.longest_path = []


def compute_critical_paths(num_node, comp_cost, comm_cost, input_list, output_list):
    compute_nodes = _prepare_node_data(
        num_node, comp_cost, comm_cost, input_list, output_list
    )
    roots = [idx for idx in range(num_node) if idx not in set(output_list)]
    root_nodes = [compute_nodes[idx] for idx in roots]
    for root_node in root_nodes:
        _compute_longest_paths(root_node)

    # Build the dictionary of longest paths
    path_dict = {}
    path_value_dict = {}
    for root_node in root_nodes:
        _build_longest_path_dict(root_node, path_dict, path_value_dict)

    # Get the list of longest paths in order
    longest_paths_in_order = _get_longest_paths_in_order(path_value_dict)

    return torch.tensor(longest_paths_in_order), path_dict


def _compute_longest_paths(node):
    """
    Recursively computes the longest path from the given node to any leaf node,
    and stores the list of node values on that path in node.longest_path.
    """
    if not node.children:  # If the node is a leaf
        node.longest_path = [node.idx]
        node.longest_path_value = node.compute_value
    else:
        max_length = -1
        max_path = []
        # Recurse on all children to find their longest paths
        for child in node.children:
            if child.longest_path_value == 0:
                _compute_longest_paths(child)
            # Update max_path if this child's path is longer
            if child.longest_path_value > max_length:
                max_length = child.longest_path_value
                max_path = child.longest_path
        # Prepend current node's value to the longest child path
        node.longest_path = [node.idx] + max_path
        node.longest_path_value = (
            max_length + node.compute_value + node.communicate_value
        )


def _build_longest_path_dict(node, path_dict, path_value_dict):
    """
    Traverses the tree and builds a dictionary mapping node.idx to node.longest_path.
    """
    path_dict[node.idx] = node.longest_path
    path_value_dict[node.idx] = node.longest_path_value
    for child in node.children:
        _build_longest_path_dict(child, path_dict, path_value_dict)


def _get_longest_paths_in_order(path_value_dict):
    """
    Returns a list of longest paths for each node.idx in order.
    """
    # Sort the indices to ensure order
    indices = sorted(path_value_dict.keys())
    return [path_value_dict[idx] for idx in indices]


def _prepare_node_data(num_node, comp_cost, comm_cost, input_list, output_list):
    nodes = {}
    for idx in range(num_node):
        nodes[idx] = ComputeNode(
            idx=idx,
            compute_value=comp_cost[idx].item(),
            communicate_value=comm_cost[idx].item(),
        )

    # Set up the children based on the output list
    for parent_idx, child_idx in zip(input_list, output_list):
        parent_node = nodes[parent_idx]
        child_node = nodes[child_idx]
        parent_node.children.append(child_node)

    return nodes
