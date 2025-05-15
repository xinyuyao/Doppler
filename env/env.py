import copy

import gym
import torch


class Env(gym.Env):
    def __init__(self, compute_g, cost_g, comm_cost, comp_cost, ndevices, device):
        self.cost_g = cost_g
        self.init_compute_g = compute_g 
        self.comm_cost = comm_cost
        self.comp_cost = comp_cost
        self.episode_length = cost_g.number_of_nodes() 
        self.num_node = cost_g.number_of_nodes() 
        self.ndevices = ndevices
        self.device = device # gpu/cpu
        
        
        self.input_nodes_dict = {}
        for node in range(cost_g.num_nodes()):
            input_nodes = list(set(cost_g.predecessors(node)))
            self.input_nodes_dict[node] = input_nodes
        
        
    
    def reset(self):
        self.curr_loc_assign = torch.full((self.num_node,), -1)
        self.visited_node = []
        self.curr_step = 0
        self.compute_graph = copy.deepcopy(self.init_compute_g)
        self.legal_actions = self._init_legal_actions()
        done = False
        
        self.node_assignment_dict = {}
        self.node_start_end_compute_dict = {key: [-1, -1] for key in range(self.episode_length)}
        self.earliest_time_to_start_list = []
        self.device_available_to_start_compute_time = torch.zeros(self.ndevices)
        self.schedule = torch.zeros(self.ndevices, self.num_node)
        
        return self.compute_graph, done, self.legal_actions
    
    def step(self, node_action, device_action, update_graph=None):
        self.curr_loc_assign[node_action] = device_action
        self.visited_node.append(node_action)
        self.curr_step += 1

        if update_graph: # Placeto only
            self.compute_graph.ndata["feat"][node_action][2+device_action] = 1 # update one-hot device
            self.compute_graph.ndata["feat"][node_action][2+self.ndevices] = 0 # update current node/not
            self.compute_graph.ndata["feat"][node_action][3+self.ndevices] = 1 # update visted/unvisited
        
        if self.curr_step == self.episode_length:
            done = True
            reward = 0
        else:
            done = False
            reward = 0
        
        self.schedule[device_action][node_action] = 1
        self._update_legal_actions(node_action)
        self._update_device_info(node_action, device_action)
    
    
        return self.compute_graph, done, self.legal_actions, reward
    
    def step_node(self, node_action):
        self.earliest_time_to_start_list = self._calculate_earliest_avaiable_device(node_action)

        self.device_features = self._compute_device_state(node_action)
        
        if update_graph: # Placeto only
            self.compute_graph.ndata["feat"][node_action][2+self.ndevices] = 1 # update current node/not
            
        return self.device_features, self.schedule
    
    def _init_legal_actions(self):
        executable_node = []
        for node in range(self.num_node):
            predecessors = self.cost_g.predecessors(node)
            if len(predecessors) == 0:
                executable_node.append(node)
        return executable_node


    def _update_legal_actions(self, last_selected_node):
        self.legal_actions.remove(last_selected_node)
        
        successor_nodes = self.cost_g.successors(last_selected_node)
        for node in successor_nodes:
            predecessors = self.cost_g.predecessors(node)
            if all(pred.item() in self.visited_node for pred in predecessors):
                self.legal_actions.append(node.item())
        
        
    def _compute_device_state(
        self,
        node_action,
    ):
        # 0. sum of computation
        device_computation_sum = torch.zeros(self.ndevices)
        for i in range(self.schedule.size(0)):
            indices = torch.nonzero(self.schedule[i] == 1, as_tuple=True)[0].tolist()
            row_sum = sum(self.comp_cost[idx] for idx in indices)
            device_computation_sum[i] = row_sum

        input_node_list = self.input_nodes_dict[node_action]
        device_node_dict = {device: [] for device in range(self.ndevices)}
        for node in input_node_list:
            device = self.node_assignment_dict[
                node.item()
            ]  # Get the device assigned to this node
            device_node_dict[device].append(
                node
            )  # Append the node to the corresponding device list

        device_input_sum_list = torch.zeros(self.ndevices)
        min_start_time_list = torch.zeros(self.ndevices)
        max_end_time_list = torch.zeros(self.ndevices)
        for device, node_list in device_node_dict.items():
            # 1. sum of input computation
            total_sum = sum(self.comp_cost[node.item()] for node in node_list)
            device_input_sum_list[device] = total_sum

            # 2. min start time of all input
            start_times = (
                [self.node_start_end_compute_dict[node.item()][0] for node in node_list]
                if len(node_list) > 0
                else [0]
            )
            min_start_time_list[device] = min(start_times)

            # 3. max end time of all input
            end_times = (
                [self.node_start_end_compute_dict[node.item()][1] for node in node_list]
                if len(node_list) > 0
                else [0]
            )
            max_end_time_list[device] = max(end_times)

        # 4. earliest starting time to compute

        device_feat_state = torch.cat(
            [
                device_computation_sum.unsqueeze(1),
                device_input_sum_list.unsqueeze(1),
                min_start_time_list.unsqueeze(1),
                max_end_time_list.unsqueeze(1),
                self.earliest_time_to_start_list.clone().detach().unsqueeze(1),
            ],
            dim=1,
        )
        return device_feat_state
                
                
    def _update_device_info(self, node_action, device_action):
        self.node_assignment_dict[node_action] = device_action
        pred_node_compute_time = self.comp_cost[node_action]

        pred_device_start_time = self.device_features[device_action][0]
        self.node_start_end_compute_dict[node_action][0] = pred_device_start_time
        self.device_available_to_start_compute_time[device_action] = (
            pred_device_start_time + pred_node_compute_time
        )
        self.node_start_end_compute_dict[node_action][1] = (
            self.device_available_to_start_compute_time[device_action].item()
        )
    
    def _calculate_earliest_avaiable_device(self, node_action):
        input_loc_list = []
        transfer_time_list = []
        end_compute_time_list = []
        input_list = self.input_nodes_dict[node_action]
        earliest_time_to_start_list = []

        if len(input_list) == 0:
            input_loc_list = [0]
            transfer_time_list = [0]
            end_compute_time_list = [0]
        else:
            for i in input_list:
                input_transfer_time = self.comm_cost[i.item()]
                input_loc = self.curr_loc_assign[i.item()]

                input_loc_list.append(input_loc)
                transfer_time_list.append(input_transfer_time)
                node_end_compute_time = self.node_start_end_compute_dict[i.item()][1]
                end_compute_time_list.append(node_end_compute_time)

        earliest_time_to_start_list = self._check_earliest_available_device(
            input_loc_list,
            transfer_time_list,
            end_compute_time_list,
        )

        return earliest_time_to_start_list
            

    def _check_earliest_available_device(
        self,
        input_loc_list,
        transfer_time_list,
        end_compute_time_list,
    ):
        earliest_time_to_start = torch.zeros(self.ndevices)
        for i in range(self.ndevices):
            ealiest_time_due_to_input = float("-inf")
            for inp in range(len(input_loc_list)):
                if i != input_loc_list[inp]:
                    end_transfer_time = end_compute_time_list[inp] + transfer_time_list[inp]
                    ealiest_time_due_to_input = max(
                        ealiest_time_due_to_input, end_transfer_time
                    )
                else:
                    ealiest_time_due_to_input = max(
                        ealiest_time_due_to_input, end_compute_time_list[inp]
                    )

            earliest_time_due_to_last_compute_on_same_device = (
                self.device_available_to_start_compute_time[i]
            )
            earliest_time_to_start[i] = max(
                ealiest_time_due_to_input, earliest_time_due_to_last_compute_on_same_device
            )

        earliest_time, earliest_device = earliest_time_to_start.min(dim=0)

        return earliest_time_to_start