import copy
import datetime
import math
import os
import time

import torch
import wandb
from env.cost import RealExecuteEngine, Simulator
from env.env import Env
from scheduler.gdp_agent import GDPAgent
from scheduler.placeto_agent import PlacetoAgent
from scheduler.rl_agent import RLAgent
from utils import (
    generate_cost_features,
    generate_gdp_features,
    generate_placeto_features,
    get_gdp_graph,
    get_placeto_graph,
    get_rl_graph,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
torch.manual_seed(42)


class Run_placement:
    def __init__(self, args):
        self.args = args

        if args.use_placeto:
            alg_name = "placeto"
        elif args.use_gdp:
            alg_name = "gdp"
        elif args.run_il:
            alg_name = "il"
        else:
            alg_name = "rl"

        self.logdir = os.path.join(
            self.args.log_path,
            "{}_{}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), alg_name
            ),
        )
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        with open(os.path.join(self.logdir, "args.txt"), "w") as f:
            for arg, value in vars(self.args).items():
                f.write(f"{arg}: {value}\n")

        # self.logdir = self.args.log_path # TODO: CHANGE LATER TO ABOVE CODE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cost_g, comm_cost, comp_cost = generate_cost_features(
            self.args.compute_graph_path
        )  # used for environment to keep track of the state

        lr_decay = (self.args.learning_rate - 0) / self.args.num_episode

        if alg_name == "placeto":
            self.topo_sort, op_parents_dict, op_children_dict, op_parallel_dict = (
                generate_placeto_features(self.args.compute_graph_path)
            )
            self.graph = get_placeto_graph(
                self.args.compute_graph_path, self.args.num_device
            )
            self.agent = PlacetoAgent(
                self.graph.ndata["feat"].shape[1],
                10,
                self.args.num_device,
                self.device,
                self.topo_sort,
                self.args.learning_rate,
                lr_decay,
                op_parents_dict,
                op_children_dict,
                op_parallel_dict,
            )
        elif alg_name == "gdp":
            self.graph = get_gdp_graph(
                self.args.compute_graph_path, self.args.num_device
            )
            self.agent = GDPAgent(
                self.device,
                self.args.num_device,
                self.args.learning_rate,
                lr_decay,
                self.graph.number_of_nodes(),
                self.graph.ndata["feat"].shape[1],
            )
        else:
            self.graph, b_level_cost, b_level_dict, t_level_dict = get_rl_graph(
                self.args.compute_graph_path
            )
            self.agent = RLAgent(
                self.args.num_device,
                self.graph.ndata["feat"].shape[1],
                5,
                b_level_dict,
                t_level_dict,
                b_level_cost,
                self.args.learning_rate,
                lr_decay,
                self.device,
                self.args.load_pretrain_node_policy_path,
                self.args.load_pretrain_device_policy_path,
            )

        if args.run_sim_rl or args.run_il:
            self.cost_engine = Simulator(self.cost_g, comm_cost, comp_cost)
        else:
            self.cost_engine = RealExecuteEngine()

        self.env = Env(
            self.graph,
            self.cost_g,
            comm_cost,
            comp_cost,
            self.args.num_device,
            self.device,
        )

    def train(self):

        node_epsilon_decay = (self.args.node_epsilon - 0) / self.args.num_episode
        curr_node_epsilon = self.args.node_epsilon

        device_epsilon_decay = (self.args.device_epsilon - 0) / self.args.num_episode
        curr_device_epsilon = self.args.device_epsilon

        running_time_sum = torch.zeros(1)

        for episode_id in range(self.args.num_episode):
            if self.args.use_wandb:
                wandb.log({"episode_id": episode_id})
                wandb.define_metric("episode_id")
                wandb.define_metric("*", step_metric="episode_id")

            with open(os.path.join(self.logdir, "output.txt"), "a") as file:
                file.write(
                    (
                        "====================================== starting episode %d ===================================\n"
                        % episode_id
                    )
                )

            logging_into_text_file(
                self.logdir, torch.tensor(self.agent.lr), "current learning rate"
            )
            logging_into_text_file(
                self.logdir, torch.tensor(curr_node_epsilon), "current node epsilon"
            )
            logging_into_text_file(
                self.logdir, torch.tensor(curr_device_epsilon), "current device epsilon"
            )

            start_time = time.time()
            if self.args.use_gdp:
                cost, reward, dev_schedule, running_time_sum = self.gdp_run_one_episode(
                    episode_id,
                    copy.deepcopy(self.graph),
                    curr_node_epsilon,
                    curr_device_epsilon,
                    running_time_sum,
                )
            else:
                cost, reward, dev_schedule, running_time_sum = self.run_one_episode(
                    episode_id,
                    copy.deepcopy(self.graph),
                    curr_node_epsilon,
                    curr_device_epsilon,
                    running_time_sum,
                )
            end_time = time.time()
            print("Time for running one episode: ", end_time - start_time)
            logging_into_text_file(self.logdir, dev_schedule, "Device Assignment")
            logging_into_text_file(self.logdir, cost, "Running time")

            if (
                not self.args.use_placeto
                and not self.args.use_gdp
                and self.args.use_node_policy_network
            ):
                torch.save(
                    self.agent.node_policy.state_dict(),
                    os.path.join(self.logdir, "node_policy_network.pt"),
                )
            if self.args.use_device_policy_network:
                torch.save(
                    self.agent.device_policy.state_dict(),
                    os.path.join(self.logdir, "device_policy_network.pt"),
                )

            curr_node_epsilon -= node_epsilon_decay
            curr_device_epsilon -= device_epsilon_decay

            if self.args.use_wandb:
                wandb.log({"running_time": cost})
                wandb.log({"rewards": reward})
                wandb.log(
                    {"running_time_avg": running_time_sum.item() / (episode_id + 1)}
                )

    def gdp_run_one_episode(
        self, episode_id, graph, node_epsilon, device_epsilon, running_time_sum
    ):
        device_list = []
        node_list = []
        reward_list = []
        done_list = []
        device_schedule = torch.zeros(graph.number_of_nodes(), dtype=int)

        self.agent.reset()
        node_list = torch.arange(graph.number_of_nodes(), dtype=torch.int32)
        device_list = self.agent.device_selection(graph, device_epsilon)
        print("device_list: ", device_list)
        reward_list = torch.zeros(graph.number_of_nodes(), dtype=torch.float32)
        done_list = torch.zeros(graph.number_of_nodes(), dtype=torch.int32)
        done_list[-1] = 1  # last node is done
        device_schedule = torch.tensor(device_list)

        logging_into_text_file(self.logdir, torch.tensor(node_list), "node_list")
        logging_into_text_file(self.logdir, torch.tensor(device_list), "device_list")

        update_node = False
        update_device = False
        if self.args.use_node_policy_network:
            update_node = True
        if (
            self.args.use_placeto
            or self.args.use_gdp
            or self.args.use_device_policy_network
        ):
            update_device = True

        logging_into_text_file(self.logdir, torch.tensor(update_node), "update_node")
        logging_into_text_file(
            self.logdir, torch.tensor(update_device), "update_device"
        )

        cost = self.cost_engine.get_cost(device_schedule.tolist())
        running_time_sum += cost
        running_time_avg = running_time_sum / (episode_id + 1)

        if running_time_avg > cost:
            reward = running_time_avg - cost
        else:
            reward = (-1) * (cost - running_time_avg)
        reward_list[-1] = reward
        logging_into_text_file(self.logdir, torch.tensor(reward), "Reward")

        if self.args.run_sim_rl:
            pg_weight = self.args.simulation_pg_weight
        else:
            pg_weight = self.args.real_sys_pg_weight

        self.agent.finish_episode(
            reward_list,
            done_list,
            self.args.use_wandb,
            pg_weight,
            self.args.il_weight,
            self.args.entropy_weight,
            update_node,
            update_device,
        )

        return cost, reward, device_schedule, running_time_sum

    def run_one_episode(
        self, episode_id, graph, node_epsilon, device_epsilon, running_time_sum
    ):
        device_list = []
        node_list = []
        reward_list = []
        done_list = []
        device_schedule = torch.zeros(graph.number_of_nodes(), dtype=int)

        self.agent.reset()
        next_obs, next_done, next_legal_actions = (
            self.env.reset()
        )  # next_obs is the compute graph with init node,edge features

        # Your code block
        # with torch.cuda.device(0):
        # inference_start_time = time.time()
        for step in range(graph.number_of_nodes()):
            if (
                self.args.use_placeto
                or self.args.use_gdp
                or self.args.use_node_policy_network
            ):
                node_action = self.agent.node_selection(
                    next_obs, node_epsilon, next_legal_actions, step
                )
            else:
                node_action = self.agent.cp_node_selection(next_legal_actions)

            device_feat_state, next_device_assign_state = self.env.step_node(
                node_action, self.args.use_placeto
            )
            if (
                self.args.use_placeto
                or self.args.use_gdp
                or self.args.use_device_policy_network
            ):
                device_action = self.agent.device_selection(
                    next_obs,
                    node_action,
                    device_epsilon,
                    step,
                    device_feat_state,
                    next_device_assign_state,
                )
            else:
                device_action = self.agent.cp_device_selection(device_feat_state)

            if self.args.use_placeto:
                next_obs, next_done, next_legal_actions, reward = self.env.step(
                    node_action, device_action, self.args.use_placeto
                )
            else:
                next_obs, next_done, next_legal_actions, reward = self.env.step(
                    node_action, device_action
                )

            node_list.append(node_action)
            device_list.append(device_action)
            reward_list.append(reward)
            done_list.append(next_done)
            device_schedule[node_action] = device_action

            # inference_end_time = time.time()
            # if episode_id != 0:
            #     with open(os.path.join(self.logdir, f"{os.path.basename(self.args.compute_graph_path)}_Inference_Time_Cost.txt"), "a") as file:
            #         file.write(f"{inference_end_time - inference_start_time}\n")

        # # After block
        # peak_mem = torch.cuda.max_memory_allocated(self.device)
        # print(f"Peak memory allocated during inference: {peak_mem / 1024 ** 2:.2f} MB")
        # with open(os.path.join(self.logdir, f"{os.path.basename(self.args.compute_graph_path)}_Inference_Memory_Cost.txt"), "a") as file:
        #     file.write(f"{peak_mem / 1024 ** 2:.2f}\n") # MB

        logging_into_text_file(self.logdir, torch.tensor(node_list), "node_list")
        logging_into_text_file(self.logdir, torch.tensor(device_list), "device_list")

        update_node = False
        update_device = False
        if self.args.use_node_policy_network:
            update_node = True
        if (
            self.args.use_placeto
            or self.args.use_gdp
            or self.args.use_device_policy_network
        ):
            update_device = True

        logging_into_text_file(self.logdir, torch.tensor(update_node), "update_node")
        logging_into_text_file(
            self.logdir, torch.tensor(update_device), "update_device"
        )

        cost = self.cost_engine.get_cost(device_schedule.tolist())
        running_time_sum += cost
        running_time_avg = running_time_sum / (episode_id + 1)

        if running_time_avg > cost:
            reward = running_time_avg - cost
        else:
            reward = (-1) * (cost - running_time_avg)
        reward_list[-1] = reward
        logging_into_text_file(self.logdir, torch.tensor(reward), "Reward")

        if self.args.run_sim_rl:
            pg_weight = self.args.simulation_pg_weight
        else:
            pg_weight = self.args.real_sys_pg_weight

        # training_start_time = time.time()
        # Your code block
        # with torch.cuda.device(self.device):
        self.agent.finish_episode(
            reward_list,
            done_list,
            self.args.use_wandb,
            pg_weight,
            self.args.il_weight,
            self.args.entropy_weight,
            update_node,
            update_device,
        )

        # # After block
        # peak_mem = torch.cuda.max_memory_allocated(self.device)
        # print(f"Peak memory allocated during training: {peak_mem / 1024 ** 2:.2f} MB")
        # with open(os.path.join(self.logdir, f"{os.path.basename(self.args.compute_graph_path)}_Training_Memory_Cost.txt"), "a") as file:
        #     file.write(f"{peak_mem / 1024 ** 2:.2f}\n") # MB

        # training_end_time = time.time()
        # if episode_id != 0:
        #     with open(os.path.join(self.logdir, f"{os.path.basename(self.args.compute_graph_path)}_Training_Time_Cost.txt"), "a") as file:
        #         file.write(f"{training_end_time - training_start_time}\n")

        return cost, reward, device_schedule, running_time_sum


def logging_into_text_file(dir_name, var, var_name):
    with open(os.path.join(dir_name, "output.txt"), "a") as file:
        file.write(f"{var_name}: ")
        if var.dim() == 0:
            file.write(f"{var.item()}\n")
        else:
            file.write("[")
            for value in var.tolist():
                file.write(f"{value}, ")
            file.write("]\n")