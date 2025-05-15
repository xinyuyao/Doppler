import argparse
import os
import json

import wandb

from run_placement import Run_placement


def get_args():
    parser = argparse.ArgumentParser(description='Placement Experiment Arguments')
    parser.add_argument('--num_device',
                        default=4,
                        type=int,
                        help='number devices to run computation graphs')
    parser.add_argument('--num_episode',
                        default=8_000,
                        type=int,
                        help='# rl episodes')
    parser.add_argument('--learning_rate',
                        default=1e-4,
                        type=float,
                        help='Hyperparameter for RL training')
    parser.add_argument('--node_epsilon',
                        default=0.5,
                        type=float,
                        help='Exploration exploitation rate for node policy network')
    parser.add_argument('--device_epsilon',
                        default=0.5,
                        type=float,
                        help='Exploration exploitation rate for device policy network')
    parser.add_argument('--il_weight',
                        default=0.0,
                        type=float,
                        help='weight for imitation loss')
    parser.add_argument('--simulation_pg_weight',
                        default=0.0,
                        type=float,
                        help='weight for policy gradient loss cost by simulator')
    parser.add_argument('--real_sys_pg_weight',
                        default=1.0,
                        type=float,
                        help='weight for policy gradient loss cost by real execution engine')
    parser.add_argument('--entropy_weight',
                        default=0.001,
                        type=float,
                        help='regularization')
    parser.add_argument('--run_il',
                        default=False,
                        help='Learning RL policy using imitation loss')
    parser.add_argument('--run_sim_rl',
                        default=False,
                        help='Learning RL policy using policy gradient cost by simulator')
    parser.add_argument('--run_real_sys_rl',
                        default=True,
                        help='Learning RL policy using policy gradient cost by real execution engine')
    parser.add_argument('--use_node_policy_network',
                        action="store_false",
                        help='Use RL node policy network. Default: True')
    parser.add_argument('--use_device_policy_network',
                        action="store_false",
                        help='Use RL device policy network. Default: True')
    parser.add_argument('--use_placeto',
                        action="store_true",
                        help='Run Placeto baseline. Default: False')
    parser.add_argument('--compute_graph_path',
                        default="compute_graph/full_transformer_layer_261nodes.txt",
                        type=str,
                        help='Path for loading computation graph')
    parser.add_argument('--load_pretrain_node_policy_path',
                        default="",
                        type=str,
                        help='Path for loading pretrained node policy network')
    parser.add_argument('--load_pretrain_device_policy_path',
                        default="",
                        type=str,
                        help='Path for loading pretrained device policy network')
    parser.add_argument('--log_path',
                        default="heuristic_log/",
                        type=str,
                        help='Path for logging')
    parser.add_argument('--use_wandb',
                        action="store_false",
                        help='logging using wandb. Default: True')


    return parser.parse_args()

def log_with_wandb(args):
    # logging using wandb
    args_dict = {
        "num_device": args.num_device,
        "num_episode": args.num_episode,
        "learning_rate": args.learning_rate,
        "node_epsilon": args.node_epsilon,
        "device_epsilon": args.device_epsilon,
        "il_weight": args.il_weight,
        "simulation_pg_weight": args.simulation_pg_weight,
        "real_sys_pg_weight": args.real_sys_pg_weight,
        "entropy_weight": args.entropy_weight,
        "run_il": args.run_il,
        "run_sim_rl": args.run_sim_rl,
        "run_real_sys_rl": args.run_real_sys_rl,
        "use_node_policy_network": args.use_node_policy_network,
        "use_device_policy_network": args.use_device_policy_network,
        "use_placeto": args.use_placeto,
        "compute_graph_path": args.compute_graph_path,
        "load_pretrain_node_policy_path": args.load_pretrain_node_policy_path,
        "load_pretrain_device_policy_path": args.load_pretrain_device_policy_path,
        "log_path": args.log_path,
        "use_wandb": args.use_wandb
    }
    if args.run_il:
        wandb.init(
            project="run-il-exps",
            config=args_dict,
        )
    elif args.run_sim_rl:
        wandb.init(
            project="run-sim-rl-exps",
            config=args_dict,
        )
    else:
        wandb.init(
            project="run-real-sys-rl-exps",
            config=args_dict,
        )

if __name__ == '__main__':
    # Get user arguments and construct config
    args = get_args()
    
    if args.use_wandb:
        log_with_wandb(args)
    
    exp = Run_placement(args)
    exp.train()

