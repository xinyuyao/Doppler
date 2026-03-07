#!/bin/bash
set -euo pipefail

BASE_CONFIG="./config_files/run_doppler/rl_real_sys_learning.json"
COMPUTE_GRAPH_PATH="compute_graph/full_transformer_layer_261nodes.txt"
NUM_DEVICE=4
PRETRAIN_NODE_POLICY_PATH=""
PRETRAIN_DEVICE_POLICY_PATH=""

TMP_CFG=$(mktemp)
trap 'rm -f "$TMP_CFG"' EXIT

jq --arg compute_graph_path "$COMPUTE_GRAPH_PATH" \
   --argjson num_device "$NUM_DEVICE" \
   --arg pretrain_node_policy_path "$PRETRAIN_NODE_POLICY_PATH" \
   --arg pretrain_device_policy_path "$PRETRAIN_DEVICE_POLICY_PATH" \
   '.compute_graph_path = $compute_graph_path | .num_device = $num_device | .pretrain_node_policy_path = $pretrain_node_policy_path | .pretrain_device_policy_path = $pretrain_device_policy_path' \
   "$BASE_CONFIG" > "$TMP_CFG"

python3 main.py --config_path "$TMP_CFG"