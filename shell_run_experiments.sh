#!/bin/bash
set -euo pipefail

BASE_CONFIG="./config_files/run_doppler/imitation_learning.json"
COMPUTE_GRAPH_PATH="compute_graph/chain_mm_112nodes.txt"
PRETRAIN_NODE_POLICY_PATH=""
PRETRAIN_DEVICE_POLICY_PATH=""

TMP_CFG=$(mktemp)
trap 'rm -f "$TMP_CFG"' EXIT

jq --arg compute_graph_path "$COMPUTE_GRAPH_PATH" \
   '.compute_graph_path = $compute_graph_path' \
   "$BASE_CONFIG" > "$TMP_CFG"

python3 main.py --config_path "$TMP_CFG"