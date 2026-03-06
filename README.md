# DOPPLER: Dual-Policy Learning for Device Assignment in Asynchronous Dataflow Graphs

[![arXiv](https://img.shields.io/badge/arXiv-2505.23131-b31b1b.svg)](https://openreview.net/pdf?id=OQQK8gMC5H)
[![ICLR](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![DOPPLER device assignment on a Llama layer](gif_visualizations/ffnn-CORGI-SYS-assingment-animation.gif)

Official code for the ICLR 2026 paper:

> **DOPPLER: Dual-Policy Learning for Device Assignment in Asynchronous Dataflow Graphs**
> Xinyu Yao, Daniel Bourgeois, Abhinav Jain, Yuxin Tang, Jiawen Yao, Zhimin Ding, Arlei Silva, Chris Jermaine

## 🦉 Overview


DOPPLER addresses the **device assignment problem** for ML workloads running on asynchronous dataflow execution engines (such as [einsummable](https://github.com/dcbdan/einsummable)). Given a computation graph where nodes represent operations and edges represent data dependencies, the goal is to assign each operation to a device (e.g., GPU) so that the total execution time is minimized.

Unlike prior work that targets bulk-synchronous systems, DOPPLER is designed for **asynchronous** runtimes where devices operate without barrier synchronization, making device utilization and scheduling order jointly critical.

### Key Idea: Dual Policy

DOPPLER decomposes device assignment into two sub-decisions learned by separate neural networks:

- **SEL policy (Node Selection):** Selects *which* ready operation to schedule next. Uses an MPNN with bottom-level and top-level critical path embeddings as contextual features.
- **PLC policy (Device Placement):** Selects *which device* to assign the chosen operation to. Uses an MPNN combined with per-device state features encoding workload and communication costs.

Both policies are trained jointly using a three-stage curriculum:

| Stage | Method | Cost Signal |
|-------|--------|-------------|
| 1 | Imitation Learning (IL) | Critical path heuristic |
| 2 | Simulation RL (Sim-RL) | SimPy-based simulator |
| 3 | Real System RL (Real-RL) | Actual execution engine |

## 🛠️ Installation

### Part 1: Dual-Policy IL/RL Training Environment Setup
#### Option 1: Conda (recommended)

Requirements: [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), CUDA 12.1

```bash
conda env create -f environment.yml
conda activate doppler
export DGLBACKEND=pytorch
```

> **Tip:** Add `export DGLBACKEND=pytorch` to your `~/.bashrc` or `~/.zshrc` to avoid setting it each session.

#### Option 2: Singularity container

A Singularity definition file is provided for HPC environments:

```bash
singularity build rl_env.sif rl_env.def
singularity shell --nv rl_env.sif
```

### Part 2: Einsummable Real Execution Engine Environment Setup (Stage III training)
#### Set up Doppler Repository

Real System RL requires a running einsummable execution engine on a remote machine. Configure the connection by copying `.env.example` and filling in your credentials:

```bash
cp env/.env.example env/.env
# Edit env/.env with your remote connection details
```

The `.env` file should contain:

```
REMOTE_USER=<ssh username>
REMOTE_IP=<ip address of the execution engine host>
REMOTE_PATH=<path on remote host where assignment file is written>
PASSWORD=<ssh password>
```

#### Set up Doppler_Einsummable Repository
> **Note:** For Stages 1 and 2 (IL and Sim-RL), the real execution engine is not required.

Create the environment of [Doppler_Einsummable](https://github.com/xinyuyao/Doppler_Einsummable) by running the following command:

```bash
# setup Einsummable repository and build files
cd ..
git clone https://github.com/xinyuyao/Doppler_Einsummable.git
cd Doppler_Einsummable
mkdir build
cd build
cmake ../ -DGPU_EXECUTION_ENGINE=ON (or specify cpu engine)
make proto (if no protocol buffer)
make -j
# set up singularity container for running Einsummable
singularity build --fakeroot gpuengine.sif gpuengine.def
singularity shell --nv rl_env.sif
```

## 📊 Compute Graphs

Pre-generated computation graphs are provided in `compute_graph/`. Each file encodes a directed acyclic graph with node computation costs and communication costs:

| File | Description | Nodes | # GPUs |
|------|-------------|:-------:|:-------:|
| `chain_mm_112nodes.txt` | Chain of matrix multiplications | 112 |  4  |
| `ffnn_192nodes.txt` | Feedforward neural network | 192 |  4  |
| `transformer_block_215nodes.txt` | Transformer block | 215 |  4  |
| `full_transformer_layer_261nodes.txt` | Full transformer layer | 261 |  4  |

Variants prefixed with `gdp_` use a different feature encoding compatible with the GDP baseline and `8gpu_` denotes the computation graph for running on eight GPUs.

## 🚀 Running Experiments

All experiments are configured via JSON files in `config_files/`. Each config file sets all hyperparameters for a given training stage and method.

### DOPPLER (three-stage training)

DOPPLER is trained in three progressive stages. Each stage builds on the previous one, with policy checkpoints saved after every episode to `heuristic_log/<timestamp>_<alg>/`.

---

#### Stage 1 — Imitation Learning (IL)

The SEL and PLC policies are bootstrapped by imitating the critical path heuristic. This stage is fast, requires no real hardware, and produces a strong initialization for RL fine-tuning.

```bash
python main.py --config_path config_files/run_doppler/imitation_learning.json
```

Key settings: `run_il=true`, `il_weight=1.0`.

---

#### Stage 2 — Simulation RL (Sim-RL)

The policies are fine-tuned with REINFORCE using a SimPy discrete-event simulator as the cost oracle. This is significantly faster than querying real hardware and further improves over the IL baseline. No GPU cluster is required.

Before running, set the Stage 1 checkpoint paths in `config_files/run_doppler/rl_simulation_learning.json`:

```json
"load_pretrain_node_policy_path": "heuristic_log/<timestamp>_il/node_policy_network.pt",
"load_pretrain_device_policy_path": "heuristic_log/<timestamp>_il/device_policy_network.pt"
```

Then run:

```bash
python main.py --config_path config_files/run_doppler/rl_simulation_learning.json
```

Key settings: `run_sim_rl=true`, `simulation_pg_weight=1.0`.

---

#### Stage 3 — Real System RL (Real-RL)

The policies are fine-tuned using actual wall-clock execution times measured on a real GPU cluster running the [einsummable](https://github.com/dcbdan/einsummable) engine. This stage requires the einsummable server to be running concurrently (see [Installation > Part 2](#part-2-einsummable-real-execution-engine-environment-setup-stage-iii-training)).

The training script writes a device assignment file to `REMOTE_PATH` (configured in `env/.env`). The einsummable server reads the assignment, executes the ML workload on real hardware, and returns the measured execution time as the RL reward.

**Terminal 1** — start DOPPLER training (after setting checkpoint paths in the config as above):

```bash
python main.py --config_path config_files/run_doppler/rl_real_sys_learning.json
```

**Terminal 2** — start the einsummable execution server for your target workload:

| Workload | Command |
|----------|---------|
| Feedforward Neural Network | `./rl_ffnn` |
| Chain of Matrix Multiplications | `./rl_chainmm` |
| Transformer Block | `./rl_transformer_block` |
| Full Transformer Layer | `./rl_transformer_layer` |

Make sure `compute_graph_path` in the Stage 3 config matches the workload you start in Terminal 2.

Key settings: `run_real_sys_rl=true`, `real_sys_pg_weight=1.0`.

---

#### Changing the target compute graph

By default, all DOPPLER configs use `compute_graph/chain_mm_112nodes.txt`. To target a different graph, either edit `compute_graph_path` in the JSON config, or pass the flag on the command line:

```bash
python main.py --config_path config_files/run_doppler/imitation_learning.json \
               --compute_graph_path compute_graph/transformer_block_215nodes.txt
```

Alternatively, use the provided shell script which patches the config on-the-fly using `jq`:

```bash
# Edit COMPUTE_GRAPH_PATH and BASE_CONFIG inside the script, then:
bash shell_run_experiments.sh
```

> **Note:** GDP uses a different graph feature encoding. Use the `gdp_`-prefixed graph files (e.g., `compute_graph/gdp_ffnn_192nodes.txt`) when running the GDP baseline.

---

### Baselines

All baselines share the same `main.py` entry point and support the same `--compute_graph_path` override.

**PlaceTo** ([Mirhoseini et al., 2019](https://arxiv.org/pdf/1906.08879)):

PlaceTo uses a single graph-embedding policy trained end-to-end with policy gradient. It assigns devices one node at a time following a fixed topological order, without a separate node-selection policy.

```bash
python main.py --config_path config_files/run_placeto/imitation_learning.json
```

**GDP** ([Addanki et al., 2019](https://arxiv.org/pdf/1910.01578)):

GDP uses a TransformerXL + GraphSAGE architecture and assigns all nodes in a single forward pass. Its inputs use a different feature encoding; use the `gdp_`-prefixed compute graph files.

```bash
python main.py --config_path config_files/run_gdp/imitation_learning.json
```

**Critical Path heuristic** ([Kwok & Ahmad, 1999](https://dl.acm.org/doi/abs/10.1145/344588.344618)):

Assigns operations to devices using only static critical path analysis — no learning, no simulator. Useful as a deterministic reference baseline. Both `use_node_policy_network` and `use_device_policy_network` are disabled so the critical path heuristic drives all decisions.

```bash
python main.py --config_path config_files/run_critical_path/run_critical_path.json
```

---

### Outputs and checkpoints

Each run creates a timestamped directory under `heuristic_log/`:

```
heuristic_log/<YYYY-MM-DD_HH-MM-SS>_<alg>/
├── args.txt                    # Full config used for this run
├── output.txt                  # Per-episode logs: reward, cost, device assignment, epsilon
├── node_policy_network.pt      # Latest SEL policy checkpoint (overwritten each episode)
└── device_policy_network.pt    # Latest PLC policy checkpoint (overwritten each episode)
```

To monitor training with Weights & Biases, add `--use_wandb` or set `"use_wandb": true` in the config. The following metrics are logged per episode: `running_time` (cost of current episode), `rewards`, and `running_time_avg` (cumulative average cost).

---

### Config options

Each JSON config corresponds to the following CLI arguments. All values can be overridden from the command line.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_device` | Number of devices (GPUs) | `4` |
| `num_episode` | Number of training episodes | `8000` |
| `learning_rate` | Initial learning rate (linearly decayed to 0) | `4e-4` |
| `node_epsilon` | Initial ε-greedy exploration rate for SEL policy (linearly decayed to 0) | `0.5` |
| `device_epsilon` | Initial ε-greedy exploration rate for PLC policy (linearly decayed to 0) | `0.5` |
| `il_weight` | Weight for imitation learning loss | `0.0` |
| `simulation_pg_weight` | Weight for simulator-based policy gradient loss | `0.0` |
| `real_sys_pg_weight` | Weight for real-system policy gradient loss | `1.0` |
| `entropy_weight` | Entropy regularization weight | `0.01` |
| `run_il` | Enable Stage 1: imitation learning | `false` |
| `run_sim_rl` | Enable Stage 2: simulation RL | `false` |
| `run_real_sys_rl` | Enable Stage 3: real system RL | `true` |
| `use_node_policy_network` | Use learned SEL policy (if `false`, falls back to critical path) | `true` |
| `use_device_policy_network` | Use learned PLC policy (if `false`, falls back to critical path) | `true` |
| `use_placeto` | Run PlaceTo baseline instead of DOPPLER | `false` |
| `use_gdp` | Run GDP baseline instead of DOPPLER | `false` |
| `compute_graph_path` | Path to computation graph file | `compute_graph/chain_mm_112nodes.txt` |
| `load_pretrain_node_policy_path` | Path to a pretrained SEL policy checkpoint to warm-start from | `""` |
| `load_pretrain_device_policy_path` | Path to a pretrained PLC policy checkpoint to warm-start from | `""` |
| `log_path` | Root directory for logs and checkpoints | `heuristic_log/` |
| `use_wandb` | Enable Weights & Biases logging | `false` |

### Loading pretrained models

Checkpoints are saved after every episode to `<log_path>/<timestamp>_<alg>/node_policy_network.pt` and `device_policy_network.pt`. To warm-start a later stage from a prior stage's checkpoint, set the full paths in the config:

```json
"load_pretrain_node_policy_path": "heuristic_log/2025-01-01_00-00-00_il/node_policy_network.pt",
"load_pretrain_device_policy_path": "heuristic_log/2025-01-01_00-00-00_il/device_policy_network.pt"
```

> **Note:** Checkpoint paths must be absolute or relative to the working directory from which `main.py` is launched.

## 📁 Repository Structure

```
.
├── main.py                     # Entry point; parses args or JSON config
├── run_placement.py            # Training loop for all methods
├── cp_utils.py                 # Critical path computation
├── utils.py                    # Graph feature extraction and preprocessing
├── compute_graph/              # Computation graph files (.txt)
├── config_files/               # JSON configs for each method and stage
│   ├── run_doppler/
│   ├── run_gdp/
│   ├── run_placeto/
│   └── run_critical_path/
├── scheduler/
│   ├── rl_agent.py             # DOPPLER agent (SEL + PLC policy networks)
│   ├── placeto_agent.py        # PlaceTo baseline agent
│   ├── gdp_agent.py            # GDP baseline agent (TransformerXL + GraphSAGE)
│   └── primative_nn.py         # Shared NN primitives
├── env/
│   ├── env.py                  # RL environment (state, step, legal actions)
│   ├── cost.py                 # Cost engines: SimPy simulator & real engine
│   └── .env.example            # Template for remote execution credentials
├── heuristic_log/              # Output logs and saved checkpoints
├── shell_run_experiments.sh    # Shell script to run experiments
└── rl_env.def                  # Singularity container definition
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{yao2026doppler,
  title     = {{DOPPLER}: Dual-Policy Learning for Device Assignment in Asynchronous Dataflow Graphs},
  author    = {Yao, Xinyu and Bourgeois, Daniel and Jain, Abhinav and Tang, Yuxin and Yao, Jiawen and Ding, Zhimin and Silva, Arlei and Jermaine, Chris},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2505.23131}
}
```
