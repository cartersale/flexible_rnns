# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs" (Nature Neuroscience, 2024). The code trains recurrent neural network (RNN) models on multiple cognitive tasks and analyzes the computational structure and dynamics they develop. The project investigates how networks share dynamical motifs across tasks to achieve flexible multitask computation.

Paper: https://www.nature.com/articles/s41593-024-01668-6

## Environment Setup

This project uses Python 3.10 and TensorFlow 2 with TF1 compatibility mode.

```bash
# Create conda environment
conda create --name flex_mult python=3.10
conda activate flex_mult

# Install package in editable mode
pip install -e .
```

### Configuration

The `.env` file contains paths that need to be configured:
- `HOME_DIR`: Path to this repository
- `DATA_DIR`: Path to directory where processed data is stored

The code loads this via `python-dotenv` in `task.py` and other modules. Update `.env` with your local paths before running experiments.

### Data Access

Trained networks and processed data (including fixed point locations) are available at the Allen Institute database:
- Trained networks: 'multitask_shared' bucket
- Processed data: 'multitask_processed' bucket
- URL: https://open.quiltdata.com/b/aind-trained-networks/tree/

## Architecture

### Module Organization

**`stepnet/`**: Core module for training RNNs with random initialization
- `network.py`: RNN model definition, custom RNN cells (LeakyRNN, etc.), performance metrics
- `task.py`: Task definitions for cognitive tasks (15 tasks total in 'all' ruleset), trial generation
- `train.py`: Main training loop, hyperparameter defaults, model checkpointing
- `tools.py`: Utilities for model loading, data processing
- `general_model_train.py`: Example training script with hyperparameter specifications
- `analysis/`: Analysis modules for trained networks (clustering, dimensionality, variance, etc.)

**`transfer_learn/`**: Transfer learning experiments starting from pretrained networks
- Similar structure to `stepnet/` but with transfer learning capabilities
- `experiment.py`, `ruleset.py`: Transfer learning configuration and management

**`utils/`**: Shared utility functions used across the codebase

**`analysis/`**: Figure generation scripts organized by manuscript figures (fig1/, fig2/, etc.)
- `multitask_network_analyses.ipynb`: Main analysis notebook
- Each subdirectory contains code to reproduce specific manuscript figures

### TensorFlow Usage

The codebase uses TensorFlow 2 with TF1 compatibility mode:
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

All code is written for TF1 API patterns (Sessions, placeholders, etc.) running under TF2.

### Key Data Structures

**Task Configuration (`task.py`)**:
- `rules_dict`: Defines task sets (e.g., 'all' has 15 tasks including fdgo, delaygo, delaydm1, etc.)
- `Trial` class: Represents batches of task trials with inputs/outputs formatted for RNN training
- Tasks involve working memory, decision-making, and sensorimotor transformations

**Network Model (`network.py`)**:
- Custom RNN cells: LeakyRNN (vanilla), LeakyGRU, LSTM
- Activation functions: relu, softplus, tanh
- Weight initialization: 'randortho', 'randgauss', 'diag'
- Performance evaluation via `get_perf()` function

**Hyperparameters (`train.py`)**:
- Network size: `n_rnn` (typically 128 or 256 units)
- Regularization: `l1_h`, `l2_h` (activity), `l1_weight`, `l2_weight` (weights)
- Noise: `sigma_rec` (recurrent), `sigma_x` (input)
- Learning: `learning_rate`, `optimizer` (adam)

## Training Networks

Training scripts set hyperparameters and call `train.train()`:

```python
from train import train
from task import rules_dict

# Example: train softplus LeakyRNN on all 15 tasks
ruleset = 'all'
rule_trains = rules_dict[ruleset]

train(filedir='path/to/save',
      seed=0,
      max_steps=1e8,
      ruleset=ruleset,
      rule_trains=rule_trains,
      hp={'activation': 'softplus',
          'w_rec_init': 'diag',
          'n_rnn': 256,
          'l2_h': 1e-6,
          'l2_weight': 1e-6,
          'sigma_rec': 0.05,
          'sigma_x': 0.1,
          'rnn_type': 'LeakyRNN'})
```

Training saves checkpoints to `filedir` with model parameters and performance logs.

## Running Analyses

Analysis code typically:
1. Loads trained model checkpoints
2. Generates neural activity on task trials
3. Computes metrics (e.g., fixed points, variance partitioning, clustering)
4. Saves results to `results/` directory as JSON/pickle files
5. Generates figures (saved to `figs/`)

The `results/` directory contains pre-computed analysis outputs organized by network architecture and analysis type.

## Common Patterns

**Loading Models**: Use `tools.load_model()` or similar utilities to restore trained networks from checkpoints.

**Task Generation**:
```python
from task import generate_trials
trial = generate_trials(rule, hp, mode='test')
```

**Performance Evaluation**: Models perform ring attractor tasks where outputs encode fixation or angular responses. The `get_perf()` function compares network outputs to target locations.

**Result Storage**: Analysis scripts save results as JSON to `results/` with hierarchical naming (e.g., `results/signed_input_wl_paper/LeakyRNN_softplus_diag_256_all_0/...`).

## Development Notes

- The project is installable as the `flexmult` package via setup.py
- Dependencies in `requirements.txt` include a GitHub dependency on `recurrent-whisperer` for fixed point analysis
- The codebase follows the paper's terminology: "structural cells" refer to clustered units with shared task selectivity patterns
- Network architectures are identified by strings like `LeakyRNN_softplus_diag_256_all_0` encoding: RNN type, activation, initialization, size, ruleset, seed
