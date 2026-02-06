# NADES: Node-level Differentially Private Graph Knowledge Distillation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Differentially Private Graph Knowledge Distillation for Fraud Detection via Node-Adaptive Ensemble Soft Supervision with Dirichlet Perturbation



## 🎯 Overview

NADES (Node-level Differentially Private Graph Knowledge Distillation) is a framework that enables privacy-preserving graph neural network training for fraud detection tasks. The framework employs:

- **Bounded Participation Constraint (BPC)**: Ensures each private node participates in at most `s` teacher subgraphs
- **Bucket-conditioned Teacher Specificity Vectors (TSV)**: Captures teacher reliability profiles across different public context buckets
- **Node Hybrid Vector (NHV)**: Constructs node representations as `q_v = [log(1+deg_pub(v)), ||x_v||_2, h_v]` combining structural activity, feature strength, and student encoder embeddings
- **Attention-based Teacher Ensemble (ABTE)**: Adaptively aggregates teacher predictions based on node characteristics using cosine similarity between NHV and bucket-conditioned TSV
- **Dirichlet Perturbation Mechanism**: Provides node-level differential privacy guarantees


## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+
- PyTorch Geometric 2.0+

### Install Dependencies

```bash
# Clone the repository
git https://github.com/Gethin120/NADES
cd DPHSF/experiment/NADES

# Install required packages
pip install torch torch-geometric
pip install numpy scipy scikit-learn
pip install pyyaml
pip install cugraph  # For GPU-accelerated graph operations (optional)
```

## 🏃 Quick Start

### Basic Usage

```bash
# Run NADES on Yelp dataset with epsilon=1.0
python main.py --model NADES --dataset yelp --epsilon 1.0

# Run with custom configuration
python main.py \
    --model NADES \
    --dataset elliptic \
    --epsilon 0.5 \
    --num_partitions 10 \
    --max_overlap_per_node 3 \
    --num_queries 200
```

### Using Configuration Files

Configuration files are located in `config/` directory:

```bash
python main.py --model NADES --dataset yelp
# Automatically loads config/NADES_yelp.yaml
```

## 📖 Usage

### Command Line Arguments

#### Model Selection
- `--model`: Model name (`NADES`, `PATE`, `ScalePATE`, `PrivGNN`, `BGNN`)
- `--dataset`: Dataset name (`yelp`, `amazon`, `elliptic`, `comp`, `dgraphfin`)

#### Privacy Parameters
- `--epsilon`: Privacy budget (ε) for differential privacy
- `--delta`: Privacy parameter (δ), typically 1e-5
- `--privacy_ratio`: Ratio of private vs public nodes (default: 0.7)

#### Model Architecture
- `--num_layers`: Number of GNN layers (default: 2)
- `--hidden_channels`: Hidden dimension (default: 128)
- `--out_channels`: Output classes (default: 2)
- `--dropout`: Dropout rate (default: 0.2)

#### Partitioning Parameters
- `--num_partitions`: Number of teacher subgraphs (S)
- `--max_overlap_per_node`: Maximum participation per node (s)
- `--partition_method`: Partitioning strategy (`D` for default)

#### Training Parameters
- `--epochs`: Training epochs
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 1024)
- `--num_queries`: Number of DP queries (default: 200)

### Example Scripts

See `run.sh` for example configurations:

```bash
# Run with different epsilon values
bash run.sh
```

## 📊 Datasets

NADES supports the following datasets:

| Dataset | Description | Nodes | Edges | Features |
|---------|-------------|-------|-------|----------|
| **YelpChi** | Review fraud detection | ~45K | ~3.8M | 32 |
| **Amazon** | User fraud detection | ~12K | ~4.7M | 25 |
| **Elliptic** | Bitcoin transaction fraud | ~200K | ~234K | 166 |

### Dataset Preparation

Datasets are automatically downloaded and preprocessed. Ensure you have sufficient disk space and network connectivity.

## ⚙️ Configuration

Configuration files use YAML format and are located in `config/`:

```yaml
# Example: config/NADES_yelp.yaml
model: NADES
dataset: yelp
epsilon: 1.0
delta: 1e-5
num_partitions: 10
max_overlap_per_node: 3
num_queries: 200
hidden_channels: 128
num_layers: 2
batch_size: 1024
lr: 0.001
```

## 🏗️ Architecture

NADES consists of four main components:

### 1. BPC-Constrained Teacher Subgraph Construction
- Partitions private graph into `S` overlapping subgraphs
- Ensures each node participates in at most `s` subgraphs
- Supports affinity-guided, random, or hash-based partitioning

### 2. Teacher Training & TSV Construction
- Each teacher trains independently on its subgraph
- Computes bucket-conditioned Teacher Specificity Vectors (TSV)
- TSV captures teacher reliability across 4 public context buckets

### 3. ABTE Coordinator
- Constructs Node Hybrid Vector (NHV): `q_v = [log(1+deg_pub(v)), ||x_v||_2, h_v]`
  - `log(1+deg_pub(v))`: Structural activity (logarithmic degree in public graph)
  - `||x_v||_2`: Feature strength (L2 norm of node features)
  - `h_v`: Student encoder embedding (from pre-trained GraphMAE encoder)
- Uses cosine similarity between NHV and bucket-conditioned TSV to compute attention weights
- Adaptively aggregates teacher predictions based on node-specific characteristics

### 4. Dirichlet Mechanism & Privacy Accounting
- Applies Dirichlet noise to aggregated predictions
- Uses RDP (Rényi Differential Privacy) accounting
- Provides node-level (ε, δ)-DP guarantees



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
