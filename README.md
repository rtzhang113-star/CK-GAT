# CK-GAT

This repository contains the core implementation of **CK-GAT: Context-Aware Graph Attention Network with Asymmetric KAN for Blockchain Service Reliability Prediction**.

CK-GAT combines context-as-node graph construction, dual-branch graph attention, an asymmetric KAN-based multi-task prediction head, and GradNorm dynamic task weighting for sparse BaaS reliability prediction.

## Repository structure

```text
.
|-- config.py                         # Command-line configuration
|-- run_centralized_gnn.py            # Training and evaluation entry point
|-- core_utils/                       # Training, logging, early stopping, and metrics
|-- data_processing/                  # Pair-level splitting and QoS preprocessing
|-- graph_construction/               # Context-aware user and service graph construction
|-- models/                           # GAT, GCN-ablation, MLP, and KAN model components
`-- dataset/                          # Local dataset location (data not redistributed)
```

Experimental outputs, trained checkpoints, logs, figures, and result tables are intentionally excluded from this repository.

## Environment

The reference implementation uses Python 3.10, PyTorch 2.5.1, and CUDA 11.8. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA environment when necessary.

## Dataset preparation

The experiments use processed requester-peer QoS matrices derived from the H-BRP dataset. The original dataset is not redistributed in this repository. For dataset availability, access, and preprocessing instructions, please refer to the dataset source paper and its accompanying official repository:

- P. Zheng, Z. Zheng, and L. Chen, "Selecting reliable blockchain peers via hybrid blockchain reliability prediction," *IET Software*, vol. 17, no. 4, pp. 362--377, 2023. [https://doi.org/10.1049/sfw2.12118](https://doi.org/10.1049/sfw2.12118)
- Original H-BRP data and implementation repository: [https://github.com/InPlusLab/BlockchainReliabilityPrediction](https://github.com/InPlusLab/BlockchainReliabilityPrediction)

Place the following processed files in `dataset/` before running the code:

```text
SuccessRate{suffix}.csv
rightBlock{suffix}.csv
recentHeight{suffix}.csv
roundtripTime{suffix}.csv
ClientWithCTX.csv
PeerWithCTX.csv
```

For example, Setting 2 uses the suffix `_12_1000`, whereas Setting 4 uses `_100_5000`. The context files must contain the columns `as`, `country`, `isp`, and `timezone`.

The preprocessing code performs a requester-peer-pair-level train/validation/test split shared by all four QoS targets. SR, RB, and RH remain on their original `[0, 1]` scales. RTT is transformed with `log(1 + RTT)` and scaled using the training-set maximum only; the same factor is then applied to validation and test data.

## Running CK-GAT

The following command runs the multi-task CK-GAT configuration with KAN and GradNorm:

```bash
python run_centralized_gnn.py \
  --dataset_path dataset \
  --metric_file_suffix _12_1000 \
  --train_density 0.01 \
  --valid_density 0.30 \
  --task_mode multi \
  --interaction_type kan \
  --use_gradnorm True \
  --dimension 256 \
  --heads 4 \
  --kan_grid_size 3 \
  --kan_spline_order 2 \
  --optimizer AdamW \
  --lr 0.001 \
  --batch_size 128 \
  --epochs 200 \
  --enable_early_stop True \
  --early_stop_patience 15 \
  --random_seed 3407 \
  --rounds 5
```

Use `--train_density` values from `0.01` to `0.05` to reproduce the evaluated sparsity levels. Use the same seed and split settings when comparing model variants.

## Notes

- Context-aware graphs use only static contextual attributes and do not use QoS target labels.
- The test set is evaluated only after the checkpoint with the best validation SR MAE has been selected.
- The code automatically falls back to CPU when CUDA is unavailable.
