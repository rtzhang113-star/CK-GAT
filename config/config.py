import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="CK-GAT training and evaluation")

    parser.add_argument("--dataset_path", type=str, default="dataset/")
    parser.add_argument("--metric_file_suffix", type=str, default="_100_5000")
    parser.add_argument("--client_ctx_file", type=str, default="ClientWithCTX.csv")
    parser.add_argument("--peer_ctx_file", type=str, default="PeerWithCTX.csv")
    parser.add_argument("--train_density", type=float, default=0.01)
    parser.add_argument("--valid_density", type=float, default=0.3)

    parser.add_argument(
        "--ablation_no_gnn", type=bool, default=False, help="Ablation: remove the GNN backbone"
    )
    parser.add_argument(
        "--ablation_no_context",
        type=bool,
        default=False,
        help="Ablation: remove context nodes and edges",
    )
    parser.add_argument(
        "--ablation_use_gcn", type=bool, default=False, help="Ablation: replace GAT with GCN"
    )
    parser.add_argument(
        "--ablation_single_layer",
        type=bool,
        default=False,
        help="Ablation: use one GAT/GCN layer instead of two",
    )

    parser.add_argument(
        "--inner",
        type=bool,
        default=False,
        help="Use element-wise products instead of concatenation for interaction features",
    )

    parser.add_argument("--task_mode", type=str, default="multi", choices=["single", "multi"])

    parser.add_argument(
        "--use_gradnorm", type=bool, default=True, help="Enable GradNorm dynamic task weighting"
    )
    parser.add_argument(
        "--gradnorm_beta", type=float, default=1.0, help="GradNorm balancing exponent"
    )
    parser.add_argument(
        "--weights_lr", type=float, default=0.001, help="Learning rate for GradNorm task weights"
    )

    parser.add_argument("--dimension", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--heads", type=int, default=1, help="Number of GAT attention heads")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout rate")
    parser.add_argument("--alpha", type=float, default=0.1, help="LeakyReLU negative slope")

    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs")
    parser.add_argument(
        "--lr", type=float, default=0.0014, help="Learning rate for embeddings and prediction heads"
    )
    parser.add_argument(
        "--att_lr", type=float, default=0.001, help="Learning rate for the graph encoder"
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.00035,
        help="Weight decay for embeddings and prediction heads",
    )
    parser.add_argument(
        "--att_decay", type=float, default=1e-4, help="Weight decay for the graph encoder"
    )
    parser.add_argument("--enable_early_stop", type=bool, default=True)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument(
        "--loss_func",
        type=str,
        default="MSELoss",
        choices=["L1Loss", "MSELoss", "CauchyLoss", "SmoothL1Loss"],
        help="Loss function",
    )
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument("--max_gradient", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument(
        "--lr_step", type=int, default=50, help="Step size for the learning-rate scheduler"
    )
    parser.add_argument(
        "--loss_weights",
        nargs=4,
        type=float,
        default=[1.0, 1.0, 1.0, 1.0],
        help="Static loss weights for SR, RB, RH, and RTT",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of learning-rate warm-up epochs"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning-rate scheduler",
    )

    parser.add_argument("--devices", type=str, default="cuda")
    parser.add_argument("--random_seed", type=int, default=3407)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--verbose", type=bool, default=False, help="Enable detailed logging")
    parser.add_argument(
        "--log_every_n_epochs", type=int, default=100, help="Epoch interval between log messages"
    )

    parser.add_argument(
        "--client_feature_cols",
        nargs="+",
        default=["as", "country", "isp", "timezone"],
        help="Requester context columns",
    )
    parser.add_argument(
        "--peer_feature_cols",
        nargs="+",
        default=["as", "country", "isp", "timezone"],
        help="Peer context columns",
    )

    parser.add_argument(
        "--gat_impl",
        type=str,
        default="dgl",
        choices=["spgat", "dgl"],
        help="GAT implementation: custom sparse GAT or DGL GATConv",
    )

    parser.add_argument("--interaction_type", type=str, default="mlp", choices=["mlp", "kan"])

    parser.add_argument(
        "--head_input_norm",
        type=bool,
        default=True,
        help="Apply LayerNorm before the MLP or KAN prediction heads",
    )

    parser.add_argument(
        "--head_input_dropout",
        type=float,
        default=0.25,
        help="Dropout rate before the prediction heads",
    )

    parser.add_argument(
        "--sr_head_hidden_dims",
        nargs="+",
        type=int,
        default=[64],
        help="Hidden dimensions of the SR-specific MLP or KAN head, for example: 128 64",
    )

    parser.add_argument(
        "--other_tasks_head_hidden_dims",
        nargs="+",
        type=int,
        default=[64],
        help="Hidden dimensions of the shared RB/RH/RTT MLP or KAN head, for example: 64",
    )

    parser.add_argument("--kan_grid_size", type=int, default=5, help="KAN spline grid size")
    parser.add_argument("--kan_spline_order", type=int, default=3, help="KAN spline order")

    args = parser.parse_args()

    actual_test_density = 1.0 - args.train_density - args.valid_density
    print(f"Test Size: {actual_test_density:.3f}")

    if args.devices == "cuda" and not torch.cuda.is_available():
        args.devices = "cpu"

    return args
