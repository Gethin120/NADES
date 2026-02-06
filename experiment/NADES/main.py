import warnings
import os
import sys
import torch
import yaml
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

# Ensure project root in sys.path BEFORE importing project modules
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.utils import set_seed  # noqa: E402
from methods import methods_nades  # noqa: E402
from dataset import get_NADES_dataset  # noqa: E402
from methods.base import FraudDetection  # noqa: E402

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# Ensure project root in sys.path to avoid ModuleNotFoundError
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main(args):
    set_seed(42)

    logger = logging.getLogger("main_train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    log_dir = CURRENT_DIR
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "experiment_nades.log"))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(
        "Starting training with method {} in dataset {}".format(
            args.model, args.dataset
        )
    )
    logger.info(args)
    data = get_NADES_dataset(args.dataset)
    in_channels = data.num_features
    metadata = args.dataset
    Method = methods_nades[args.model]
    torch.autograd.set_detect_anomaly(True)

    base_kwargs = {
        "detection_type": args.detection_type,
        "epochs": args.epochs,
        "num_neighbor": args.num_neighbors,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "in_channels": in_channels,
        "out_channels": args.out_channels,
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
        "logger": logger,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "max_grad_norm": args.grad_norm_max,
        "patience": args.early_stopping,
        "metadata": metadata,
    }

    all_args = vars(args)
    excluded_keys = set(base_kwargs.keys()) | {
        'model', 'dataset', 'early_stopping', 'grad_norm_max', 'num_neighbors'
    }

    param_mapping = {
        # BGNN
        'bgnn_alpha': 'alpha',
        'bgnn_gamma': 'gamma',
        'bgnn_temperature': 'temperature',
        'bgnn_boosting': 'boosting',
        'bgnn_temp': 'temp',
        'bgnn_dataset_name': 'dataset_name',
        # PrivGNN
        'privgnn_max_degree': 'max_degree',
        'privgnn_conv': 'conv',
        'num_teachers': 'num_teachers',  # PrivGNN uses multiple teachers
    }

    extra_kwargs = {}
    mapped_params = set()

    for k, v in all_args.items():
        if k in excluded_keys or v is None:
            continue

        param_name = param_mapping.get(k, k)

        if param_name in mapped_params:
            continue

        extra_kwargs[param_name] = v
        mapped_params.add(param_name)

    method_kwargs = {**base_kwargs, **extra_kwargs}
    method: FraudDetection = Method(**method_kwargs)
    method.fit(data)


def batch_size_type(value):
    """自定义类型转换函数：支持 int 或 'full' 字符串"""
    if isinstance(value, str):
        if value.lower() == "full":
            return "full"
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"batch_size 必须是整数或 'full'，得到: {value}"
            )
    elif isinstance(value, int):
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"batch_size 必须是整数或 'full'，得到: {type(value)}"
        )


def parse_args(args_init):
    model = args_init.model.split("-")[0]
    dataset = args_init.dataset
    root = "/home/workspace/gethin_learn/DPHSF/"
    yaml_file = root + "config/" + model + "_" + dataset + ".yaml"
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args = argparse.Namespace(**args)
    args.model = args_init.model
    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )
    # model arguments
    parser.add_argument(
        "--model", type=str, default="NADES", help="The model name. [NADES,PATE,ScalePATE,BGNN,PrivGNN]"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="The number of layers of encoder"
    )
    parser.add_argument("--hidden_channels", "-hs", type=int, default=128)
    parser.add_argument("--out_channels", "-o", type=int, default=2)
    # dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon",
        help="The dataset name. [yelp, amazon, comp, elliptic, dgraphfin]",
    )
    parser.add_argument(
        "--detection_type",
        type=str,
        default="txs",
        help="yelp:review, amazon:user, comp:company, elliptic:txs",
    )
    parser.add_argument(
        "--num_neighbors", type=int, default=3, help="Number of sampeld neighbors."
    )
    parser.add_argument(
        "--batch_size",
        type=batch_size_type,
        default=1024,
        help="Number of batchsize. [full, 1024]",
    )
    # training arguments
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability of feature privacy-preserving.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=1000,
        help="Number of epochs to downstream task.",
    )
    parser.add_argument(
        "--early_stopping", type=int, default=30, help="Early stopping patience."
    )
    # privacy arguments
    parser.add_argument(
        "--epsilon",
        type=float,
        default=30.0,
        help="eta of Dirichlet mechanism.[200.0]",
    )
    parser.add_argument(
        "--delta",
        "-delta",
        type=float,
        default=1e-5,
        help="Probabilistic parameters of breaking privacy protections.",
    )
    parser.add_argument(
        "--grad_norm_max",
        type=float,
        default=1.0,
        help="Gradient clipping max norm for privacy/stability.",
    )
    parser.add_argument(
        "--privacy_ratio", type=float, default=0.7,
        help="Ratio of private data "
    )
    parser.add_argument(
        "--partition_method", type=str, default="D", help="Partition method."
    )
    parser.add_argument(
        "--teacher_epochs", type=int, default=50,
        help="Teacher training epochs (NADES/PATE/ScalePATE)"
    )
    parser.add_argument(
        "--teacher_patience", type=int, default=20,
        help="Teacher early stopping patience (NADES/PATE/ScalePATE)"
    )
    # === NADES-specific parameters ===
    parser.add_argument(
        "--num_partitions", type=int, default=15,
        help="Number of partitions for NADES (default: 15)"
    )
    parser.add_argument(
        "--max_overlap_per_node",
        type=int,
        default=2,
        help="Max overlap per node.",
    )
    parser.add_argument(
        "--num_queries", type=int, default=200, help="Number of queries.[200, 400, 1000]"
    )
    parser.add_argument(
        "--consistency_weight", type=float, default=1.0, help="Consistency weight."
    )
    parser.add_argument(
        "--ssl_method", type=str, default="none",
        help="SSL method for NADES: [infomax, grace, graphmae, none]"
    )

    # === PATE/ScalePATE-specific parameters ===
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=5.0,
        help="Noise scale b for Laplacian mechanism in PATE. Controls privacy cost per query (epsilon ≈ 2/b).",
    )
    parser.add_argument(
        "--num_teachers", type=int, default=10,
        help="Number of teacher models (PATE/ScalePATE, default: 10 for PATE, 15 for ScalePATE)"
    )
    parser.add_argument(
        "--confidence_threshold", type=int, default=8,
        help="Confidence threshold T for ScalePATE (minimum vote count to accept query)"
    )
    parser.add_argument(
        "--gaussian_noise_scale", type=float, default=25.0,
        help="Gaussian noise scale σ for ScalePATE"
    )
    parser.add_argument(
        "--detection_epsilon", type=float, default=0.01,
        help="Privacy cost for detection queries in ScalePATE (rejection queries)"
    )
    # === BGNN-specific parameters ===
    parser.add_argument(
        "--bgnn_alpha", type=float, default=1.0,
        help="Alpha parameter for BGNN"
    )
    parser.add_argument(
        "--bgnn_gamma", type=float, default=1.0,
        help="Gamma parameter for BGNN"
    )
    parser.add_argument(
        "--bgnn_temperature", type=float, default=1.0,
        help="Temperature for BGNN distillation"
    )
    parser.add_argument(
        "--bgnn_boosting", action="store_true",
        help="Enable boosting for BGNN"
    )
    parser.add_argument(
        "--bgnn_temp", action="store_true",
        help="Use temperature for BGNN"
    )

    # === PrivGNN-specific parameters ===
    parser.add_argument(
        "--privgnn_max_degree", type=int, default=100,
        help="Max degree for PrivGNN"
    )
    parser.add_argument(
        "--privgnn_conv", type=str, default="sage",
        help="Convolution type for PrivGNN: [sage, gcn, gat]"
    )

    args = parser.parse_args()

    main(args)
