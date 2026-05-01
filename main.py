import argparse
import json
import random
from pathlib import Path

from analyze.analyzer import Analyzer
from models import build_model
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm

from dataset import build_dataset, get_num_classes
from utils import set_seed, get_device
from models.trainer import Trainer


def main(args):
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")
    print(f"Experiment: {args.experiment}")

    # Create result directory
    folder_name = f"{args.model_name}_{args.dataset}_{args.optimizer}" if not args.experiment else args.experiment
    result_dir = Path(args.result_dir) / folder_name
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {result_dir}")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    dataset = build_dataset(args.dataset, args.data_dir, args.batch_size, args.num_workers, args.max_train_data, args.max_test_data)
    num_classes = get_num_classes(args.dataset)
    print(f"Number of classes: {num_classes}")

    # Build model
    model = build_model(args.model_name, num_classes=num_classes).to(device)
    print(f"Model architecture: {args.model_name}")
    # summary(model, input_size=(1, 3, 32, 32))

    trainer = Trainer(args.model_name, model, args.optimizer, device)
    
    if args.pretrained_model:
        trainer.load_checkpoint(args.pretrained_model)
    else:
        raise ValueError("Pretrained model checkpoint is required for analysis. Please provide --pretrained_model argument.")

    # Start analysis
    print("\nStarting tunnel effect analysis...")
    analyzer = Analyzer(model, device)
    analysis_results = analyzer.analyze(dataset, num_classes=num_classes, probe=args.probe, output_dir=result_dir / "analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tunnel effect analysis")
    parser.add_argument(
        "--model_name",
        type=str,
        default="vgg19",
        choices=["vgg19", "resnet34", "mlp12"],
        help="Model architecture",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd_momentum",
        choices=["fullbatch_gd", "sgd", "sgd_momentum", "adam", "adamw", "muon"],
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "cinic10"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store/load data",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Frequency of saving checkpoints",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--max_train_data",
        "--max_train",
        type=int,
        default=None,
        help="Maximum number of training samples to use (for debugging)",
    )
    parser.add_argument(
        "--max_test_data",
        "--max_test",
        type=int,
        default=None,
        help="Maximum number of test samples to use (for debugging)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the model default number of training epochs",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to a pretrained model checkpoint to load (skips training)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Enable linear probe training",
    )

    args = parser.parse_args()
    main(args)
