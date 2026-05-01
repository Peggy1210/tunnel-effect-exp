from dataclasses import dataclass
from dataset.common import DatasetBundle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Callable
import os
from pathlib import Path
import json
from models.muon import Muon


from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    epochs: int


def get_train_config(model_name: str, model: nn.Module, optimizer: str) -> TrainConfig:
    model_name = model_name.lower()
    if model_name.startswith("mlp"):
        if optimizer == "fullbatch_gd":
            epochs = 300
            opt = optim.SGD(model.parameters(), lr=0.005, momentum=0.0, weight_decay=0)
            sch = None
        elif optimizer == "sgd":
            epochs = 200
            opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.0, weight_decay=0)
            sch = None
        elif optimizer == "sgd_momentum":
            epochs = 200
            opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
            sch = None
        elif optimizer == "adam":
            epochs = 200
            opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif optimizer == "adamw":
            epochs = 200
            opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif optimizer == "muon":
            epochs = 200
            opt = Muon(model.parameters(), lr=0.0005, momentum=0.9, ns_steps=5, adamw_lr=1e-4, adamw_wd=1e-3)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer} for MLP")
    elif model_name.startswith("vgg"):
        if optimizer == "fullbatch_gd":
            epochs = 300
            opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0)
            sch = None
        elif optimizer == "sgd":
            epochs = 160
            opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=5e-4)
            sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 120], gamma=0.1)
        elif optimizer == "sgd_momentum":
            epochs = 160
            opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 120], gamma=0.1)
        elif optimizer == "adam":
            epochs = 200
            opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif optimizer == "adamw":
            epochs = 200
            opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif optimizer == "muon":
            epochs = 200
            opt = Muon(model.parameters(), lr=0.005, momentum=0.95, ns_steps=5, adamw_lr=3e-4, adamw_wd=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer} for VGG")
    elif model_name.startswith("resnet"):
        if optimizer == "fullbatch_gd":
            epochs = 300
            opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0)
            sch = None
        elif optimizer == "sgd":
            epochs = 160
            opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=5e-4)
            sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 120], gamma=0.1)
        elif optimizer == "sgd_momentum":
            epochs = 160
            opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 120], gamma=0.1)
        elif optimizer == "adam":
            epochs = 200
            opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif optimizer == "adamw":
            epochs = 200
            opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif optimizer == "muon":
            epochs = 200
            opt = Muon(model.parameters(), lr=0.005, momentum=0.95, ns_steps=5, adamw_lr=3e-4, adamw_wd=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer} for ResNet")

    return TrainConfig(optimizer=opt, scheduler=sch, epochs=epochs)

class Trainer:
    """Trainer class for neural networks."""

    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        optimizer: str = 'sgd_momentum',
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.optimizer_name = optimizer
        self.model = model
        self.device = device
        self.model = self.model.to(device)

        train_cfg = get_train_config(model_name, model, optimizer)
        self.optimizer = train_cfg.optimizer
        self.scheduler = train_cfg.scheduler
        self.epochs = train_cfg.epochs

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        self.train_history = {"loss": [], "accuracy": []}
        self.test_history = {"loss": [], "accuracy": []}

    def train(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        
        n_accum = len(train_loader) if self.optimizer_name == "fullbatch_gd" else 1
        self.optimizer.zero_grad()

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            outputs = self.model(x)
            raw_loss = self.criterion(outputs, y)
            loss = raw_loss / n_accum
            loss.backward()

            # Backward pass
            if self.optimizer_name != "fullbatch_gd":
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Metrics
            total_loss += raw_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total

        if self.optimizer_name == "fullbatch_gd":
            self.optimizer.step()
            self.optimizer.zero_grad()

        return epoch_loss, epoch_acc

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on test set."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        epoch_loss = total_loss / len(test_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def fit(
        self,
        dataset: DatasetBundle,
        result_dir: Optional[str] = None,
        save_freq: int = 1,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the model for multiple epochs.

        Returns:
            Dictionary with training history
        """
        train_loader = dataset.train_loader
        test_loader = dataset.test_loader

        result_dir = Path(result_dir) if result_dir else None
        if result_dir:
            result_dir.mkdir(exist_ok=True, parents=True)

        # best_acc = -1.0
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)

            self.train_history["loss"].append(train_loss)
            self.train_history["accuracy"].append(train_acc)
            self.test_history["loss"].append(test_loss)
            self.test_history["accuracy"].append(test_acc)
            if verbose: print(f"Epoch [{epoch + 1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            if self.scheduler:
                self.scheduler.step()

            # # Save best checkpoint
            # if result_dir and test_acc > best_acc:
            #     best_acc = test_acc
            #     ckpt_path = result_dir / f"best.pt"
            #     self.save_checkpoint(ckpt_path)

            # Save checkpoint
            if result_dir and (epoch + 1) % save_freq == 0:
                ckpt_path = result_dir / f"epoch{epoch + 1}.pt"
                self.save_checkpoint(ckpt_path)

        # Save final checkpoint
        if result_dir:
            ckpt_path = result_dir / f"epoch{epoch}.pt"
            self.save_checkpoint(ckpt_path)

        return {
            "train_history": self.train_history,
            "test_history": self.test_history,
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_history": self.train_history,
                "test_history": self.test_history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_history = checkpoint["train_history"]
        self.test_history = checkpoint["test_history"]

    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))