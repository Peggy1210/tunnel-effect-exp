from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_layer_names(model: nn.Module) -> list:
    """
    Returns:
        List of layer names in the model that we want to analyze.
    """
    return model.get_layer_names()


@torch.no_grad()
def extract_layer_features(model: nn.Module, layer_name: str, loader: DataLoader, device: torch.device):
    """
    Returns:
        feats : (N, D) torch.Tensor on CPU
        labels : (N,) torch.Tensor on CPU
    """
    import gc
    model.eval()
    
    feats_list = []
    labels_list = []
    
    for x, y in loader:
        x = x.to(device)
        reps = model.get_layer_representations(x, layer_name)
        feats_list.append(reps.detach().cpu())
        labels_list.append(y.detach().cpu())
        del x, reps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    del feats_list, labels_list
    gc.collect()
    
    return feats, labels


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader,
                     device: torch.device):
    """
    Returns:
        feats_per_layer : dict[layer_name -> (N, D) torch.Tensor] on CPU
        labels          : (N,) torch.Tensor on CPU
    """
    model.eval()
    all_reps = None
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        reps = model.get_representations(x)   # list of (B, D) tensors
        if all_reps is None:
            all_reps = OrderedDict([(layer, []) for layer in reps])
        for layer, rep in reps.items():
            all_reps[layer].append(rep.detach().cpu())
        all_labels.append(y.detach().cpu())

    feats = {layer: torch.cat(reps, dim=0) for layer, reps in all_reps.items()}
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def probe_eval(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / max(total, 1)


def train_linear_probe(
    train_feats: torch.Tensor,
    test_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int = 10,
    batch_size: int = 512,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    d_in = train_feats.shape[1]

    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_feats, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    probe = LinearProbe(d_in, num_classes).to(device)
    optimizer   = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    probe.train()
    for _ in tqdm(range(epochs)):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    probe.eval()
    train_acc = probe_eval(probe, train_loader, device)
    test_acc = probe_eval(probe, test_loader, device)
    return train_acc, test_acc
