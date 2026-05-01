from typing import Dict, Tuple

import numpy as np
import torch


@torch.no_grad()
def compute_numerical_rank(features: torch.Tensor, threshold_ratio: float = 1e-3, device: torch.device = None, n_random_features: int = None) -> int:
    """
    Numerical rank of the sample covariance matrix.
    
    Args:
        features: Input tensor (N, D)
        threshold_ratio: Threshold for rank computation (ratio of largest singular value)
        device: Device to compute on
        n_random_features: If set, randomly select this many features before computing rank
                          (e.g., 8000 as mentioned in the paper)
    """
    x = features.float()
    if device is not None:
        x = x.to(device)

    x = x - x.mean(dim=0, keepdim=True)

    n_samples, n_features = x.shape
    
    # Randomly select a subset of features if specified
    if n_random_features is not None and n_random_features < n_features:
        feature_indices = torch.randperm(n_features, device=x.device)[:n_random_features]
        x = x[:, feature_indices]
    
    if x.shape[1] > n_samples:
        cov = (x.T @ x) / max(n_samples - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov)
        singular_vals = torch.sqrt(torch.clamp(eigvals, min=0)) * (n_samples - 1) ** 0.5
        singular_vals = singular_vals.sort(descending=True)[0]
    else:
        try:
            singular_vals = torch.linalg.svdvals(x)
        except RuntimeError:
            singular_vals = torch.from_numpy(
                np.linalg.svd(x.cpu().numpy(), compute_uv=False)
            ).to(device)

    if singular_vals.numel() == 0:
        return 0

    threshold = singular_vals[0].item() * threshold_ratio
    rank = int((singular_vals > threshold).sum().item())
    return rank


@torch.no_grad()
def compute_variance(features: torch.Tensor, labels: torch.Tensor, device: torch.device = None) -> Tuple[float, float]:
    """
    Returns (inter_class_variance, intra_class_variance).
    """
    x = features.float()
    if device is not None:
        x = x.to(device)
    classes = labels.unique()
    global_mean = x.mean(dim=0)

    inter_sum = 0.0
    intra_sum = 0.0
    total = x.shape[0]

    for c in classes:
        mask = labels == c
        xc = x[mask]
        nc = xc.shape[0]
        centroid = xc.mean(dim=0)

        # Inter: n_c * ||centroid - global_mean||^2
        inter_sum += nc * ((centroid - global_mean) ** 2).sum().item()

        # Intra: sum ||x - centroid||^2 for x in class c
        intra_sum += ((xc - centroid) ** 2).sum().item()

    inter_var = inter_sum / total
    intra_var = intra_sum / total
    return inter_var, intra_var


def _gram_linear(x: torch.Tensor) -> torch.Tensor:
    """Linear kernel Gram matrix."""
    return x @ x.T


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Double-center a Gram matrix."""
    n = K.shape[0]
    ones = torch.ones(n, 1, device=K.device, dtype=K.dtype)
    K = K - ones @ (ones.T @ K) / n
    K = K - (K @ ones) @ ones.T / n
    return K

@torch.no_grad()
def compute_cka(x: torch.Tensor, y: torch.Tensor, device: torch.device = None) -> float:
    x = x.float()
    y = y.float()

    if device is not None:
        x = x.to(device)
        y = y.to(device)

    Kx = _center_gram(_gram_linear(x))
    Ky = _center_gram(_gram_linear(y))

    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()

    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 0.0
    return (hsic_xy / denom).item()

@torch.no_grad()
def compute_weight_norm(w1: torch.Tensor, w2: torch.Tensor, device: torch.device = None) -> float:
    w1 = w1.float()
    w2 = w2.float()
    
    if device is not None:
        w1 = w1.to(device)
        w2 = w2.to(device)

    diff = (w1 - w2).flatten()
    nm = diff.numel()
    return diff.norm(2).item() / (nm ** 0.5)


@torch.no_grad()
def compute_energy(features: torch.Tensor, topk: int=None, device: torch.device = None) -> torch.Tensor:
    """Return singular values of centered features (descending)."""
    if features.ndim > 2:
        features = features.flatten(start_dim=1)
    x = features.float()
    if device is not None:
        x = x.to(device)
    x = x - x.mean(dim=0, keepdim=True)
    try:
        sv = torch.linalg.svdvals(x)
    except RuntimeError:
        sv = torch.from_numpy(np.linalg.svd(x.cpu().numpy(), compute_uv=False))
    # sv = sv.sort(descending=True)[0]

    energy = sv ** 2
    total_energy = energy.sum()
    if topk is not None:
        energy = energy[:topk]
    return sv / sv.sum(), (energy / total_energy)


# @torch.no_grad()
# def compute_eRank(features: torch.Tensor) -> float:
#     """Compute Effective Rank: exp(H(p)) where p are normalized singular values.
#     Returns eRank scalar.
#     """
#     sv = singular_values(features)
#     if sv.numel() == 0:
#         return 0.0
#     sv = sv.clamp(min=1e-12)
#     p = sv / sv.sum()
#     H = - (p * torch.log(p)).sum().item()
#     return float(np.exp(H))


def _flatten_tensor_list(tensors):
    return torch.cat([t.flatten() for t in tensors]).float()


def compute_gwa(model: torch.nn.Module, loader, device: torch.device, loss_fn=None, max_batches: int = None):
    """Compute gradient-weight alignment (cosine) per weight-containing layer.

    Returns dict {param_name: cosine_similarity} for parameters that have 'weight' in the name.
    This computes the average gradient across the loader (batch-weighted).
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()
    # Map parameter names to zero tensors for accumulating gradients
    param_map = {name: p for name, p in model.named_parameters() if 'weight' in name}
    acc_grads = {name: torch.zeros_like(p.data, device=device) for name, p in param_map.items()}
    total_samples = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        batch_size = x.shape[0]
        total_samples += batch_size
        for name, p in param_map.items():
            if p.grad is not None:
                acc_grads[name] += p.grad.detach() * batch_size

    if total_samples == 0:
        return {name: 0.0 for name in param_map.keys()}

    # average gradients
    for name in acc_grads:
        acc_grads[name] = acc_grads[name] / float(total_samples)

    # compute cosine similarity per param (weight)
    gwa = {}
    for name, p in param_map.items():
        w = p.data.detach().to(device).float()
        g = acc_grads[name].to(device).float()
        w_flat = w.flatten()
        g_flat = g.flatten()
        denom = (w_flat.norm() * g_flat.norm()).item()
        if denom == 0:
            gwa[name] = 0.0
        else:
            gwa[name] = float((w_flat @ g_flat).item() / denom)

    model.zero_grad()
    return gwa
