import torch
import torch.optim as optim

def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Iterative Newton-Schulz orthogonalisation.
    Converges to the orthogonal factor of G = U S V^T (i.e. U V^T).
    Works on (m, n) matrices; handles tall and wide by transposing.
    """
    assert G.ndim == 2
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T
    # Normalize
    G = G / (G.norm() + eps)
    # Quintic iteration coefficients (Kosson et al.)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    for _ in range(steps):
        A  = X @ X.T
        X  = a * X + b * (A @ X) + c * (A @ A @ X)
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon — Momentum Orthogonalized by Newton-Schulz.

    For 2-D+ params (Conv, Linear weights): applies NS orthogonalisation to
    the Nesterov momentum buffer before the weight update.
    For 1-D params (biases, BN) and the final classifier: falls back to AdamW.

    Args:
        params       : model parameters (pass model.parameters())
        lr           : learning rate for Muon updates
        momentum     : momentum coefficient (default 0.95)
        ns_steps     : Newton-Schulz iterations (default 5)
        adamw_lr     : LR for fallback AdamW (default 3e-4)
        adamw_wd     : weight decay for fallback AdamW (default 0)
    """

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5,
                 adamw_lr=3e-4, adamw_wd=0.0):
        # Split params into Muon group (2-D+) and AdamW fallback (1-D)
        param_list = list(params)
        muon_params  = [p for p in param_list if p.ndim >= 2]
        adamw_params = [p for p in param_list if p.ndim <  2]

        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        adamw_lr=adamw_lr, adamw_wd=adamw_wd)
        super().__init__([
            {"params": muon_params,  "use_muon": True},
            {"params": adamw_params, "use_muon": False},
        ], defaults)

        # Internal AdamW for fallback params
        if adamw_params:
            self._adamw = torch.optim.AdamW(
                adamw_params, lr=adamw_lr, weight_decay=adamw_wd)
        else:
            self._adamw = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        # ── AdamW fallback step
        if self._adamw is not None:
            self._adamw.step()

        # ── Muon step for 2-D+ params
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            lr       = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Initialise / update Nesterov momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)                   # m = β*m + g
                g_nesterov = g.add(buf, alpha=momentum)      # g + β*m  (Nesterov)

                # Reshape to 2-D for Newton-Schulz
                orig_shape = g_nesterov.shape
                g2d = g_nesterov.reshape(orig_shape[0], -1)

                # Orthogonalise
                g_orth = _newton_schulz(g2d.float(), steps=ns_steps)
                g_orth = g_orth.reshape(orig_shape).to(p.dtype)

                # Scale so the update has the same RMS as a unit-norm grad
                scale = max(1, g2d.shape[0] / g2d.shape[1]) ** 0.5
                p.add_(g_orth, alpha=-lr * scale)

        return loss

    def state_dict(self):
        sd = super().state_dict()
        if self._adamw is not None:
            sd['_adamw'] = self._adamw.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        adamw_sd = state_dict.pop('_adamw', None)
        super().load_state_dict(state_dict)
        if adamw_sd and self._adamw is not None:
            self._adamw.load_state_dict(adamw_sd)
