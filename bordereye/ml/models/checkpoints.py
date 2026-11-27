"""Device selection for the BevNetDeep workflows."""
from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or select the fastest available backend."""
    requested = "auto" if device is None else str(device).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device auto or cpu")
    if target.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available; use --device auto or cpu")
    return target
