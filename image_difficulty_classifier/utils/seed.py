import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed: int
        The seed value to use.
    deterministic: bool
        If True, configures CUDA/cuDNN for deterministic behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def capture_rng_states() -> dict:
    """Capture RNG states to persist in checkpoints."""
    state: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(state: Optional[dict]) -> None:
    """Restore RNG states from a checkpoint payload."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    # Convert arbitrary containers to torch.ByteTensor where needed
    def _to_byte_tensor(x) -> torch.ByteTensor:
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.uint8:
                x = x.to(dtype=torch.uint8)
            return x  # type: ignore[return-value]
        try:
            return torch.tensor(x, dtype=torch.uint8)  # type: ignore[return-value]
        except Exception:
            # Fallback: create empty state to avoid crash
            return torch.empty(0, dtype=torch.uint8)  # type: ignore[return-value]

    if "torch_cpu" in state:
        cpu_state = _to_byte_tensor(state["torch_cpu"])  # type: ignore[arg-type]
        if cpu_state.numel() > 0:
            torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        cuda_states = state["torch_cuda_all"]
        if isinstance(cuda_states, (list, tuple)):
            cuda_states_bt = [_to_byte_tensor(s) for s in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states_bt)  # type: ignore[arg-type]
        else:
            # Older formats may store a single state; broadcast to all devices
            bt = _to_byte_tensor(cuda_states)
            if bt.numel() > 0:
                torch.cuda.set_rng_state_all([bt for _ in range(torch.cuda.device_count())])  # type: ignore[list-item]


