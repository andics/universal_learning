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
    def _to_byte_tensor_cpu(x) -> torch.ByteTensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(device="cpu", dtype=torch.uint8)  # type: ignore[return-value]
        try:
            return torch.tensor(x, device="cpu", dtype=torch.uint8)  # type: ignore[return-value]
        except Exception:
            return torch.empty(0, device="cpu", dtype=torch.uint8)  # type: ignore[return-value]

    def _to_byte_tensor_cuda(x, device_index: int) -> torch.ByteTensor:
        device = f"cuda:{device_index}"
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=device, dtype=torch.uint8)  # type: ignore[return-value]
        try:
            return torch.tensor(x, device=device, dtype=torch.uint8)  # type: ignore[return-value]
        except Exception:
            return torch.empty(0, device=device, dtype=torch.uint8)  # type: ignore[return-value]

    if "torch_cpu" in state:
        cpu_state = _to_byte_tensor_cpu(state["torch_cpu"])  # type: ignore[arg-type]
        if cpu_state.numel() > 0:
            torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        cuda_states = state["torch_cuda_all"]
        if isinstance(cuda_states, (list, tuple)):
            device_count = torch.cuda.device_count()
            n = min(len(cuda_states), device_count)
            cuda_states_bt = [_to_byte_tensor_cuda(cuda_states[i], i) for i in range(n)]
            # If fewer states than devices, repeat the last state
            if n < device_count and n > 0:
                last = cuda_states_bt[-1]
                cuda_states_bt.extend([last] * (device_count - n))
            torch.cuda.set_rng_state_all(cuda_states_bt)  # type: ignore[arg-type]
        else:
            # Older formats may store a single state; broadcast to all devices
            device_count = torch.cuda.device_count()
            if device_count > 0:
                bt_list = [_to_byte_tensor_cuda(cuda_states, i) for i in range(device_count)]
                torch.cuda.set_rng_state_all(bt_list)  # type: ignore[arg-type]


