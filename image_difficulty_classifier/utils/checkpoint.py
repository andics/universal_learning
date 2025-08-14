import glob
import os
from typing import Any, Dict, Optional

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_step(filename: str) -> int:
    # filename like checkpoint_epoch3_step123.pt or checkpoint_step123.pt
    base = os.path.basename(filename)
    try:
        if "checkpoint_epoch" in base:
            # New format: extract step number from epoch_step format
            parts = base.split("_step")
            if len(parts) >= 2:
                num = parts[-1].split(".pt")[0]
                return int(num)
        # Old format: checkpoint_step123.pt
        num = base.split("checkpoint_step")[-1].split(".pt")[0]
        return int(num)
    except Exception:
        return -1


def get_latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    # Match both old and new formats
    patterns = [
        os.path.join(checkpoints_dir, "checkpoint_epoch*_step*.pt"),
        os.path.join(checkpoints_dir, "checkpoint_step*.pt")
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=_extract_step)
    return candidates[-1]


def save_checkpoint(
    checkpoints_dir: str,
    step: int,
    state: Dict[str, Any],
    epoch: Optional[int] = None,
) -> str:
    ensure_dir(checkpoints_dir)
    if epoch is not None:
        ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
    else:
        ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_step{step}.pt")
    torch.save(state, ckpt_path)
    latest_symlink = os.path.join(checkpoints_dir, "latest.pt")
    try:
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(os.path.basename(ckpt_path), latest_symlink)
    except OSError:
        # Symlinks may not work on all platforms; ignore.
        pass
    return ckpt_path


def load_checkpoint(ckpt_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(ckpt_path, map_location=map_location or "cpu")


