import os
import time

import torch

CHECKPOINT_DIR = "checkpoints"


def save_checkpoint(net, prefix, name=None):
    """
    Save model checkpoint to disk
    """
    if name:
        checkpoint_filename = f"{prefix}-{name}-{int(time.time())}.ckpt"
    else:
        checkpoint_filename = f"{prefix}-{int(time.time())}.ckpt"

    print(f"\nSaving checkpoint as {checkpoint_filename}\n")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
    torch.save(net.state_dict(), checkpoint_path)
    return checkpoint_path
