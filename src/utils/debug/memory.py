import torch


class MemoryTracker:
    def __init__(self):
        self.memory = 0

    def track_change(self, s):
        gbs = get_gpu_memory_gb()
        change = gbs - self.memory
        change_s = f"+{change:0.1f}" if change >= 0 else f"{change:0.1f}"
        print(f"{change_s} {gbs:0.1f} GB {s}")
        self.memory = gbs


tracker = MemoryTracker()


def print_gpu_memory_gb(s=""):
    tracker.track_change(s)
    # gbs = get_gpu_memory_gb()
    # print(f"{gbs:0.1f} GB {s}")


def print_max_gpu_memory_gb(s=""):
    gbs = get_max_gpu_memory_gb()
    print(f"{gbs:0.1f} GB (max) {s}")


def print_tensor_size(t, s=""):
    gbs = get_tensor_size_gb(t)
    print(f"{gbs:0.1f} GB {s}")


def print_tensor_list_size(ts, s=""):
    gbs = sum([get_tensor_size_gb(t) for t in ts])
    print(f"{gbs:0.1f} GB {s}")


def get_tensor_size_gb(t):
    bs = t.element_size() * t.nelement()
    return bs / 1024 ** 3


def get_gpu_memory_gb():
    return torch.cuda.memory_allocated() / 1024 ** 3


def get_max_gpu_memory_gb():
    return torch.cuda.max_memory_allocated() / 1024 ** 3
