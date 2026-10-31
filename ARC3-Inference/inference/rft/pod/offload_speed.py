"""Host <-> GPU copy speeds for checkpoint offload, done the way host_offload.py does them.

Fresh pageable memory per copy (what save_on_cpu does), a reused pageable arena, and a pinned
arena; then whether a pinned copy on a side stream overlaps with matmuls on the main stream.

    PYTHONPATH=. python inference/rft/pod/offload_speed.py   # SIZE_GB=1 REPS=4
"""

import os
import time

import torch

from inference.rft.host_offload import HostArena

size_gb = float(os.environ.get("SIZE_GB", "1"))  # one layer's checkpoint at 64K tokens is 1.24 GiB
reps = int(os.environ.get("REPS", "4"))
n = int(size_gb * 2**30) // 2
x = torch.randn(n, device="cuda", dtype=torch.bfloat16)


def timed(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def best(fn) -> float:
    return min(timed(fn) for _ in range(reps))


print(torch.cuda.get_device_name(0), "torch", torch.__version__, f"{size_gb:g} GiB per copy")
speeds = {}
speeds["D2H into fresh pageable memory"] = best(lambda: torch.empty(n, dtype=torch.bfloat16).copy_(x))
pageable = HostArena(segment_gb=size_gb)
view = pageable.take(n * 2).view(torch.bfloat16)
speeds["D2H into a reused pageable arena"] = best(lambda: view.copy_(x))
speeds["H2D from a reused pageable arena"] = best(lambda: x.copy_(view))
start = time.perf_counter()
pinned = HostArena(segment_gb=size_gb, pin=True)
pinned_view = pinned.take(n * 2).view(torch.bfloat16)
allocate = time.perf_counter() - start
print(f"pinned segment of {pinned.capacity_gb:g} GiB allocated in {allocate:.2f} s, pinned: {pinned_view.is_pinned()}")
speeds["D2H into a pinned arena"] = best(lambda: pinned_view.copy_(x, non_blocking=True))
speeds["H2D from a pinned arena"] = best(lambda: x.copy_(pinned_view, non_blocking=True))
for name, seconds in speeds.items():
    print(f"{name:34s} {size_gb / seconds:6.1f} GiB/s")

a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
stream = torch.cuda.Stream()


def compute():
    for _ in range(10):
        a @ a


def compute_and_copy():
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pinned_view.copy_(x, non_blocking=True)
    compute()


matmuls, copy, both = best(compute), speeds["D2H into a pinned arena"], best(compute_and_copy)
print(f"matmuls {matmuls:.3f} s, pinned copy {copy:.3f} s, both at once {both:.3f} s "
      f"({'overlapped' if both < 0.9 * (matmuls + copy) else 'serialized'})")
