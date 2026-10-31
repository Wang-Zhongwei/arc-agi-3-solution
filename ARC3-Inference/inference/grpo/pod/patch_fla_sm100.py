"""fla 0.5.2 with Triton 3.4 on sm100 (B200): the GDN backward's pipelined configs (num_stages 2-4) fail to compile;
give the backward num_stages 1 on Blackwell, as fla already does for the forward (a301c3a, train_pod.sh). The new
condition only changes Blackwell, so the patch is safe on any GPU and can run at image build time. Idempotent.

    <trainer venv>/bin/python inference/grpo/pod/patch_fla_sm100.py
"""

import importlib.util
import pathlib

OLD = "for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [1])"
NEW = "for num_stages in ([2, 3, 4] if check_shared_mem('ampere') and not IS_NVIDIA_BLACKWELL else [1])"

# find the file without importing fla (its import probes the GPU)
path = pathlib.Path(importlib.util.find_spec("fla").submodule_search_locations[0]) / "ops/common/chunk_delta_h.py"
text = path.read_text()
if text.count(NEW) == 1:
    print("fla sm100 patch: already applied")
elif text.count(OLD) == 1:
    path.write_text(text.replace(OLD, NEW))
    print("fla sm100 patch: applied")
else:
    raise SystemExit(f"fla sm100 patch: not found in {path} (fla changed?)")
