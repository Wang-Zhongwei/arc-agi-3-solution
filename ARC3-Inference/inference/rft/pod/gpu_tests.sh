#!/bin/bash
# GPU unit tests on a light pod (no checkpoint): the RL tests, including the CUDA-only async
# offload test, then host <-> GPU copy speeds. Upload inference/ and tests/ to $WORK, then:
#   ssh ... 'cd /workspace/rl; nohup bash inference/rft/pod/gpu_tests.sh > gpu_tests.log 2>&1 < /dev/null &' < /dev/null
# $WORK/DONE or $WORK/FAILED marks the end; results land in $WORK/results/.
set -uo pipefail

WORK=${WORK:-/workspace/rl}
VENV=${VENV:-/workspace/venv}
cd "$WORK"
mkdir -p results
rm -f DONE FAILED
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; echo "$*" > FAILED; exit 1; }

log "phase 1: python packages"
# the image's system Python is externally managed (PEP 668): a venv that still sees its torch
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || fail "venv"
export PATH="$VENV/bin:$PATH"
if ! python -c "import transformers, peft, pytest; assert transformers.__version__ == '5.20.1'" 2>/dev/null; then
    python -m pip install -q -U pip
    python -m pip install -q "transformers==5.20.1" peft pytest pillow numpy || fail "pip install"
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" \
    | tee results/env.txt || fail "no GPU"

log "phase 2: unit tests"
PYTHONPATH=. python -m pytest -q -rs tests > results/pytest.log 2>&1
status=$?
log "pytest exit $status: $(tail -1 results/pytest.log)"

log "phase 3: copy speeds"
PYTHONPATH=. python inference/rft/pod/offload_speed.py > results/offload_speed.txt 2>&1 || log "copy speed script failed"
cat results/offload_speed.txt

if [ $status -eq 0 ]; then date -u > DONE; else echo "pytest exit $status" > FAILED; fi
log "all done"
