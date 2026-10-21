#!/bin/bash
# Check a merged model as published: on the training pod (venv, data/, train/ and results/train_flags.txt on its
# volume), download the merged non-PLE files from Hugging Face over the base checkpoint there (HF layout: its own
# PLE shard and index stay, so no PLE download), run train_rft's validation pass on it (a fresh, zero LoRA on top;
# training's flags and val samples) and compare with the adapter's last val loss. The base folder ends up merged;
# Hugging Face has the base again if it is needed. Writes PASS or FAIL, then FULL_DONE, and stops the pod. Nothing waits for a person.
#   HF token in /workspace/.hf_token (mode 600), then
#   ssh ... 'cd /workspace/rft; nohup bash inference/rft/pod/verify_merged_pod.sh > /dev/null 2>&1 < /dev/null &' < /dev/null
#   WORKDIR=/workspace/rft RUNLOG=verify_merged/verify.log PULL=verify_merged bash inference/rft/pod/watch_pod.sh POD HOST PORT DEST
set -uo pipefail

WORK=${WORK:-/workspace/rft}
VENV=${VENV:-/workspace/venv}
HF_REPO=${HF_REPO:-helldriver007/qwen3.8-flash-next-w4a16-rft-v1}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}  # the base in HF layout (model-00016-of-00017.safetensors = PLE)
EXPECT_VAL=${EXPECT_VAL:-}  # default: the adapter's last val loss in train/train_log.jsonl
VAL_TOL=${VAL_TOL:-0.003}
MAX_MINUTES=${MAX_MINUTES:-90}
POD_END=${POD_END:-stop}
OUT=$WORK/verify_merged
cd "$WORK"
mkdir -p "$OUT"
rm -f "$OUT/FULL_DONE" "$OUT/ABORTED" "$OUT/PASS" "$OUT/FAIL"
exec >> "$OUT/verify.log" 2>&1
for var in RUNPOD_POD_ID RUNPOD_API_KEY; do
    [ -n "${!var:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$var=")" 2>/dev/null || true
done
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
self_stop() {
    log "ending this pod ($1): $POD_END"
    if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl config --apiKey "${RUNPOD_API_KEY:-}" > /dev/null 2>&1
        if [ "$POD_END" = delete ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi \
            || log "runpodctl failed"
    fi
}
abort() { log "ABORTED: $*"; echo "$*" > "$OUT/ABORTED"; self_stop abort; exit 1; }
( sleep $((MAX_MINUTES * 60)); echo "time limit" > "$OUT/ABORTED"; self_stop "time limit ${MAX_MINUTES} min" ) &
export PATH="$VENV/bin:$PATH" PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH

[ -s /workspace/.hf_token ] || abort "no /workspace/.hf_token"
export HF_TOKEN="$(cat /workspace/.hf_token)"
[ -s data/val.jsonl ] || abort "no data/val.jsonl"
if [ -z "$EXPECT_VAL" ]; then
    EXPECT_VAL=$(python -c "import json; print([r for r in map(json.loads, open('train/train_log.jsonl')) if r['event'] == 'val'][-1]['loss'])") \
        || abort "no val loss in train/train_log.jsonl"
fi
log "verify hf.co/$HF_REPO: expect val $EXPECT_VAL +- $VAL_TOL; $(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [ ! -e "$OUT/download.done" ]; then
    [ -s "$MODEL_DIR/model-00016-of-00017.safetensors" ] || abort "$MODEL_DIR has no HF PLE shard (the check keeps the base's PLE)"
    start=$(date +%s)
    # the merged files only: not the Kaggle index/manifest (they name model-ple-*), not PLE files (an older full upload)
    hf download "$HF_REPO" --local-dir "$MODEL_DIR" --exclude "rft/*" --exclude "model-ple-*" \
        --exclude "model.safetensors.index.json" --exclude "ple_reshard_manifest.json" > "$OUT/download.log" 2>&1 \
        || abort "download: $(tail -1 "$OUT/download.log")"
    [ -s "$MODEL_DIR/MERGED_LORA.json" ] || abort "no MERGED_LORA.json from hf.co/$HF_REPO: not a merged checkpoint"
    log "merged files downloaded over the base in $(( $(date +%s) - start )) s"
    date -u > "$OUT/download.done"
fi
rm -rf "$OUT/run"
python -m inference.rft.train_rft --model "$MODEL_DIR" --data data --out "$OUT/run" --epochs 0 \
    $(cat results/train_flags.txt 2>/dev/null) > "$OUT/train_rft.log" 2>&1 || abort "train_rft: $(tail -1 "$OUT/train_rft.log")"
got=$(python -c "import json; print([r for r in map(json.loads, open('$OUT/run/train_log.jsonl')) if r['event'] == 'val'][-1]['loss'])") \
    || abort "no val row"
if python -c "import sys; sys.exit(abs($got - $EXPECT_VAL) > $VAL_TOL)"; then
    echo "$got" > "$OUT/PASS"; log "PASS: merged val $got, adapter $EXPECT_VAL (tolerance $VAL_TOL)"
else
    echo "$got" > "$OUT/FAIL"; log "FAIL: merged val $got, adapter $EXPECT_VAL (tolerance $VAL_TOL)"
fi
date -u > "$OUT/FULL_DONE"
log "FULL_DONE"
( sleep 600; self_stop "10 min after FULL_DONE" ) &
