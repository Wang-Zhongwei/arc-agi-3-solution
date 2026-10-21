#!/bin/bash
# One-shot publish of a trained adapter on the training pod (its volume already holds the checkpoint, venv and
# data): merge the adapter into the checkpoint, swap HF's PLE shard for the base Kaggle model's PLE files (the
# layout serving loads), check the merged model's val loss against training's, upload it as a private Kaggle
# Model, then FULL_DONE and stop the pod. A failed check aborts before the upload. Each step leaves a .done
# file, so a rerun continues where the last one stopped. Nothing waits for a person.
#
# After a pod restart /root is empty: copy ~/.kaggle/{access_token,kaggle.json} to /root/.kaggle (mode 600), then
#   ssh ... 'cd /workspace/rft; nohup bash inference/rl/pod/merge_upload_pod.sh > /dev/null 2>&1 < /dev/null &' < /dev/null
#   WORKDIR=/workspace/rft RUNLOG=publish/publish.log PULL=publish bash inference/rl/pod/watch_pod.sh POD HOST PORT DEST
set -uo pipefail

WORK=${WORK:-/workspace/rft}
VENV=${VENV:-/workspace/venv}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
ADAPTER=${ADAPTER:-train/adapter_final}
EXPECT_VAL=${EXPECT_VAL:-}  # default: the adapter's last val loss in train/train_log.jsonl
VAL_TOL=${VAL_TOL:-0.003}
OWNER=${OWNER:-jonathanwang2022}
SLUG=${SLUG:-qwen3.8-flash-next-w4a16-rft-v1}
TITLE=${TITLE:-Qwen3.8 Flash Next W4A16 RFT v1}
DESCRIPTION=${DESCRIPTION:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound with the RFT v1 LoRA (r16, attention + GDN projections, 2 epochs) merged into its BF16 weights. Same files and layout as dfranzen/intel-qwen3.8-flash-next-w4a16-autoround; MERGED_LORA.json lists what changed.}
MAX_MINUTES=${MAX_MINUTES:-240}
POD_END=${POD_END:-stop}
OUT=$WORK/publish
cd "$WORK"
mkdir -p "$OUT"
rm -f "$OUT/FULL_DONE" "$OUT/ABORTED"
exec >> "$OUT/publish.log" 2>&1
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
export PATH="$VENV/bin:$PATH" PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KAGGLEHUB_CACHE=/workspace/kcache  # on the volume: a fetched file moves into the checkpoint by rename
step() { [ -e "$OUT/$1.done" ]; }
done_() { date -u > "$OUT/$1.done"; log "$1 done"; }

log "publish $ADAPTER -> $OWNER/$SLUG (pod ${RUNPOD_POD_ID:-?}); disk: $(du -sh "$MODEL_DIR" | cut -f1) in $MODEL_DIR"
[ -s "$ADAPTER/adapter_model.safetensors" ] || abort "no $ADAPTER"
[ -s /root/.kaggle/access_token ] || abort "no /root/.kaggle/access_token (copy it after a restart)"
python -c "import kagglehub, kaggle" 2>/dev/null || pip install -q kagglehub kaggle || abort "pip install"
if [ -z "$EXPECT_VAL" ]; then
    EXPECT_VAL=$(python -c "import json; print([r for r in map(json.loads, open('train/train_log.jsonl')) if r['event'] == 'val'][-1]['loss'])") \
        || abort "no val loss in train/train_log.jsonl"
fi
log "expect val $EXPECT_VAL +- $VAL_TOL"

if ! step merge; then
    python -m inference.rl.merge_lora --checkpoint "$MODEL_DIR" --adapter "$ADAPTER" > "$OUT/merge.log" 2>&1 \
        || abort "merge: $(tail -1 "$OUT/merge.log")"
    grep -A12 '^{' "$OUT/merge.log" | grep -E "tensors|kept|cosine"
    done_ merge
fi
if ! step layout; then
    python -m inference.rl.kaggle_layout --checkpoint "$MODEL_DIR" > "$OUT/layout.log" 2>&1 \
        || abort "layout: $(tail -1 "$OUT/layout.log")"
    rm -rf /workspace/kcache
    tail -1 "$OUT/layout.log"
    done_ layout
fi
if ! step verify; then
    # val loss of the merged weights (with a fresh, zero LoRA on top), with training's flags and val samples
    python -m inference.rl.train_rft --model "$MODEL_DIR" --data data --out "$OUT/verify" --epochs 0 \
        $(cat results/train_flags.txt 2>/dev/null) > "$OUT/verify.log" 2>&1 || abort "verify: $(tail -1 "$OUT/verify.log")"
    got=$(python -c "import json; print([r for r in map(json.loads, open('$OUT/verify/train_log.jsonl')) if r['event'] == 'val'][-1]['loss'])") \
        || abort "verify: no val row"
    log "merged val $got (adapter $EXPECT_VAL)"
    python -c "import sys; sys.exit(abs($got - $EXPECT_VAL) > $VAL_TOL)" || abort "merged val $got vs $EXPECT_VAL: off by more than $VAL_TOL"
    done_ verify
fi
if ! step upload; then
    mv "$MODEL_DIR/.cache" "/workspace/hf_cache_$(date +%s)" 2>/dev/null || true
    log "uploading $(du -sh "$MODEL_DIR" | cut -f1), $(ls "$MODEL_DIR" | wc -l) files"
    for attempt in 1 2 3; do
        VERSION_NOTES="RFT v1 adapter merged (attempt $attempt)" \
            bash inference/rl/pod/kaggle_model_upload.sh "$MODEL_DIR" "$OWNER" "$SLUG" "$TITLE" "$DESCRIPTION" \
            > "$OUT/upload_$attempt.log" 2>&1 && break
        log "upload attempt $attempt failed: $(tail -1 "$OUT/upload_$attempt.log")"
        [ $attempt = 3 ] && abort "upload failed 3 times"
        sleep 60
    done
    tail -2 "$OUT/upload_$attempt.log"
    done_ upload
fi
date -u > "$OUT/FULL_DONE"
log "FULL_DONE: $OWNER/$SLUG/Transformers/default"
( sleep 600; self_stop "10 min after FULL_DONE" ) &
