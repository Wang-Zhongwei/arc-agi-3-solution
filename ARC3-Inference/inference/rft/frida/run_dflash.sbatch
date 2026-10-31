#!/bin/bash
#SBATCH --job-name=arc3_dflash_train
#SBATCH --account=arc-agi-3
#SBATCH --partition=frida
#SBATCH --time=1-00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=B200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=/shared/workspace/arc-agi-3/dflash_finetune/runs/%x-%j.txt

# DFlash2 drafter fine-tuning on Frida (inference/rft/dflash_train.py) in the GRPO container, one B200, k = 7 draft
# tokens: the frozen Intel W4A16 target runs forward only, as for run_mtp.sbatch, and incoai's DFlash2 drafter trains.
# Data: mtp_data.py's output, the same samples and game split as the MTP head's.
#   sbatch run_dflash.sbatch --eval_only=1        # measure incoai's drafter as loaded (calibration)
#   sbatch run_dflash.sbatch --test=1             # also run tests/test_dflash_train.py on the GPU first (flex)
#   sbatch run_dflash.sbatch                      # train (resumes <version> when it has latest/)
#   sbatch run_dflash.sbatch --version=<v> --lr=3e-5 --train_samples=600 --extra_args="--accumulate 2"
# No reservation: any free B200 (sbatch --reservation=maint_arc ... to use the team's reserved nodes).
# Results: dflash_finetune/runs/<version>/: train_log.jsonl (eval, sample and step records), latest/ (drafter +
# optimizer), best/ (the drafter with the best eval acceptance length, step 0 included), draft/ (config.json and
# model.safetensors of the best drafter, a vLLM DFlash2 draft directory ready to upload).
# Settings, as --key=value (or the environment variable KEY): version (date-time), data, eval_only (0), test (0),
# lr (3e-5), train_samples (1400), eval_samples (48), eval_every (100), blocks (2048), fp8_kv (both), max_hours (22),
# wandb_project (arc3-dflash), wandb_entity (jonathan_wang1999-epsilla), extra_args (more dflash_train.py flags).

for argument in "$@"; do
  if [[ $argument == --*=* ]]; then
    key=${argument%%=*}
    key=${key#--}
    declare "${key^^}=${argument#*=}"
  fi
done

WORK_DIR=/shared/workspace/arc-agi-3
BASE=$WORK_DIR/dflash_finetune
REPO_DIR=${REPO_DIR:-$BASE/repo}                 # the code (branch dflash2-drafter)
MODEL_DIR=$WORK_DIR/models/Qwen3.8-Flash-Next-W4A16-AutoRound
DRAFT_DIR=${DRAFT_DIR:-$WORK_DIR/dflash2_b200/models/dflash2}  # incoai/Qwen3.8-Flash-Next-DFlash2 a22a1744
DATA=${DATA:-$WORK_DIR/mtp_finetune/data/mtp_v1}
CONTAINER=$WORK_DIR/containers/pytorch_26.09-py3.sqsh
VERSION=${VERSION:-$(date "+%Y%m%d-%H%M")}
EVAL_ONLY=${EVAL_ONLY:-0}
TEST=${TEST:-0}
LR=${LR:-3e-5}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-1400}  # the MTP head's ~50M trained tokens, 70 per game
EVAL_SAMPLES=${EVAL_SAMPLES:-48}
EVAL_EVERY=${EVAL_EVERY:-100}
BLOCKS=${BLOCKS:-2048}
FP8_KV=${FP8_KV:-both}
MAX_HOURS=${MAX_HOURS:-22}
EXTRA_ARGS=${EXTRA_ARGS:-}
OUTPUT_DIR=$BASE/runs/$VERSION

[ -s "$DATA/train.jsonl" ] && [ -s "$DATA/eval.jsonl" ] && [ -d "$DATA/images" ] || { echo "$DATA is not an mtp_data output"; exit 1; }
[ -s "$DRAFT_DIR/model.safetensors" ] && [ -s "$DRAFT_DIR/config.json" ] || { echo "$DRAFT_DIR lacks the DFlash2 checkpoint"; exit 1; }
mkdir -p $OUTPUT_DIR
RESUME=""
[ -f $OUTPUT_DIR/latest/state.json ] && RESUME="--resume"
FLAGS="--draft-tokens 7 --blocks $BLOCKS --fp8-kv $FP8_KV --lr $LR --train-samples $TRAIN_SAMPLES"
FLAGS="$FLAGS --eval-samples $EVAL_SAMPLES --eval-every $EVAL_EVERY --max-hours $MAX_HOURS"
[ "$EVAL_ONLY" = 1 ] && FLAGS="$FLAGS --eval-only"
# wandb: WANDB_API_KEY from the team's .env (offline into <out>/wandb without it: wandb sync <out>/wandb/offline-run-*)
WANDB_KEY=$(grep -s "^WANDB_API_KEY=" $WORK_DIR/.env | head -1 | cut -d= -f2-)
WANDB_PROJECT=${WANDB_PROJECT:-arc3-dflash}
WANDB_ENTITY=${WANDB_ENTITY:-jonathan_wang1999-epsilla}
FLAGS="$FLAGS --wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY --wandb-run-name dflash-$VERSION"
(cd $REPO_DIR && git rev-parse HEAD 2>/dev/null || cat $REPO_DIR/COMMIT 2>/dev/null) > $OUTPUT_DIR/commit_${SLURM_JOB_ID:-local}.txt
echo "version $VERSION, data $DATA ($(wc -l < $DATA/train.jsonl) train / $(wc -l < $DATA/eval.jsonl) eval samples), flags '$FLAGS $RESUME $EXTRA_ARGS', started $(date)"

SCRIPT=$OUTPUT_DIR/run_${SLURM_JOB_ID:-local}.sh
cat > $SCRIPT <<EOF
#!/bin/bash
export PYTHONUNBUFFERED=TRUE TOKENIZERS_PARALLELISM=false
[ -n "\$WANDB_API_KEY" ] || echo "no WANDB_API_KEY: wandb offline into /output/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/repo/ARC3-Inference:/repo/tufa-arc-agi-framework/src
cd /output
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$TEST" = 1 ]; then
  (cd /repo/ARC3-Inference && python3 -m pytest -q -rs -p no:cacheprovider tests/test_dflash_train.py) > /output/tests.log 2>&1 \
    || { echo "tests failed: /output/tests.log"; tail -30 /output/tests.log; exit 1; }
  tail -3 /output/tests.log
fi
python3 -m inference.rft.dflash_train --model /model --draft /draft --data /data --out /output $FLAGS $RESUME $EXTRA_ARGS \
  > /output/trainer.log 2>&1
EOF
chmod a+x $SCRIPT

# the key reaches the container through the environment only, never a file in the run folder
[ -n "$WANDB_KEY" ] && export WANDB_API_KEY="$WANDB_KEY"
unset WANDB_KEY
srun \
  --container-mounts $REPO_DIR:/repo,$MODEL_DIR:/model,$DRAFT_DIR:/draft,$DATA:/data,$OUTPUT_DIR:/output \
  --container-image $CONTAINER \
  /output/run_${SLURM_JOB_ID:-local}.sh

status=$?
chmod -R g+rwX $OUTPUT_DIR 2>/dev/null
echo "finished $(date), exit $status"
exit $status
