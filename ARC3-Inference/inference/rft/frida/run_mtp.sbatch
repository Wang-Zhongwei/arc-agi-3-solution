#!/bin/bash
#SBATCH --job-name=arc3_mtp_train
#SBATCH --account=arc-agi-3
#SBATCH --partition=frida
#SBATCH --time=1-00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=B200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=/shared/workspace/arc-agi-3/mtp_finetune/runs/%x-%j.txt

# MTP draft-head fine-tuning on Frida (inference/rft/mtp_train.py) in the GRPO container, one B200: the frozen Intel
# W4A16 target runs forward only (68.5 GiB, PLE table memory-mapped from the checkpoint into the page cache, hence
# --mem), the head trains. Data: mtp_data.py's output (train.jsonl, eval.jsonl, images/).
#   sbatch run_mtp.sbatch --eval_only=1                       # measure lordhansolo's head as loaded (calibration)
#   sbatch run_mtp.sbatch --eval_only=1 --positions=mrope     # the same with the target's M-RoPE positions
#   sbatch run_mtp.sbatch                                     # train (resumes <version> when it has latest/)
#   sbatch run_mtp.sbatch --version=<v> --lr=3e-5 --train_samples=600 --extra_args="--accumulate 2"
# No reservation: any free B200 (sbatch --reservation=maint_arc ... to use the team's reserved nodes).
# Results: mtp_finetune/runs/<version>/: train_log.jsonl (eval, sample and step records), latest/ (head + optimizer),
# best/ (the head with the best eval acceptance length, step 0 included),
# draft/ (a vLLM draft directory of the best head: lordhansolo's files with the fine-tuned mtp.* tensors in
# mtp-finetuned-bf16.safetensors; the other files are hard links to lord_mtp/, ready to upload).
# Settings, as --key=value (or the environment variable KEY): version (date-time), data, eval_only (0), positions
# (index), lr (3e-5), train_samples (1400), eval_samples (48), eval_every (100), steps (3), max_hours (22),
# wandb_project (arc3-mtp), wandb_entity (jonathan_wang1999-epsilla), extra_args (more mtp_train.py flags, quoted).

for argument in "$@"; do
  if [[ $argument == --*=* ]]; then
    key=${argument%%=*}
    key=${key#--}
    declare "${key^^}=${argument#*=}"
  fi
done

WORK_DIR=/shared/workspace/arc-agi-3
BASE=$WORK_DIR/mtp_finetune
REPO_DIR=${REPO_DIR:-$BASE/repo}                 # the code (branch mtp-data)
MODEL_DIR=$WORK_DIR/models/Qwen3.8-Flash-Next-W4A16-AutoRound
LORD_DIR=$BASE/lord_mtp                          # primitive-ai mtp_nvfp4 files = lordhansolo's Kaggle draft
DATA=${DATA:-$BASE/data/mtp_v1}
CONTAINER=$WORK_DIR/containers/pytorch_26.09-py3.sqsh
VERSION=${VERSION:-$(date "+%Y%m%d-%H%M")}
EVAL_ONLY=${EVAL_ONLY:-0}
POSITIONS=${POSITIONS:-index}
LR=${LR:-3e-5}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-1400}  # ~50M trained tokens, 70 per game
EVAL_SAMPLES=${EVAL_SAMPLES:-48}
EVAL_EVERY=${EVAL_EVERY:-100}
STEPS=${STEPS:-3}
MAX_HOURS=${MAX_HOURS:-22}
EXTRA_ARGS=${EXTRA_ARGS:-}
OUTPUT_DIR=$BASE/runs/$VERSION

[ -s "$DATA/train.jsonl" ] && [ -s "$DATA/eval.jsonl" ] && [ -d "$DATA/images" ] || { echo "$DATA is not an mtp_data output"; exit 1; }
[ -s "$LORD_DIR/mtp-experts-nvfp4.safetensors" ] || { echo "$LORD_DIR lacks the draft checkpoint"; exit 1; }
mkdir -p $OUTPUT_DIR
RESUME=""
[ -f $OUTPUT_DIR/latest/state.json ] && RESUME="--resume"
FLAGS="--steps $STEPS --positions $POSITIONS --lr $LR --train-samples $TRAIN_SAMPLES --eval-samples $EVAL_SAMPLES"
FLAGS="$FLAGS --eval-every $EVAL_EVERY --max-hours $MAX_HOURS --draft-vocab /repo/ARC3-Inference/configs/draft_vocab_32k.json"
[ "$EVAL_ONLY" = 1 ] && FLAGS="$FLAGS --eval-only"
# wandb: WANDB_API_KEY from the team's .env (offline into <out>/wandb without it: wandb sync <out>/wandb/offline-run-*)
WANDB_KEY=$(grep -s "^WANDB_API_KEY=" $WORK_DIR/.env | head -1 | cut -d= -f2-)
WANDB_PROJECT=${WANDB_PROJECT:-arc3-mtp}
WANDB_ENTITY=${WANDB_ENTITY:-jonathan_wang1999-epsilla}
FLAGS="$FLAGS --wandb-project $WANDB_PROJECT --wandb-entity $WANDB_ENTITY --wandb-run-name mtp-$VERSION"
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
python3 -m inference.rft.mtp_train --model /model --mtp /lord --data /data --out /output $FLAGS $RESUME $EXTRA_ARGS \
  > /output/trainer.log 2>&1
EOF
chmod a+x $SCRIPT

# the key reaches the container through the environment only, never a file in the run folder
[ -n "$WANDB_KEY" ] && export WANDB_API_KEY="$WANDB_KEY"
unset WANDB_KEY
srun \
  --container-mounts $REPO_DIR:/repo,$MODEL_DIR:/model,$LORD_DIR:/lord,$DATA:/data,$OUTPUT_DIR:/output \
  --container-image $CONTAINER \
  /output/run_${SLURM_JOB_ID:-local}.sh

status=$?
chmod -R g+rwX $OUTPUT_DIR 2>/dev/null
echo "finished $(date), exit $status"
exit $status
