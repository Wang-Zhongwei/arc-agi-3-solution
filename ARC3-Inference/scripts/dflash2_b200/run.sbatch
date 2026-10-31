#!/bin/bash
#SBATCH --job-name=dflash2_k7_b200
#SBATCH --account=arc-agi-3
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=B200:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=700G
#SBATCH --time=02:30:00
#SBATCH --output=/shared/workspace/arc-agi-3/dflash2_b200/out/%j.txt

# DFlash2 k=7 (prefix_match_unit 16) on Lord's e975732 rc3 runtime, next to k=3 and MTP-3, one B200 each:
#   A = dflash2-k7 v3, B = dflash2-k3 v2, C = porting-advaith v6 (Lord NVFP4 MTP-3)
# Each arm runs its Kaggle setup script (setup_<arm>.py, the rendered setup minus the RTX GPU-name check) with the
# KV pool its Kaggle run had, then bench.py replays recorded GRPO episodes (porting-advaith v6 harness) at 14
# concurrent streams with the harness's sampling. Results: out/<jobid>/<arm>/{vllm-openai-server.log,bench/summary.json}
#
#   sbatch run.sbatch                      # open partition
#   ARMS="A" sbatch run.sbatch             # one arm (then --gpus-per-node=B200:1)
#   sbatch --reservation=maint_arc run.sbatch

D=/shared/workspace/arc-agi-3/dflash2_b200
W=/shared/workspace/arc-agi-3
export ARMS=${ARMS:-A B C}
export BENCH_ARGS=${BENCH_ARGS:-"--workload $D/run/workload.jsonl --concurrency 14 --window 900 --gen-scale 1.0 --warmup-max 1500"}
export OUT=$D/out/$SLURM_JOB_ID
mkdir -p $OUT
echo "job $SLURM_JOB_ID on $(hostname), GPUs $CUDA_VISIBLE_DEVICES, arms $ARMS"

srun --container-mounts $W:$W,$D/input:/kaggle/input --container-image $W/containers/pytorch_26.09-py3.sqsh \
  bash -c '
    set -u
    python3 --version; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    df -BG --output=target,avail /tmp | tail -1
    # unpacked runtimes (~20 GB each) on node-local /tmp when it has room, else on the job folder
    if [ "$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)" -gt 120 ]; then RT=/tmp/dflash2_b200_$SLURM_JOB_ID; else RT=$OUT/rt; fi
    IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    i=0
    for arm in $ARMS; do
      mkdir -p $OUT/$arm
      CUDA_VISIBLE_DEVICES=${GPUS[$i]} TAAF_IMAGE_RUNTIME_ROOT=$RT/$arm/vllm-image-runtime TAAF_KAGGLE_BUNDLE_DIR='$D'/bundle \
        setsid python3 '$D'/run/run_arm.py $arm $OUT/$arm $BENCH_ARGS > $OUT/$arm/driver.log 2>&1 &
      echo "arm $arm on GPU ${GPUS[$i]}, runtime $RT/$arm"
      i=$((i + 1))
    done
    wait
    rm -rf $RT
    for arm in $ARMS; do echo "== $arm"; cat $OUT/$arm/bench/summary.json 2>/dev/null || tail -5 $OUT/$arm/driver.log; done
  '
