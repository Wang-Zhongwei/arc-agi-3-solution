#!/bin/bash
# Stage the Kaggle inputs of the dflash2-k7 v3 / dflash2-k3 v2 / porting-advaith v6 runs on Frida (login node).
#   D/runtime  = lordhansolo/vllm-main-e975732-arc3: manifest, overlay e3a6fe0d, applier (copied in by scp) plus
#                the 6 selected image layers, fetched from Docker Hub by digest and checked against the manifest
#   D/models/intel-w4a16 = the Frida W4A16 AutoRound files with the ORIGINAL config.json (Frida's config.json was
#                edited for upstream vLLM: QSA layers relabelled full_attention)
#   D/models/dflash2 = incoai/Qwen3.8-Flash-Next-DFlash2 @ a22a1744 (= Kaggle jonathanwang2022/qwen3-8-flash-next-dflash2)
#   D/input    = the /kaggle/input tree the setup scripts resolve
set -euo pipefail
D=/shared/workspace/arc-agi-3/dflash2_b200
W=/shared/workspace/arc-agi-3
mkdir -p $D/runtime $D/models/intel-w4a16 $D/models/dflash2 $D/out

# image layers
cd $D/runtime
TOK=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:vllm/vllm-openai:pull" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['token'])")
python3 -c "
import json
for l in json.load(open('runtime-manifest.json'))['selected_layers']:
    print(l['digest'], l['file'])" | while read digest file; do
  if [ -f "$file" ] && [ "$(sha256sum "$file" | cut -d' ' -f1)" = "${digest#sha256:}" ]; then
    echo "have $file"; continue
  fi
  curl -sSL --retry 5 -H "Authorization: Bearer $TOK" -o "$file.part" \
    "https://registry-1.docker.io/v2/vllm/vllm-openai/blobs/$digest"
  got=$(sha256sum "$file.part" | cut -d' ' -f1)
  [ "$got" = "${digest#sha256:}" ] || { echo "sha mismatch $file: $got"; exit 1; }
  mv "$file.part" "$file"; echo "fetched $file $(stat -c %s "$file")"
done

# DFlash2 drafter
cd $D/models/dflash2
REV=a22a1744e85c17364dcc54a6a420af7135d4c7f6
for f in config.json model.safetensors; do
  [ -s $f ] || curl -sSL --retry 5 -o $f https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/resolve/$REV/$f
done
sha256sum model.safetensors | cut -c1-16
stat -c '%s %n' model.safetensors

# target model view with the original config
cd $D/models/intel-w4a16
SRC=$W/models/Qwen3.8-Flash-Next-W4A16-AutoRound
for f in $SRC/*; do
  name=$(basename "$f")
  case "$name" in config.json|config.json.autoround.bak) continue ;; esac
  ln -sfn "$f" "$name"
done
cp $SRC/config.json.autoround.bak config.json
grep -c qwen_sparse_attention config.json

# /kaggle/input
I=$D/input
mkdir -p $I/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default \
  $I/models/jonathanwang2022/qwen3-8-flash-next-dflash2/transformers/default \
  $I/models/lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/pytorch/hf-mixed-mtp-nvfp4
ln -sfn $D/runtime $I/vllm-main-e975732-arc3
ln -sfn $D/models/intel-w4a16 $I/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1
ln -sfn $D/models/dflash2 $I/models/jonathanwang2022/qwen3-8-flash-next-dflash2/transformers/default/1
ln -sfn $W/mtp_finetune/lord_mtp $I/models/lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/pytorch/hf-mixed-mtp-nvfp4/1
ln -sfn $W/bench_b200/results/208013/workload.jsonl $D/run/workload.jsonl
find $I -maxdepth 7 | sort
echo STAGED
