#!/bin/bash
# Upload a checkpoint folder as a private Kaggle Model instance, in the layout serving already uses
# (OWNER/SLUG/Transformers/default/N, like dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1).
#   bash kaggle_model_upload.sh MODEL_DIR OWNER SLUG "Title" "Description"
# Credentials: ~/.kaggle/access_token (KAGGLE_API_TOKEN) and the username; never printed. Creates the model
# if it does not exist, then uploads the folder's files (subfolders such as .cache are skipped).
set -euo pipefail
DIR=$1 OWNER=$2 SLUG=$3 TITLE=$4 DESCRIPTION=${5:-}
export KAGGLE_USERNAME=$OWNER
[ -n "${KAGGLE_API_TOKEN:-}" ] || export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
meta=$(mktemp -d)
python3 - "$meta" "$OWNER" "$SLUG" "$TITLE" "$DESCRIPTION" <<'PY'
import json, sys
meta, owner, slug, title, description = sys.argv[1:]
json.dump(dict(ownerSlug=owner, title=title, slug=slug, subtitle="", isPrivate=True, description=description,
               publishTime="", provenanceSources=""), open(f"{meta}/model-metadata.json", "w"), indent=1)
PY
kaggle models get "$OWNER/$SLUG" > /dev/null 2>&1 || kaggle models create -p "$meta"
python3 - "$DIR" "$OWNER" "$SLUG" "$DESCRIPTION" <<'PY'
import json, sys
directory, owner, slug, description = sys.argv[1:]
json.dump(dict(ownerSlug=owner, modelSlug=slug, instanceSlug="default", framework="transformers",
               overview=description, usage="", licenseName="Apache 2.0", fineTunable=False, trainingData=[],
               modelInstanceType="Unspecified", baseModelInstanceId=0, externalBaseModelUrl=""),
          open(f"{directory}/model-instance-metadata.json", "w"), indent=1)
PY
kaggle models instances create -p "$DIR"
echo "uploaded: $OWNER/$SLUG/Transformers/default"
