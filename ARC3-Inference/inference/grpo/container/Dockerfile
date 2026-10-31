# GRPO node image (inference/grpo/README.md): the rollout server's vLLM nightly (CUDA 13.0, torch 2.14, Python 3.12)
# plus the trainers' venv (torch 2.8 stack) and the harness venv (uv). Code, venvs and the fla patch are baked in; the
# checkpoint, the game files and the run directory are mounts. Build from the repo root, tracked files only:
#   git archive --format=tar HEAD ARC3-Inference tufa-arc-agi-framework docs/RL | docker build \
#     --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t arc3-grpo:$(git rev-parse --short HEAD) \
#     -f ARC3-Inference/inference/grpo/container/Dockerfile -
FROM vllm/vllm-openai@sha256:d428631bb917a33d22308265d1b8be91522f862bf9fd133bee62dde3313664ae

# sm100 (B200): fla's GDN backward unpipelined (pod/patch_fla_sm100.py, a no-op on other GPUs).
# sm90 (H200): also tilelang, which fla's GDN backward needs there (fla #640).
ARG GPU_ARCH=sm100
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT
ENV REPO=/opt/repo TRAINER_VENV=/opt/trainer-venv GIT_COMMIT=$GIT_COMMIT

# the trainers' tested stack (inference/rft/pod/requirements-pod.txt), not the image's torch 2.14
COPY ARC3-Inference/inference/rft/pod/requirements-pod.txt /tmp/requirements-pod.txt
RUN python3 -m venv $TRAINER_VENV \
 && $TRAINER_VENV/bin/python -m pip install -q --no-cache-dir --upgrade pip \
 && $TRAINER_VENV/bin/python -m pip install -q --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
 && $TRAINER_VENV/bin/python -m pip install -q --no-cache-dir -r /tmp/requirements-pod.txt safetensors wandb \
 && if [ "$GPU_ARCH" = sm90 ]; then $TRAINER_VENV/bin/python -m pip install -q --no-cache-dir tilelang==0.1.15; fi
COPY ARC3-Inference/inference/grpo/pod/patch_fla_sm100.py /tmp/patch_fla_sm100.py
RUN $TRAINER_VENV/bin/python /tmp/patch_fla_sm100.py && date -u > $TRAINER_VENV/INSTALL_DONE

# the harness venv from uv.lock (TAAF editable from ../tufa-arc-agi-framework); the base image's UV_OVERRIDE pins
# vLLM's own packages and is not meant for this venv
COPY . $REPO
WORKDIR $REPO/ARC3-Inference
RUN env -u UV_OVERRIDE uv sync --frozen \
 && env -u UV_OVERRIDE uv pip install -q wandb \
 && uv run --no-sync python -c "import arcengine, arc_agi, taaf"

ENTRYPOINT []
CMD ["bash"]
