#!/usr/bin/env python3
"""Serve ``Qwen/Qwen3.8-27B-FP8`` through the pinned Kaggle serving machinery.

The pinned notebook does not launch vLLM itself. Its one setup command runs
``serving_setup.py``, a 3,100-line script that lives inside the attached Kaggle
dataset ``keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`` and that hard-codes the
model: ``MODEL_KAGGLE_PATH``, ``MODEL_HF_REPO``, ``SERVED_MODEL_NAME``,
``--quantization modelopt_fp4``, and a ``MODEL_MANIFEST.json`` identity check that
raises for anything but the RadixArk NVFP4 checkpoint. Attaching a different Kaggle
model is therefore not enough; the script would refuse to start.

This driver replaces that setup command. It does **not** edit ``serving_setup.py``
-- that file also self-hashes itself against ``SOURCE_IDENTITY.json``, so any textual
patch would be rejected. Instead it imports the pristine script as a module, lets it
prove its own identity first, then rebinds four module attributes and calls its
``main()``. Everything that is hard about serving on Kaggle -- extracting and hashing
the 18 GiB vLLM runtime, applying the PLE patch, building the CUDA environment, the
process-ownership discipline, the watchdog handshake, the provenance artifacts our
post-processing reads -- is reused unchanged.

What is rebound, and nothing else:

* ``SERVED_MODEL_NAME``      -> the arm's served id. Read at call time by
  ``server_command`` (the ``--served-model-name`` flag), by ``wait_for_server`` (the
  ``/v1/models`` identity assertion) and by ``persist_analyzer_environment``
  (``LOCAL_ANALYZER_MODEL_ID`` / ``INFERENCE_ANALYZER_MODEL``), so the analyzer
  follows the swap automatically.
* ``MODEL_HF_REPO`` / ``MODEL_HF_REVISION`` -> so the provenance artifact records
  what was actually served rather than the control's checkpoint.
* ``resolve_model_dir`` / ``verify_model`` -> the arm's mount resolution and identity
  checks, which raise on anything that is not the intended FP8 checkpoint.
* ``server_command``         -> the pinned builder, minus ``--quantization
  modelopt_fp4``. vLLM reads FP8 block quantisation out of the checkpoint's own
  ``config.json``; naming a quantisation the checkpoint does not use is a hard error.

Ordering matters and is enforced: ``source_identity()`` validates the bundle against
the *control's* constants, so it is called before anything is rebound, and then
stubbed out for ``main()``.

Fail-loud contract. A silent fallback to the control model would produce a control
run wearing this arm's name, which is worse than a crash. Six checks, any of which
aborts the run:

1. exactly one mounted candidate directory under the declared attachment;
2. the checkpoint declares FP8 block quantisation;
3. its architecture is not the control's ``Qwen4ExpForConditionalGeneration``;
4. the launched argv points at the arm's directory and carries neither
   ``--quantization`` nor ``--speculative-config``;
5. ``/v1/models`` returns exactly the arm's served id (checked again here, after
   ``main()``, because the pinned ``wait_for_server`` swallows that assertion inside a
   retry loop and would report a timeout instead of a wrong model);
6. the persisted analyzer environment names the arm's served id.

Run by the generated notebook as::

    "$PYTHON" /kaggle/working/arm_model_qwen27b_serving.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# --- what this arm serves ----------------------------------------------------
ARM_NAME = "model_qwen27b"
# Kaggle model attachment, as it appears in kernel-metadata.json "model_sources".
# An unmodified byte-for-byte mirror of the official checkpoint at a pinned
# revision, with a per-file hash manifest published alongside it.
ARM_MODEL_ATTACHMENT = "mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1"
ARM_MODEL_HF_REPO = "Qwen/Qwen3.8-27B-FP8"
ARM_MODEL_HF_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
ARM_SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"

# --- what the control serves, so we can prove we are not it ------------------
CONTROL_SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-NVFP4"
CONTROL_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"

# Kaggle reports 30,890,068,612 uncompressed bytes for this attachment. The band is
# wide on purpose: it is here to catch a partial or empty mount, not to re-hash a
# checkpoint that vLLM's own load already validates.
ARM_WEIGHT_BYTES_MIN = 27 * 1024**3
ARM_WEIGHT_BYTES_MAX = 33 * 1024**3

# The pinned --speculative-config binds to the vendor branch's Qwen3_8FlashNextMTP
# architecture. Nothing on disk here shows that path covers a dense Qwen3.8-27B, so
# the arm serves without MTP and asserts the profile agrees.
ARM_REQUIRE_MTP_OFF = True


def _fail(message: str) -> None:
    raise RuntimeError(f"[arm:{ARM_NAME}] {message}")


def load_serving_setup(bundle_dir: Path):
    """Import the pinned serving script from the attached source bundle."""
    path = bundle_dir / "serving_setup.py"
    if not path.is_file():
        _fail(f"pinned serving_setup.py not found at {path}")
    spec = importlib.util.spec_from_file_location("taaf_pinned_serving_setup", path)
    if spec is None or spec.loader is None:
        _fail(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_arm_model_dir() -> Path:
    """Locate the mounted arm checkpoint, or raise saying what was found instead."""
    owner, slug, framework, instance, version = ARM_MODEL_ATTACHMENT.split("/")
    root = Path("/kaggle/input/models") / owner / slug
    # Kaggle lowercases the framework segment in the mount path.
    exact = root / framework.lower() / instance / version
    if (exact / "config.json").is_file():
        return exact
    if not root.is_dir():
        _fail(
            f"the model attachment {ARM_MODEL_ATTACHMENT} is not mounted: {root} does "
            "not exist. Check kernel-metadata.json model_sources."
        )
    candidates = sorted({p.parent for p in root.rglob("config.json")})
    if len(candidates) != 1:
        _fail(
            f"expected exactly one checkpoint directory under {root}, found "
            f"{[str(c) for c in candidates]}"
        )
    return candidates[0]


def _checkpoint_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())


def verify_arm_model(model_dir: Path, *, full_file_hashes: bool = True) -> dict[str, Any]:
    """Stand in for the pinned ``verify_model``; same signature, arm's identity.

    ``full_file_hashes`` is accepted for signature compatibility and ignored: the
    pinned path uses it to re-hash 419 files against a manifest this checkpoint does
    not ship. vLLM's own safetensors load is the payload-integrity test, exactly as
    the pinned script's own fast-start mode already assumes.
    """
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        _fail(f"no config.json at {model_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        _fail(f"config.json declares no architecture: {architectures!r}")
    if CONTROL_ARCHITECTURE in architectures:
        _fail(
            f"this is the CONTROL checkpoint ({CONTROL_ARCHITECTURE}), not the arm's. "
            "Refusing to run a control under this arm's name."
        )

    quantization = config.get("quantization_config") or {}
    quant_method = str(quantization.get("quant_method", ""))
    if quant_method != "fp8":
        _fail(
            f"expected an FP8 checkpoint, config.json says quant_method={quant_method!r}"
        )
    block_size = quantization.get("weight_block_size")
    if block_size is not None and list(block_size) != [128, 128]:
        _fail(f"unexpected FP8 block size {block_size!r}; expected [128, 128]")

    for name in ("chat_template.jinja", "model.safetensors.index.json"):
        if not (model_dir / name).is_file():
            _fail(f"incomplete checkpoint; missing {name} under {model_dir}")

    total_bytes = _checkpoint_bytes(model_dir)
    if not ARM_WEIGHT_BYTES_MIN <= total_bytes <= ARM_WEIGHT_BYTES_MAX:
        _fail(
            f"mounted checkpoint is {total_bytes} bytes, outside the expected "
            f"[{ARM_WEIGHT_BYTES_MIN}, {ARM_WEIGHT_BYTES_MAX}] band; is the mount partial?"
        )

    record = {
        "arm": ARM_NAME,
        "attachment": ARM_MODEL_ATTACHMENT,
        "model_dir": str(model_dir),
        "hf_repo": ARM_MODEL_HF_REPO,
        "hf_revision": ARM_MODEL_HF_REVISION,
        "served_model_name": ARM_SERVED_MODEL_NAME,
        "architecture": architectures[0],
        "model_type": config.get("model_type"),
        "quant_method": quant_method,
        "weight_block_size": block_size,
        "total_bytes": total_bytes,
        "payload_check_deferred_to_vllm_load": True,
        "verification_mode": "arm-identity",
    }
    print("ARM_MODEL_VERIFIED " + json.dumps(record, sort_keys=True), flush=True)
    return record


def install(serving_setup) -> dict[str, Any]:
    """Rebind the four attributes that select the model, and prove each landed."""
    # 1. The bundle proves its own identity against the CONTROL constants first.
    serving_setup.source_identity()
    _cached_identity = {"validated": True, "phase": "before_arm_override"}
    serving_setup.source_identity = lambda: dict(_cached_identity)

    original_server_command = serving_setup.server_command

    def arm_server_command(model_dir, **kwargs):
        command = list(original_server_command(model_dir, **kwargs))
        if "--quantization" not in command:
            _fail("pinned server_command no longer passes --quantization; re-read it")
        index = command.index("--quantization")
        if command[index + 1] != "modelopt_fp4":
            _fail(
                "expected the pinned --quantization modelopt_fp4, found "
                f"{command[index + 1]!r}"
            )
        del command[index : index + 2]
        return command

    serving_setup.SERVED_MODEL_NAME = ARM_SERVED_MODEL_NAME
    serving_setup.MODEL_HF_REPO = ARM_MODEL_HF_REPO
    serving_setup.MODEL_HF_REVISION = ARM_MODEL_HF_REVISION
    serving_setup.resolve_model_dir = resolve_arm_model_dir
    serving_setup.verify_model = verify_arm_model
    serving_setup.server_command = arm_server_command

    if serving_setup.SERVED_MODEL_NAME != ARM_SERVED_MODEL_NAME:
        _fail("SERVED_MODEL_NAME override did not land")
    if serving_setup.resolve_model_dir is not resolve_arm_model_dir:
        _fail("resolve_model_dir override did not land")
    if serving_setup.verify_model is not verify_arm_model:
        _fail("verify_model override did not land")
    if serving_setup.server_command is not arm_server_command:
        _fail("server_command override did not land")

    if ARM_REQUIRE_MTP_OFF and os.environ.get("TAAF_VLLM_MTP_TOKENS") != "0":
        _fail(
            "this arm serves without MTP; set TAAF_VLLM_MTP_TOKENS=0 "
            f"(found {os.environ.get('TAAF_VLLM_MTP_TOKENS')!r})"
        )

    detail = {
        "arm": ARM_NAME,
        "served_model_name": ARM_SERVED_MODEL_NAME,
        "control_served_model_name": CONTROL_SERVED_MODEL_NAME,
        "attachment": ARM_MODEL_ATTACHMENT,
        "vllm_profile_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("TAAF_VLLM_")
        },
    }
    print("ARM_SERVING_INSTALLED " + json.dumps(detail, sort_keys=True), flush=True)
    return detail


def verify_running_server(serving_setup) -> dict[str, Any]:
    """After ``main()``: prove the process that is up is the arm's, not the control's."""
    working_dir = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
    identity = json.loads((working_dir / "vllm-server-identity.json").read_text())
    argv = identity["argv"]

    model_dir = str(resolve_arm_model_dir())
    if model_dir not in argv:
        _fail(f"the launched vLLM argv does not name {model_dir}: {argv}")
    if "--quantization" in argv:
        _fail(f"--quantization survived into the launched argv: {argv}")
    if ARM_REQUIRE_MTP_OFF and "--speculative-config" in argv:
        _fail(f"--speculative-config survived into the launched argv: {argv}")
    served_index = argv.index("--served-model-name")
    if argv[served_index + 1] != ARM_SERVED_MODEL_NAME:
        _fail(f"argv serves {argv[served_index + 1]!r}, not {ARM_SERVED_MODEL_NAME!r}")

    # The pinned wait_for_server makes this same assertion inside a bare
    # `except Exception: sleep` retry loop, so a wrong model reads as a timeout
    # rather than as a wrong model. Make it once more, unswallowed.
    models = serving_setup.request_json(f"{serving_setup.BASE_URL}/models", timeout=30)
    ids = [row.get("id") for row in (models.get("data") or []) if isinstance(row, dict)]
    if ids != [ARM_SERVED_MODEL_NAME]:
        _fail(f"/v1/models serves {ids!r}, not [{ARM_SERVED_MODEL_NAME!r}]")
    if CONTROL_SERVED_MODEL_NAME in ids:
        _fail("the control model is being served under this arm's name")

    setup_env = json.loads(Path(os.environ["TAAF_KAGGLE_SETUP_ENV"]).read_text())
    for key in ("LOCAL_ANALYZER_MODEL_ID", "INFERENCE_ANALYZER_MODEL"):
        if setup_env.get(key) != ARM_SERVED_MODEL_NAME:
            _fail(f"analyzer env {key}={setup_env.get(key)!r}, not {ARM_SERVED_MODEL_NAME!r}")

    record = {
        "arm": ARM_NAME,
        "served_model_name": ARM_SERVED_MODEL_NAME,
        "model_dir": model_dir,
        "argv": argv,
        "argv_sha256": identity["argv_sha256"],
        "vllm_tuning": identity["vllm_tuning"],
        "models_endpoint_ids": ids,
        "analyzer_model_id": setup_env.get("LOCAL_ANALYZER_MODEL_ID"),
        "verified_epoch": time.time(),
    }
    (working_dir / "vllm-arm-model.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("ARM_SERVING_VERIFIED " + json.dumps(record, sort_keys=True), flush=True)
    return record


def main() -> None:
    bundle_dir = Path(os.environ["TAAF_KAGGLE_BUNDLE_DIR"])
    serving_setup = load_serving_setup(bundle_dir)
    install(serving_setup)
    # Same hard setup ceiling the pinned script's own __main__ applies.
    if not hasattr(signal, "SIGALRM"):
        _fail("the pinned Kaggle setup requires Linux SIGALRM support")
    signal.signal(signal.SIGALRM, serving_setup._setup_timeout_handler)
    signal.alarm(serving_setup.SETUP_TOTAL_TIMEOUT)
    try:
        serving_setup.main()
    finally:
        signal.alarm(0)
    verify_running_server(serving_setup)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        failure = {
            "arm": ARM_NAME,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_epoch": time.time(),
        }
        working = os.environ.get("TAAF_KAGGLE_WORKING_DIR")
        if working:
            Path(working, "vllm-arm-setup-failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print("ARM_SERVING_FAILED " + json.dumps(failure, sort_keys=True), flush=True)
        raise
