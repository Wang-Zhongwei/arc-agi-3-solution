"""mtp_data keeps every game of the runs served by the drafter's target and holds out games."""
import json
from pathlib import Path

from inference.rft import mtp_data, rft_data

INTEL = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround"


def _run(root: Path, name: str, target: str | None, games=("ar25-0c556536_p0", "ls20-9607627b_p0"), env=None) -> Path:
    run = root / "kaggle" / name
    (run / "transcripts").mkdir(parents=True)
    (run / "artifacts").mkdir()
    for stem in games:
        (run / "transcripts" / f"{stem}.txt").write_text("")
        (run / "artifacts" / f"{stem}_events.jsonl").write_text("")
    if target:
        (run / "kernel").mkdir()
        (run / "kernel" / "kernel-metadata.json").write_text(json.dumps({"model_sources": [f"{target}/Transformers/default/1"]}))
    agent = run / "src/ARC3-Inference/inference/agent"
    agent.mkdir(parents=True)
    (agent / "vision_context.py").write_text("")
    (run / "intermediate_states.pkl").write_bytes(b"")
    (run / "taaf_setup_env.json").write_text(json.dumps(env or {}))
    return run


def test_run_target_reads_kernel_metadata_or_setup_command(tmp_path):
    run = _run(tmp_path, "a/b/v01", INTEL)
    assert mtp_data.run_target(run) == INTEL
    other = _run(tmp_path, "a/c/v01", None)
    (other / "setup_commands.json").write_text(json.dumps(
        {"setup": "MODEL_SOURCE_PARTS = ('lordhansolo', 'qwen3-8-flash-next-mixed-nvfp4-fp8', 'PyTorch')"}))
    assert mtp_data.run_target(other) == "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8"


def test_jobs_keep_target_runs_and_split_by_game(tmp_path):
    _run(tmp_path, "me/good/v01", INTEL)
    _run(tmp_path, "me/lord/v01", "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8")
    _run(tmp_path, "me/view/v01", INTEL, env={"ARC3_VIEW_TOOL": "1"})
    jobs, problems = mtp_data.game_jobs(tmp_path, INTEL, ("ar25",))
    assert [a.stem for a in jobs["train"][str(tmp_path / "kaggle/me/good/v01")]] == ["ls20-9607627b_p0"]
    assert [a.stem for a in jobs["eval"][str(tmp_path / "kaggle/me/good/v01")]] == ["ar25-0c556536_p0"]
    assert all(a.level == rft_data.ALL_LEVELS for split in jobs.values() for attempts in split.values() for a in attempts)
    assert problems[str(tmp_path / "kaggle/me/lord/v01")].startswith("target lordhansolo/")
    assert "view tool" in problems[str(tmp_path / "kaggle/me/view/v01")]


def test_game_priors_do_not_exclude_a_run(tmp_path):
    run = _run(tmp_path, "jonathanwang2022/priority-falloff/v03", INTEL)
    assert rft_data.run_problem(run)
    assert mtp_data.run_problem(run, INTEL) is None
