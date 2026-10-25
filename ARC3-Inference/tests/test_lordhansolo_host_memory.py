"""The watchdog's host-memory samples say what holds the RAM: the cgroup's memory.stat split,
resident memory per process role and /dev/shm."""
import json
import os
import re
from pathlib import Path

from inference.framework.lordhansolo import kaggle as lord


def _readers():
    """Run the rendered watchdog's memory readers."""
    script = lord._DUCK_VLLM_WATCHDOG_SCRIPT
    start = script.index("# memory.stat fields kept with each sample")
    end = script.index("def log_host_memory")
    namespace = {"Path": Path, "os": os}
    exec(script[start:end], namespace)
    return namespace


def _process(proc_root: Path, pid: int, cmdline: list[str], cwd: Path, anon_kb: int | None) -> None:
    proc = proc_root / str(pid)
    proc.mkdir()
    rss = f"RssAnon:\t{anon_kb} kB\nRssFile:\t100 kB\nRssShmem:\t4 kB\n" if anon_kb is not None else ""
    (proc / "status").write_text(f"Name:\tpython3\n{rss}Threads:\t1\n")
    (proc / "cmdline").write_bytes(b"\0".join(part.encode() for part in cmdline) + b"\0")
    (proc / "cwd").symlink_to(cwd)


def test_process_memory_is_summed_per_role(tmp_path):
    readers = _readers()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    (proc_root / "self").mkdir()
    sandbox_dir = tmp_path / "rgb_python_tool_abc"
    sandbox_dir.mkdir()
    _process(proc_root, 10, ["VLLM::EngineCore"], tmp_path, 2_000_000)
    _process(proc_root, 11, ["/usr/bin/python3", "-m", "colab_kernel_launcher", "-f", "k.json"], tmp_path, 3_000_000)
    _process(proc_root, 12, ["/usr/bin/python3", "-I", "-S", "-c", "import sys\nrun()"], sandbox_dir, 50_000)
    _process(proc_root, 13, ["/usr/bin/python3", "-I", "-S", "-c", "import sys"], sandbox_dir, 70_000)
    _process(proc_root, 14, ["python3", "/kaggle/working/vllm-watchdog.py", "c.json", "--prefetch", "0", "r"], tmp_path, 1_000)
    _process(proc_root, 15, ["kthreadd"], tmp_path, None)

    groups, top = readers["read_process_memory"](proc_root)

    assert groups["vllm_enginecore"] == {"count": 1, "anon_bytes": 2_000_000 * 1024, "file_bytes": 102_400, "shmem_bytes": 4096}
    assert groups["harness"]["anon_bytes"] == 3_000_000 * 1024
    assert groups["sandbox"]["count"] == 2 and groups["sandbox"]["anon_bytes"] == 120_000 * 1024
    assert groups["prefetch"]["count"] == 1
    assert sum(group["count"] for group in groups.values()) == 5
    assert [process["pid"] for process in top] == [11, 10, 13, 12, 14]
    assert "\n" not in top[2]["cmd"]


def test_cgroup_memory_stat_keeps_the_split(tmp_path):
    readers = _readers()
    (tmp_path / "memory.stat").write_text("anon 7\nfile 5\nshmem 2\nunevictable 1\npgfault 99\n")
    assert readers["read_cgroup_memory_stat"](tmp_path) == {"anon": 7, "file": 5, "shmem": 2, "unevictable": 1}
    v1 = tmp_path / "v1"
    (v1 / "memory").mkdir(parents=True)
    (v1 / "memory" / "memory.stat").write_text("rss 1\ntotal_rss 8\ntotal_cache 3\n")
    assert readers["read_cgroup_memory_stat"](v1) == {"total_rss": 8, "total_cache": 3}
    assert readers["read_cgroup_memory_stat"](tmp_path / "missing") == {}


def test_shm_reader_tolerates_a_missing_mount(tmp_path):
    readers = _readers()
    assert readers["read_shm_used_bytes"](str(tmp_path / "missing")) is None
    assert readers["read_shm_used_bytes"](str(tmp_path)) >= 0


def test_report_projects_growth_onto_the_rerun_play_time(tmp_path):
    from inference.tools.host_memory_report import load_samples, report

    gib = 1024**3
    (tmp_path / "metrics").mkdir()
    samples = []
    for minute in range(0, 121, 15):
        harness = (1 + minute / 60) * gib  # +1 GiB per hour
        samples.append({
            "ts": 1000 + minute * 60, "host_available_bytes": (40 - minute / 60) * gib,
            "cgroup_memory_stat": {"anon": 100 * gib, "file": 60 * gib, "shmem": gib},
            "process_groups": {"harness": {"count": 1, "anon_bytes": harness},
                               "vllm_enginecore": {"count": 1, "anon_bytes": 5 * gib}},
            "dev_shm_used_bytes": 0,
        })
    (tmp_path / "metrics" / "host_memory.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
    text = report(load_samples(tmp_path), [30, 60], from_minute=60, project_minutes=540)
    assert "harness" in text and "vllm_enginecore" in text
    assert re.search(r"^  harness +\+1\.00 +\+1\.00 +\+9\.00$", text, re.M)
