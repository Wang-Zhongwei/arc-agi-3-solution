"""The animation layers through the real game loop (tests/anim_loop_sim.py).

Each sim runs in a subprocess under the Kaggle notebook's own setup_env, read
from kaggle/template.ipynb (priority scheduling on), plus the configuration's
flags. Two windows: `wide` (the user-combo arm's 96k window, nothing trimmed)
and `tight` (30k, so history is trimmed several times). `tight-arm-dedup` adds
ARC3_OPENER_DEDUP=1 (prefill-v2), which may be stacked on the arm at build time.

Checks: the game plays through (yields, a death and its auto-RESET, a level
up); every request's rendered prompt extends the previous one unless history
was evicted in between; the system prompt and tools never change within a run;
the digest lands once, only in a fresh turn opener (never a yield resumption
or a tool result); the echo says ANIMATED under L1; the ring survives a trim;
with every new flag off the requests are byte-identical to the base branch
(ANIM_LOOP_BASELINE_REF, default origin/arm/throughput-stack, or
ANIM_LOOP_BASELINE_ROOT=<another checkout's ARC3-Inference>).

Mutation checks apply a known bug to a copy of the harness and require these
checks or the layer unit tests to catch it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_opener_dedup_loop import notebook_env

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
TIGHT_WINDOW = "30000"
ARM_CONTEXT = {"LOCAL_ANALYZER_CONTEXT_WINDOW": "98304", "ARC3_CONTEXT_DRAIN_TOKENS": "43008"}
TRUTH = {"ARC3_ANIM_TRUTH": "1"}
RING = {"ARC3_ANIM_RING": "8"}
DIGEST = {"ARC3_ANIM_TRUTH": "1", "ARC3_ANIM_RING": "8", "ARC3_ANIM_DIGEST": "1"}
# kaggle/arms.json user-combo's env (context and drain as in ARM_CONTEXT)
ARM = {**DIGEST, **ARM_CONTEXT, "ARC3_MAX_ACTIVE_STREAMS": "14", "ARC3_LEDGER_ON_TRIM": "1",
       "LOCAL_ANALYZER_TEMPERATURE": "0.65"}
# the layer knobs with every layer flag off: must change nothing
KNOBS_OFF = {"ARC3_ANIM_TRUTH": "0", "ARC3_ANIM_RING": "0", "ARC3_ANIM_DIGEST": "0",
             "ARC3_ANIM_DIGEST_GATE": "all", "ARC3_ANIM_DIGEST_CHARS": "5000",
             "ARC3_ANIM_KEYFRAMES": "4", "ARC3_ANIM_NOOP_EXEMPT": "0"}
TIGHT = {"LOCAL_ANALYZER_CONTEXT_WINDOW": TIGHT_WINDOW}

CONFIGS = {
    "wide-off": ARM_CONTEXT,
    "wide-truth": {**TRUTH, **ARM_CONTEXT},
    "wide-ring": {**RING, **ARM_CONTEXT},
    "wide-digest": {**DIGEST, **ARM_CONTEXT},
    "wide-arm": ARM,
    "tight-off": TIGHT,
    "tight-ledger": {**TIGHT, "ARC3_LEDGER_ON_TRIM": "1"},
    "tight-digest": {**DIGEST, **TIGHT},
    "tight-arm": {**ARM, **TIGHT},
    "tight-arm-reset": {**ARM, **TIGHT, "EXPOSE_RESET": "on", "ANIM_SIM_RESET": "1"},
    "tight-arm-dedup": {**ARM, **TIGHT, "ARC3_OPENER_DEDUP": "1"},
}
RING_ON = {"wide-ring", "wide-digest", "wide-arm", "tight-digest", "tight-arm", "tight-arm-reset", "tight-arm-dedup"}
TRUTH_ON = {"wide-truth", "wide-digest", "wide-arm", "tight-digest", "tight-arm", "tight-arm-reset", "tight-arm-dedup"}
DIGEST_ON = {"wide-digest", "wide-arm", "tight-digest", "tight-arm", "tight-arm-reset", "tight-arm-dedup"}


def run_sim(tmp: Path, name: str, extra: dict[str, str], root: Path = ROOT) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ARC3_", "LOCAL_ANALYZER_", "EXPOSE_", "ANIM_SIM_"))}
    env.update(notebook_env())
    env.update(extra)
    env["PYTHONPATH"] = str(root)
    out = tmp / f"{name}.json"
    proc = subprocess.run([sys.executable, str(HERE / "anim_loop_sim.py"), str(out), "--inference-root", str(root)],
                          env=env, cwd=root, timeout=600, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(out.read_text())
    assert result["inference_file"].startswith(str(root)), result["inference_file"]
    return result


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anim")
    return {name: run_sim(tmp, name, extra) for name, extra in CONFIGS.items()}


# ---------------------------------------------------------------------------- check functions
# Each returns a list of violations, so the mutation checks can require them to be non-empty.

def texts(message) -> list[str]:
    c = message.get("content")
    if isinstance(c, str):
        return [c]
    return [p.get("text") or "" for p in c or [] if isinstance(p, dict) and p.get("type") == "text"]


def check_extends(run) -> list[str]:
    bad = []
    reqs = run["requests"]
    for a, b in zip(reqs, reqs[1:]):
        if b["evictions"] != a["evictions"]:
            continue
        if not b["prompt"].startswith(a["prompt"]):
            bad.append(f"reply {b['reply']}: prompt does not extend reply {a['reply']}")
        if b["messages"][:len(a["messages"])] != a["messages"]:
            bad.append(f"reply {b['reply']}: an earlier message changed without an eviction")
    return bad


def check_static_system(run) -> list[str]:
    systems = {json.dumps(r["messages"][0]) for r in run["requests"]}
    tools = {json.dumps(r["tools"]) for r in run["requests"]}
    return ([f"{len(systems)} different system messages"] if len(systems) != 1 else []) + \
           ([f"{len(tools)} different tool lists"] if len(tools) != 1 else [])


_LEDGER_HEAD = re.compile(r"\A<memory>\n.*?\n</memory>\n\n", re.S)


def unique_messages(run, role: str) -> list[dict]:
    """Every distinct message of ``role`` sent in the run, in first-seen order (a ledger
    block a trim put at the head of a user message is ignored)."""
    out, seen = [], set()
    for r in run["requests"]:
        for m in r["messages"]:
            if m["role"] != role:
                continue
            key = _LEDGER_HEAD.sub("", "\n".join(texts(m))) + str(m.get("tool_call_id", ""))
            if key not in seen:
                seen.add(key)
                out.append(m)
    return out


def check_digest_placement(run, expect: bool) -> list[str]:
    """Digests appear only in fresh-turn openers (never a resumption opener, a tool result
    or the system prompt), each block in exactly one opener."""
    bad = []
    openers, resumption_openers = set(), set()
    for inv in run["invocations"]:
        if inv["requests"] == 0:
            continue
        opener = _LEDGER_HEAD.sub("", "\n".join(texts(run["requests"][inv["first_request"]]["messages"][-1])))
        (resumption_openers if inv["resumed"] else openers).add(opener)
    blocks: dict[str, int] = {}
    for message in unique_messages(run, "user"):
        text = _LEDGER_HEAD.sub("", "\n".join(texts(message)))
        found = re.findall(r"^Animation digest.*$", text, re.M)
        if not found:
            continue
        if text in resumption_openers:
            bad.append("a digest in a yield-resumption opener")
        elif text not in openers:
            bad.append("a digest in a user message that is not a fresh turn opener")
        for block in found:
            blocks[block] = blocks.get(block, 0) + 1
    for m in unique_messages(run, "tool"):
        if any("Animation digest (" in t for t in texts(m)):
            bad.append("a digest in a tool result")
    if any("Animation digest (" in t for t in texts(run["requests"][0]["messages"][0])):
        bad.append("a digest in the system prompt")
    if any(count != 1 for count in blocks.values()):
        bad.append("a digest block repeats in two messages")
    if expect and not blocks:
        bad.append("no digest fired")
    if not expect and blocks:
        bad.append("a digest fired with the digest off")
    return bad


def tool_stdout(run) -> str:
    return "\n".join(t for m in unique_messages(run, "tool") for t in texts(m))


def check_echo(run, truth: bool) -> list[str]:
    out = tool_stdout(run)
    animated = "MOUSE(row=10, col=10) -> ANIMATED (blink; board area unchanged; 2 frames" in out
    noop = "MOUSE(row=10, col=10) -> NO-OP" in out
    if truth and not (animated and not noop):
        return ["truth on: the hiding click is not echoed as ANIMATED"]
    if not truth and not (noop and not animated):
        return ["truth off: the hiding click is not echoed as NO-OP"]
    if "SPACE -> NO-OP" not in out:
        return ["a real no-op is not echoed as NO-OP"]
    return []


def check_ring_survives_trim(run) -> list[str]:
    first = next((i for i, r in enumerate(run["requests"]) if r["evictions"] > 0), None)
    if first is None:
        return ["no trim happened"]
    actions_at_trim = run["requests"][first]["actions"]
    match = re.search(r"RINGFRAMES (\d+) (\d+)", tool_stdout(run))
    if not match:
        return ["no ring probe after the trim"]
    number, frames = int(match.group(1)), int(match.group(2))
    if frames <= 0:
        return [f"animation_for({number}) returned no frames after the trim"]
    if number > actions_at_trim:
        return [f"action {number} ran after the trim ({actions_at_trim}); the probe proves nothing"]
    return []


def comparable(run) -> list:
    return [{k: r[k] for k in ("messages", "tools", "prompt", "evictions", "reply")} for r in run["requests"]]


# ---------------------------------------------------------------------------- tests

def test_game_plays_through(runs):
    for name, run in runs.items():
        assert run["replies_used"] >= run["script_len"], name
        assert run["levels_completed"] == 1, name
        engine = run["engine_actions"]
        assert "ACTION2" in engine and engine[engine.index("ACTION2") + 1] == "RESET", f"{name}: no auto-RESET"
        inv = run["invocations"]
        assert sum(1 for x in inv if x["resumed"]) >= 2, f"{name}: yields must be resumed"
        if name.startswith("wide"):
            assert run["requests"][-1]["evictions"] == 0, f"{name}: the wide window must not trim"
        else:
            reqs = run["requests"]
            assert sum(1 for a, b in zip(reqs, reqs[1:]) if b["evictions"] != a["evictions"]) >= 2, name
    # L1 lets the hiding clicks through the batch no-op stop; without it the batch stops at the first
    clicks = lambda run: run["engine_actions"].count("ACTION6")  # noqa: E731
    assert clicks(runs["wide-truth"]) > clicks(runs["wide-off"])
    assert runs["tight-arm-reset"]["engine_actions"].count("RESET") == 2  # the auto-RESET and the model's


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_each_prompt_extends_the_previous_one(runs, name):
    assert check_extends(runs[name]) == []


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_system_prompt_and_tools_are_static(runs, name):
    assert check_static_system(runs[name]) == []


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_digest_placement(runs, name):
    assert check_digest_placement(runs[name], name in DIGEST_ON) == []


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_echo(runs, name):
    assert check_echo(runs[name], name in TRUTH_ON) == []


@pytest.mark.parametrize("name", ["tight-digest", "tight-arm", "tight-arm-reset", "tight-arm-dedup"])
def test_ring_survives_the_trim(runs, name):
    assert check_ring_survives_trim(runs[name]) == []


def test_ring_probe_without_ring(runs):
    assert "RINGPROBE none" in tool_stdout(runs["tight-off"])


def test_ring_reads_in_the_loop(runs):
    out = tool_stdout(runs["wide-ring"])
    assert "['animation_for', 'animations']" in out
    assert re.search(r"FOR (\d+) \1\b", out)  # animation_for(n) in a later call, timeline numbered n
    # the fatal animation, read after the auto-RESET
    assert "2 DOWN" in out
    assert "2 DOWN" in tool_stdout(runs["wide-off"])


def test_model_reset_keeps_the_animation(runs):
    """EXPOSE_RESET=on: a model-issued RESET is not an animation and does not clear the ring view."""
    out = tool_stdout(runs["tight-arm-reset"])
    assert "[action] RESET ->" in out


def test_digest_cost(runs):
    sizes = []
    for name in sorted(DIGEST_ON):
        for message in unique_messages(runs[name], "user"):
            text = _LEDGER_HEAD.sub("", "\n".join(texts(message)))
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("Animation digest"):
                    block = [line]
                    for nxt in lines[i + 1:]:
                        if nxt.startswith(("- ", "action ", "(+", "Trails")):
                            block.append(nxt)
                        else:
                            break
                    sizes.append(len("\n".join(block)))
    assert sizes and max(sizes) <= 600
    print(f"digest chars: n={len(sizes)} median={sorted(sizes)[len(sizes) // 2]} max={max(sizes)}")


def baseline_root(tmp_path: Path) -> Path:
    explicit = os.environ.get("ANIM_LOOP_BASELINE_ROOT", "")
    if explicit:
        return Path(explicit)
    ref = os.environ.get("ANIM_LOOP_BASELINE_REF", "origin/arm/throughput-stack")
    out = tmp_path / "baseline"
    out.mkdir()
    try:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", ref, "ARC3-Inference/inference"],
                                 check=True, capture_output=True, timeout=120).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"no baseline checkout ({exc}); set ANIM_LOOP_BASELINE_ROOT")
    subprocess.run(["tar", "-x", "-C", str(out)], input=archive, check=True)
    return out / "ARC3-Inference"


@pytest.mark.parametrize("name,extra", [
    ("wide-off", CONFIGS["wide-off"]),
    ("tight-off", CONFIGS["tight-off"]),
    ("tight-ledger", CONFIGS["tight-ledger"]),
])
def test_flags_off_matches_base_branch(runs, tmp_path, name, extra):
    base = run_sim(tmp_path, f"base-{name}", extra, root=baseline_root(tmp_path))
    assert comparable(base) == comparable(runs[name])


def test_layer_knobs_with_layers_off_match_base_branch(runs, tmp_path):
    base = runs["tight-off"]
    knobs = run_sim(tmp_path, "knobs-off", {**CONFIGS["tight-off"], **KNOBS_OFF})
    assert comparable(knobs) == comparable(base)


# ---------------------------------------------------------------------------- mutation checks

# (file under inference/, old, new): each must be caught by a loop check or the layer unit tests.
MUTANTS = {
    # M1: the digest goes into the system prompt instead of the opener
    "M1-digest-in-system": ("agent/tool_agent.py",
        "        if _anim_digest_enabled():\n            prompt += ANIMATION_DIGEST_ADDENDUM\n",
        "        if _anim_digest_enabled():\n            prompt += ANIMATION_DIGEST_ADDENDUM\n"
        "            prompt += 'Animation digest (novel): ' + str(__import__('time').time_ns())\n"),
    # M2: Wang-style stripping of earlier digests from history
    "M2-strip-history": ("agent/tool_agent.py",
        "        \"\"\"Put the ledger at the head of history once, after a trim.\"\"\"\n",
        "        \"\"\"Put the ledger at the head of history once, after a trim.\"\"\"\n"
        "        def _strip(m):\n"
        "            c = m.get('content')\n"
        "            cut = lambda s: re.sub(r'(?m)^Animation digest.*\\n?', '', s)\n"
        "            if isinstance(c, str):\n"
        "                return {**m, 'content': cut(c)}\n"
        "            if isinstance(c, list):\n"
        "                return {**m, 'content': [({**p, 'text': cut(p['text'])} if p.get('type') == 'text'"
        " else p) for p in c]}\n"
        "            return m\n"
        "        messages = [messages[0], *[_strip(m) for m in messages[1:-1]], *messages[-1:]]\n"),
    # M3: a layer guard dropped (truth always on)
    "M3-truth-always-on": ("framework/solver.py",
        'ANIM_TRUTH = ANIMATION_ENABLED and os.environ.get(\n    "ARC3_ANIM_TRUTH", "0"\n',
        'ANIM_TRUTH = ANIMATION_ENABLED or os.environ.get(\n    "ARC3_ANIM_TRUTH", "0"\n'),
    # M4: a yield resumption's (stale-summary) opener repeats the turn's digest
    "M4-digest-on-resume": ("agent/tool_agent.py",
        "            lines.extend(_stale_summary_lines(previous_step_summary))\n",
        "            lines.extend(_stale_summary_lines(previous_step_summary))\n"
        "            if self._anim_digest_cache is not None:\n"
        "                lines.extend(self._anim_digest_cache[1])\n"),
    # M5: a ring of one
    "M5-ring-of-one": ("framework/solver.py",
        "    animation_ring: deque = field(default_factory=lambda: deque(maxlen=max(1, ANIM_RING_SIZE)))",
        "    animation_ring: deque = field(default_factory=lambda: deque(maxlen=1))"),
    # M6: the verdict always says NO-OP
    "M6-verdict-noop": ("framework/solver.py",
        "    if _animated_noop(item):\n        return _animated_verdict(item, last=last)\n", ""),
    # M7: the stale latch arms after ANIMATED
    "M7-latch-arms": ("agent/python_tool_sandbox.py",
        "            _hid = _anim_truth and int(", "            _hid = False and int("),
    # M8: last_animation_* cleared by a non-animated action with the ring on
    "M8-record-cleared": ("framework/solver.py",
        "            if animation_summary or not ANIM_RING_ENABLED:\n",
        "            if True:\n"),
    # M10: the digest's per-game character cap is never charged
    "M10-game-cap": ("agent/tool_agent.py",
        "                self._anim_game_chars += len(text)\n", "                pass\n"),
    # M11: off by one, frames[k] = step k
    "M11-index": ("utils/animation.py",
        "        for grid in chain[1:]\n    ]\n", "        for grid in chain[0:]\n    ]\n"),
}

UNIT_TESTS = ["test_animation_layers.py", "test_animation_digest.py", "test_animation_opener.py",
              "test_animation_ring.py", "test_animation_sandbox.py"]


def mutated_root(tmp_path: Path, name: str) -> Path:
    rel, old, new = MUTANTS[name]
    root = tmp_path / f"mut-{name}"
    shutil.copytree(ROOT / "inference", root / "inference")
    path = root / "inference" / rel
    source = path.read_text()
    assert source.count(old) == 1, f"mutant {name} no longer applies"
    path.write_text(source.replace(old, new))
    return root


def failing_unit_tests(root: Path) -> list[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC3_", "LOCAL_ANALYZER_"))}
    env["PYTHONPATH"] = f"{root}{os.pathsep}{HERE}"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:warnings",
                           *[str(HERE / name) for name in UNIT_TESTS]],
                          env=env, cwd=root, capture_output=True, text=True, timeout=600)
    return sorted(set(re.findall(r"^FAILED \S+?::(\S+)", proc.stdout, re.M)))


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutants_are_caught(runs, tmp_path, name):
    root = mutated_root(tmp_path, name)
    caught = [f"unit:{test}" for test in failing_unit_tests(root)]
    mut = {config: run_sim(tmp_path, f"m-{config}", CONFIGS[config], root)
           for config in ("wide-off", "wide-arm", "tight-arm")}
    for config, run in mut.items():
        if check_extends(run):
            caught.append(f"extends:{config}")
        if check_static_system(run):
            caught.append(f"system:{config}")
        if check_digest_placement(run, config in DIGEST_ON):
            caught.append(f"digest:{config}")
        if check_echo(run, config in TRUTH_ON):
            caught.append(f"echo:{config}")
    if check_ring_survives_trim(mut["tight-arm"]):
        caught.append("ring-trim")
    if comparable(mut["wide-off"]) != comparable(runs["wide-off"]):
        caught.append("flags-off")
    print(f"mutant {name}: caught by {caught}")
    assert caught, f"mutant {name} survived every check"
