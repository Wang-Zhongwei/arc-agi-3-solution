"""ARC3_OPENER_DEDUP through the real analyze() loop (tests/opener_dedup_loop_sim.py).

Each sim runs in a subprocess under the Kaggle notebook's own setup_env, read from
kaggle/template.ipynb, so import-time knobs match a real run. Two windows:
- `wide`: the notebook's 128k window. Nothing is trimmed, so every configuration makes the
  same requests and the flags-on messages can be compared with the flags-off ones one by one.
- `tight`: a 25k window, so the history is trimmed every few requests (forced trims, with the
  gate handover they trigger).

Checks: the game plays through; each request's rendered prompt extends the previous one
unless the harness evicted history in between; the system prompt and tools never change;
the dedup removes exactly the static block (when a retained message carries it) and the
duplicate resumption image (when the last retained image is that board), nothing else; one
copy of the opener's static block is in context at the start of every invocation; flags off
is byte-identical to the baseline checkout (requests, rendered prompts and transcript).

Mutation checks apply a known bug to a copy of the harness and require the checks to catch it.

The baseline is `git archive OPENER_DEDUP_BASELINE_REF` (default origin/main) of
ARC3-Inference, or OPENER_DEDUP_BASELINE_ROOT=<path to another checkout's ARC3-Inference>.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
TEMPLATE_NB = REPO / "kaggle" / "template.ipynb"
TIGHT_WINDOW = "25000"
CAPTION_HEAD = "Current grid image: unchanged since the last grid image above"
STATIC_HEAD = "Only tool: `python`."


def notebook_env() -> dict[str, str]:
    """setup_env exactly as the Kaggle template builds it (priority scheduling on)."""
    cells = json.loads(TEMPLATE_NB.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if "setup_env = {" in "".join(c["source"]))
    start = source.index("setup_env = {")
    end = source.index("os.environ.update({k: str(v) for k,v in setup_env.items()})")
    namespace = {"SERVER_BASE_URL": "http://127.0.0.1:9/v1", "SERVED_MODEL_NAME": "flashnext",
                 "USE_PRIORITY_SCHEDULING": True}
    exec(source[start:end], namespace)  # noqa: S102 - our own committed template
    return {k: str(v) for k, v in namespace["setup_env"].items()}


def run_sim(tmp: Path, name: str, extra: dict[str, str], root: Path = ROOT) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC3_", "LOCAL_ANALYZER_"))}
    env.update(notebook_env())
    env.update(extra)
    env["PYTHONPATH"] = str(root)
    out = tmp / f"{name}.json"
    subprocess.run([sys.executable, str(HERE / "opener_dedup_loop_sim.py"), str(out), "--inference-root", str(root)],
                   env=env, check=True, cwd=root, timeout=300, capture_output=True)
    return json.loads(out.read_text())


CONFIGS = {
    "off": {},
    "static": {"ARC3_OPENER_DEDUP": "static"},
    "image": {"ARC3_OPENER_DEDUP": "resume_image"},
    "all": {"ARC3_OPENER_DEDUP": "1"},
}


def run_all(tmp: Path, root: Path = ROOT) -> dict:
    runs = {}
    for name, extra in CONFIGS.items():
        runs[f"wide-{name}"] = run_sim(tmp, f"{tmp.name}-wide-{name}", extra, root)
        runs[f"tight-{name}"] = run_sim(tmp, f"{tmp.name}-tight-{name}",
                                        {**extra, "LOCAL_ANALYZER_CONTEXT_WINDOW": TIGHT_WINDOW}, root)
    return runs


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return run_all(tmp_path_factory.mktemp("dedup"))


# ---------------------------------------------------------------------------- check functions
# Each returns a list of violations, so the mutation tests can require them to be non-empty.

def texts(message) -> list[str]:
    c = message.get("content")
    if isinstance(c, str):
        return [c]
    return [p.get("text") or "" for p in c or [] if isinstance(p, dict) and p.get("type") == "text"]


def image_urls(message) -> list[str]:
    c = message.get("content")
    if not isinstance(c, list):
        return []
    return [p["image_url"]["url"] for p in c if isinstance(p, dict) and p.get("type") == "image_url"]


def invocation_starts(run) -> list[int]:
    out, last = [], None
    for i, r in enumerate(run["requests"]):
        if r["invocation"] != last:
            out.append(i)
            last = r["invocation"]
    return out


def check_extends(run) -> list[str]:
    bad = []
    reqs = run["requests"]
    for a, b in zip(reqs, reqs[1:]):
        if b["evictions"] == a["evictions"] and not b["prompt"].startswith(a["prompt"]):
            bad.append(f"reply {b['reply']}: prompt does not extend reply {a['reply']}")
        if b["evictions"] == a["evictions"] and b["messages"][:len(a["messages"])] != a["messages"]:
            bad.append(f"reply {b['reply']}: an earlier message changed without an eviction")
    return bad


def check_static_system(all_runs) -> list[str]:
    systems = {json.dumps(r["messages"][0]) for run in all_runs for r in run["requests"]}
    tools = {json.dumps(r["tools"]) for run in all_runs for r in run["requests"]}
    return ([f"{len(systems)} different system messages"] if len(systems) != 1 else []) + \
           ([f"{len(tools)} different tool lists"] if len(tools) != 1 else [])


def check_block_in_context(run) -> list[str]:
    """At the first request of every invocation, the opener's static block is in a user message."""
    bad = []
    for i in invocation_starts(run):
        r = run["requests"][i]
        block = r["static_block"]
        if not block:
            bad.append(f"reply {r['reply']}: no static block recorded")
            continue
        if not any(block in t for m in r["messages"][1:] if m["role"] == "user" for t in texts(m)):
            bad.append(f"reply {r['reply']}: static block missing from the context")
    return bad


def check_images(run) -> list[str]:
    """An opener without its board image is a resumption whose last retained image is that board,
    and it says so; every other invocation opens with the current board."""
    bad = []
    for i in invocation_starts(run):
        r = run["requests"][i]
        opener = r["messages"][-1]
        urls = image_urls(opener)
        captioned = any(CAPTION_HEAD in t for t in texts(opener))
        if urls:
            if urls[0] != r["board_image"]:
                bad.append(f"reply {r['reply']}: opener image is not the current board")
            if captioned:
                bad.append(f"reply {r['reply']}: caption on an opener that has its image")
            continue
        if not r["resumed"]:
            bad.append(f"reply {r['reply']}: a fresh turn lost its board image")
        last = next((image_urls(m) for m in reversed(r["messages"][1:-1]) if image_urls(m)), [])
        if r["board_image"] not in last:
            bad.append(f"reply {r['reply']}: image dropped but the last retained image is another board")
        if not captioned:
            bad.append(f"reply {r['reply']}: image dropped without the caption")
    return bad


def expected_opener(off_msg, on_history, modes, block, board, resumed):
    """The flags-off opener with the documented edits, derived independently of the harness."""
    content = off_msg["content"]
    text = content[0]["text"] if isinstance(content, list) else content
    if "static" in modes and block and ("\n" + block) in text and any(
            block in t for m in on_history for t in texts(m)):
        text = text.replace("\n" + block, "", 1)
    if ("resume_image" in modes and resumed and isinstance(content, list) and len(content) == 2
            and image_urls(off_msg) == [board]
            and board in next((image_urls(m) for m in reversed(on_history) if image_urls(m)), [])):
        return {**off_msg, "content": text[: -len("Current grid image:")] + CAPTION_TEXT}
    if isinstance(content, list):
        return {**off_msg, "content": [{**content[0], "text": text}, *content[1:]]}
    return {**off_msg, "content": text}


CAPTION_TEXT = ("Current grid image: unchanged since the last grid image above "
                "(nothing has executed), so it is not repeated.")


def check_exact_dedup(on, off, modes) -> tuple[list[str], dict]:
    """Wide window (no trims): request k of `on` equals request k of `off` with only the openers
    edited as documented."""
    bad, counts = [], {"static": 0, "image": 0}
    if len(on["requests"]) != len(off["requests"]):
        return [f"{len(on['requests'])} vs {len(off['requests'])} requests"], counts
    starts = set(invocation_starts(on))
    for i, (a, b) in enumerate(zip(on["requests"], off["requests"])):
        if len(a["messages"]) != len(b["messages"]):
            bad.append(f"reply {a['reply']}: {len(a['messages'])} vs {len(b['messages'])} messages")
            continue
        for j, (ma, mb) in enumerate(zip(a["messages"], b["messages"])):
            if ma == mb:
                continue
            if not (i in starts and j == len(a["messages"]) - 1) and not any(
                    ma == on["requests"][k]["messages"][-1] for k in starts if k < i):
                bad.append(f"reply {a['reply']} message {j}: differs and is not an opener")
                continue
            if i in starts and j == len(a["messages"]) - 1:
                want = expected_opener(mb, a["messages"][1:-1], modes, a["static_block"],
                                       a["board_image"], a["resumed"])
                if ma != want:
                    bad.append(f"reply {a['reply']}: opener is not the documented edit of the flags-off one")
                counts["static"] += sum(t.count(STATIC_HEAD) for t in texts(mb)) - \
                    sum(t.count(STATIC_HEAD) for t in texts(ma))
                counts["image"] += len(image_urls(mb)) - len(image_urls(ma))
    return bad, counts


def strip_transcript(text: str) -> str:
    return re.sub(r"\| \d\d:\d\d:\d\d \|", "| HH:MM:SS |", text)


def comparable(run) -> list:
    return [{k: r[k] for k in ("messages", "tools", "prompt", "evictions", "reply", "invocation")}
            for r in run["requests"]]


# ---------------------------------------------------------------------------- tests

def test_game_plays_through_with_yields_and_trims(runs):
    for name, run in runs.items():
        assert run["won"] and run["actions"] == 12, name
        inv = run["invocations"]
        resumes = [i for i, x in enumerate(inv) if x["resumed"]]
        assert len(resumes) == 6, name
        assert any(b - a == 1 for a, b in zip(resumes, resumes[1:])), f"{name}: two resumptions in a row"
    for name in CONFIGS:
        assert runs[f"wide-{name}"]["requests"][-1]["evictions"] == 0, "the wide window must not trim"
        tight = runs[f"tight-{name}"]["requests"]
        breaks = sum(1 for a, b in zip(tight, tight[1:]) if b["evictions"] != a["evictions"])
        assert breaks >= 2, f"tight-{name}: trims must be forced ({breaks})"


@pytest.mark.parametrize("name", [f"{w}-{c}" for w in ("wide", "tight") for c in CONFIGS])
def test_each_prompt_extends_the_previous_one(runs, name):
    assert check_extends(runs[name]) == []


def test_system_prompt_and_tools_never_change(runs):
    assert check_static_system(list(runs.values())) == []


@pytest.mark.parametrize("name", ["wide-static", "wide-all", "tight-static", "tight-all", "wide-off", "tight-off"])
def test_static_block_in_context_at_every_invocation(runs, name):
    assert check_block_in_context(runs[name]) == []


@pytest.mark.parametrize("name", [f"{w}-{c}" for w in ("wide", "tight") for c in CONFIGS])
def test_board_images(runs, name):
    assert check_images(runs[name]) == []


@pytest.mark.parametrize("config,modes,static,images", [
    ("static", {"static"}, 12, 0),
    ("image", {"resume_image"}, 0, 6),
    ("all", {"static", "resume_image"}, 12, 6),
])
def test_dedup_removes_exactly_the_intended_text(runs, config, modes, static, images):
    """14 invocations; 2 carry the static block (the first turn's "Ground yourself" variant and
    the first later one); all 6 resumptions follow an opener showing the same board."""
    bad, counts = check_exact_dedup(runs[f"wide-{config}"], runs["wide-off"], modes)
    assert bad == []
    assert counts == {"static": static, "image": images}


def approx_tokens(prompt: str) -> float:
    """Text at 3.5 chars/token plus 400 per image (a 640x640 board image in the real run;
    the template renders each as one short placeholder)."""
    return len(prompt) / 3.5 + 400 * prompt.count("<|image_pad|>")


def test_dedup_shortens_the_context(runs):
    off = approx_tokens(runs["wide-off"]["requests"][-1]["prompt"])
    for config in ("static", "image", "all"):
        assert approx_tokens(runs[f"wide-{config}"]["requests"][-1]["prompt"]) < off
    tight_off = runs["tight-off"]["requests"]
    tight_all = runs["tight-all"]["requests"]
    evictions = lambda reqs: reqs[-1]["evictions"]   # noqa: E731
    assert evictions(tight_all) <= evictions(tight_off)


def test_flags_off_matches_baseline_checkout(runs, tmp_path):
    baseline = baseline_root(tmp_path)
    for window in ("wide", "tight"):
        extra = {} if window == "wide" else {"LOCAL_ANALYZER_CONTEXT_WINDOW": TIGHT_WINDOW}
        base = run_sim(tmp_path, f"baseline-{window}", extra, root=baseline)
        assert comparable(base) == comparable(runs[f"{window}-off"]), window
        assert strip_transcript(base["transcript"]) == strip_transcript(runs[f"{window}-off"]["transcript"])


def baseline_root(tmp_path: Path) -> Path:
    explicit = os.environ.get("OPENER_DEDUP_BASELINE_ROOT", "")
    if explicit:
        return Path(explicit)
    ref = os.environ.get("OPENER_DEDUP_BASELINE_REF", "origin/main")
    out = tmp_path / "baseline"
    out.mkdir()
    try:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", ref, "ARC3-Inference/inference"],
                                 check=True, capture_output=True, timeout=120).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"no baseline checkout ({exc}); set OPENER_DEDUP_BASELINE_ROOT")
    subprocess.run(["tar", "-x", "-C", str(out)], input=archive, check=True)
    return out / "ARC3-Inference"


# ---------------------------------------------------------------------------- mutation checks

MUTANTS = {
    # the carrier test always passes: later openers lose the block with no copy in the window
    "no-carrier-check": ("            static_block in chunk\n", "            True\n"),
    # the edit rewrites earlier history (drops their images) instead of only the new opener
    "edits-history": ("            messages[-1] = opening_message\n",
                      "            messages[-1] = opening_message\n"
                      "            messages[1:-1] = [_strip_message_images(m) for m in messages[1:-1]]\n"),
    # off-by-one: a stray newline is left where the block was
    "leaves-newline": ('        text = user_prompt[: -len("\\n" + static_block)] + text[len(user_prompt):]\n',
                       '        text = user_prompt[: -len(static_block)] + text[len(user_prompt):]\n'),
    # the image is dropped whenever any image is retained, whatever board it shows
    "any-image": ("        drop_image = resumed_image_url in last_images\n",
                  "        drop_image = bool(last_images)\n"),
    # the dedup touches the system prompt
    "system-prompt": ('            {"role": "system", "content": self._system_prompt}, *self._history_messages,\n',
                      '            {"role": "system", "content": self._system_prompt + ("\\n" if _opener_dedup_modes() else "")},'
                      ' *self._history_messages,\n'),
    # the flag defaults on: flags-off is no longer the baseline
    "default-on": ('    raw = os.environ.get("ARC3_OPENER_DEDUP", "").strip().lower()\n',
                   '    raw = os.environ.get("ARC3_OPENER_DEDUP", "1").strip().lower()\n'),
}


def mutated_root(tmp_path: Path, name: str) -> Path:
    old, new = MUTANTS[name]
    root = tmp_path / f"mut-{name}"
    shutil.copytree(ROOT / "inference", root / "inference")
    path = root / "inference" / "agent" / "tool_agent.py"
    source = path.read_text()
    assert source.count(old) == 1, f"mutant {name} no longer applies"
    path.write_text(source.replace(old, new))
    return root


def unit_tests_fail(root: Path) -> bool:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC3_", "LOCAL_ANALYZER_"))}
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                           str(HERE / "test_opener_dedup.py")], env=env, cwd=root, capture_output=True,
                          timeout=300)
    return proc.returncode != 0


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutants_are_caught(runs, tmp_path, name):
    root = mutated_root(tmp_path, name)
    caught = []
    if unit_tests_fail(root):
        caught.append("unit")
    mut = {}
    for config in ("off", "all"):
        for window, extra in (("wide", {}), ("tight", {"LOCAL_ANALYZER_CONTEXT_WINDOW": TIGHT_WINDOW})):
            mut[f"{window}-{config}"] = run_sim(tmp_path, f"m-{window}-{config}",
                                                {**CONFIGS[config], **extra}, root)
    if any(check_extends(r) for r in mut.values()):
        caught.append("extends")
    if check_static_system(list(mut.values()) + [runs["wide-off"]]):
        caught.append("system")
    if any(check_block_in_context(mut[k]) for k in mut):
        caught.append("block")
    if any(check_images(r) for r in mut.values()):
        caught.append("images")
    bad, counts = check_exact_dedup(mut["wide-all"], runs["wide-off"], {"static", "resume_image"})
    if bad or counts != {"static": 12, "image": 6}:
        caught.append("exact")
    if comparable(mut["wide-off"]) != comparable(runs["wide-off"]):
        caught.append("flags-off")
    print(f"mutant {name}: caught by {caught}")
    assert caught, f"mutant {name} survived every check"
