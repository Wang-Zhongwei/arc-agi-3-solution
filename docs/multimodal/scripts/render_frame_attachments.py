"""Render this branch's per-turn user messages, text and images, as one HTML page.

Synthetic 64x64 boards go through the production builders
(ToolAgent._build_user_prompt and _build_user_message) under the harness env of
a Kaggle bundle, so the page shows exactly what the model receives each turn.

    # a bundle from `make kaggle-duck ... KAGGLE_DRY_RUN=true`
    PYTHONPATH=ARC3-Inference:tufa-arc-agi-framework/src python \
        docs/multimodal/scripts/render_frame_attachments.py BUNDLE_DIR OUT.html [--fragment]

--fragment writes the page without the <!doctype>/<head>/<body> wrapper.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

Board = list[list[int]]
BLACK, GRAY, RED, BLUE, YELLOW, ORANGE, GREEN = 5, 2, 8, 9, 11, 12, 14


def bundle_env(bundle: Path) -> dict[str, str]:
    for cmd in json.loads((bundle / "setup_commands.json").read_text()):
        m = re.search(r"ARC3_PENNYROYAL_SPEC='(\{.*\})'", cmd)
        if m:
            return json.loads(m.group(1))["env"]
    raise SystemExit(f"no harness env in {bundle}/setup_commands.json")


# --- synthetic boards -------------------------------------------------------

def board(player: tuple[int, int] | None, budget: int, *, level: int = 1,
          hazard: bool = True, door: int = ORANGE, player_color: int = BLUE) -> Board:
    """A walled room: player block, goal, hazard, a door and a budget bar."""
    g = [[BLACK] * 64 for _ in range(64)]
    for i in range(64):
        g[0][i] = g[57][i] = g[i][0] = g[i][63] = GRAY
    if level == 1:
        for r in range(8, 50):
            g[r][30] = GRAY
        for r in range(26, 34):
            g[r][30] = door
        goal = (24, 48)
    else:
        for c in range(10, 54):
            g[30][c] = GRAY
        goal = (8, 52)
    for r in range(goal[0], goal[0] + 6):
        for c in range(goal[1], goal[1] + 6):
            g[r][c] = GREEN
    if hazard and level == 1:
        for r in range(40, 44):
            for c in range(20, 24):
                g[r][c] = RED
    for c in range(2, 2 + budget):
        for r in (60, 61):
            g[r][c] = YELLOW
    if player is not None:
        for r in range(player[0], player[0] + 4):
            for c in range(player[1], player[1] + 4):
                g[r][c] = player_color
    return g


def frame(grid: Board, step: int, level: int = 1):
    from inference.agent.runtime_state import Frame
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


# --- scenarios ----------------------------------------------------------------

def new_agent(tmp: Path):
    from inference.agent import tool_agent
    agent = tool_agent.ToolAgent(model="flashnext", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp / "state.json")
    agent._kept_functions = {"player_pos": "def player_pos(segmentation):\n    return None\n"}
    return agent


def render_turn(agent, entries, summary, chain, *, changed=True):
    from inference.agent.runtime_state import HistoryEntry  # noqa: F401  (entries are built by callers)
    agent._auto_diff_prev_frame = entries[0].frame
    agent._last_action_gameplay_changed = changed
    agent._last_animation_chain = [tuple(tuple(row) for row in g) for g in chain]
    current = entries[-1].frame
    prompt = agent._build_user_prompt(
        current.step, valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "SPACE"],
        current_frame=current, history_entries=entries, previous_step_summary=summary,
    )
    message = agent._build_user_message(prompt, current)["content"]
    return message if isinstance(message, list) else [{"type": "text", "text": message}]


def scenarios(tmp: Path) -> list[dict]:
    from inference.agent.runtime_state import HistoryEntry
    from inference.utils.animation import summarize_animation

    out = []

    def step_boards(start, moves, budget):
        """Boards after each move of 4 cells; returns (positions, boards)."""
        pos, boards = start, []
        for move in moves:
            dr, dc = {"RIGHT": (0, 4), "LEFT": (0, -4), "UP": (-4, 0), "DOWN": (4, 0)}.get(move, (0, 0))
            pos = (pos[0] + dr, pos[1] + dc)
            budget -= 2
            boards.append((pos, budget))
        return boards

    # Ordinary turn: three moves, then UP slides two rows per frame.
    start, budget0 = (36, 6), 40
    path = step_boards(start, ["RIGHT", "RIGHT", "RIGHT", "UP"], budget0)
    entries = [HistoryEntry("INIT", frame(board(start, budget0), 100))]
    for i, (move, (pos, budget)) in enumerate(zip(["RIGHT", "RIGHT", "RIGHT", "UP"], path), start=1):
        entries.append(HistoryEntry(move, frame(board(pos, budget), 100 + i)))
    before_up = board(path[2][0], path[2][1])
    slide = [board((path[2][0][0] - k, path[2][0][1]), path[3][1]) for k in (1, 2, 3)]
    chain = [before_up, *slide, board(path[3][0], path[3][1])]
    summary = {"executed_count": 4, "executed_actions": ["RIGHT", "RIGHT", "RIGHT", "UP"],
               "animation": summarize_animation(chain), "animation_action": "UP", "gameplay_changed": True}
    out.append(dict(id="ordinary", title="Ordinary turn",
                    note="RIGHT ×3, then UP. UP animates: the block slides one row per frame over three frames.",
                    entries=entries, summary=summary, chain=chain, changed=True))

    # Animation that leaves the board interior unchanged: SPACE flashes the door.
    pos, budget = (36, 18), 30
    entries = [HistoryEntry("INIT", frame(board(pos, budget), 200)),
               HistoryEntry("SPACE", frame(board(pos, budget - 2), 201))]
    chain = [board(pos, budget), board(pos, budget - 2, door=YELLOW),
             board(pos, budget - 2, door=RED), board(pos, budget - 2)]
    summary = {"executed_count": 1, "executed_actions": ["SPACE"],
               "animation": summarize_animation(chain), "animation_action": "SPACE", "gameplay_changed": False}
    out.append(dict(id="unchanged", title="Animation with an unchanged interior",
                    note="SPACE makes the door flash yellow, then red, then return. Only the budget bar is shorter afterwards.",
                    entries=entries, summary=summary, chain=chain, changed=False))

    # Level-up: the last RIGHT reaches the goal; the goal flashes before level 2 starts.
    pos, budget = (24, 40), 24
    entries = [HistoryEntry("INIT", frame(board(pos, budget), 300)),
               HistoryEntry("RIGHT", frame(board((24, 44), budget - 2), 301)),
               HistoryEntry("RIGHT", frame(board((40, 8), 40, level=2), 302, level=2))]
    goal_hit = board((24, 48), budget - 4)
    chain = [board((24, 44), budget - 2), goal_hit, board((24, 48), budget - 4, player_color=YELLOW),
             board((40, 8), 40, level=2)]
    summary = {"executed_count": 2, "executed_actions": ["RIGHT", "RIGHT"], "level_transition": True,
               "animation": summarize_animation(chain), "animation_action": "RIGHT", "gameplay_changed": True}
    out.append(dict(id="level", title="Level-up",
                    note="Two RIGHT moves; the second reaches the green goal. The goal flashes and level 2 starts.",
                    entries=entries, summary=summary, chain=chain, changed=True))

    # Death: the third DOWN steps onto the red hazard; the block flashes red, then RESET.
    start, budget = (28, 20), 30
    entries = [HistoryEntry("INIT", frame(board(start, budget), 400)),
               HistoryEntry("DOWN", frame(board((32, 20), budget - 2), 401)),
               HistoryEntry("DOWN", frame(board((36, 20), budget - 4), 402)),
               HistoryEntry("DOWN", frame(board((40, 20), budget - 6, player_color=RED), 403)),
               HistoryEntry("RESET", frame(board((36, 6), 40), 404))]
    chain = [board((36, 20), budget - 4), board((40, 20), budget - 6, player_color=ORANGE),
             board((40, 20), budget - 6, player_color=YELLOW), board((40, 20), budget - 6, player_color=RED)]
    summary = {"executed_count": 3, "executed_actions": ["DOWN", "DOWN", "DOWN"], "game_over": True,
               "fatal_action": "DOWN", "animation": summarize_animation(chain), "animation_action": "DOWN",
               "game_over_trace": ["DOWN -> moved", "DOWN -> moved",
                                   "DOWN -> GAME OVER (attempt ended immediately after this action)"]}
    out.append(dict(id="death", title="Death and automatic reset",
                    note="DOWN ×3; the third steps onto the red hazard. The block flashes orange, then yellow, then red. The level resets.",
                    entries=entries, summary=summary, chain=chain, changed=True))

    # No-op guard: the third LEFT hits the wall and the guard skips the rest.
    pos, budget = (36, 38), 34
    entries = [HistoryEntry("INIT", frame(board(pos, budget), 500)),
               HistoryEntry("LEFT", frame(board((36, 34), budget - 2), 501)),
               HistoryEntry("LEFT", frame(board((36, 31), budget - 4), 502)),
               HistoryEntry("LEFT", frame(board((36, 31), budget - 6), 503))]
    summary = {"executed_count": 3, "executed_actions": ["LEFT", "LEFT", "LEFT"],
               "no_op_stops": [{"index": 3, "action": "LEFT", "skipped": ["LEFT", "UP"]}],
               "gameplay_changed": True}
    out.append(dict(id="noop", title="No-op guard stop",
                    note="LEFT ×3, LEFT, UP were submitted. The third LEFT hits the wall, so the guard skips LEFT and UP. No animation.",
                    entries=entries, summary=summary, chain=[], changed=True))
    return out


# --- HTML ---------------------------------------------------------------------

CSS = """
/* Layout: one centered column of request transcripts; each turn is an ordered list of content parts. */
:root {
  --bg: #f3f5f9; --surface: #ffffff; --text-part: #eef1f7; --fg: #172033; --muted: #5b667a;
  --rule: #d6dce7; --accent: #2a4a8c; --accent-soft: #dfe6f4;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d1220; --surface: #141b2b; --text-part: #111827; --fg: #e2e7f1; --muted: #98a3b8;
  --rule: #26314a; --accent: #9db7f5; --accent-soft: #1d2a47; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0d1220; --surface: #141b2b; --text-part: #111827; --fg: #e2e7f1; --muted: #98a3b8;
  --rule: #26314a; --accent: #9db7f5; --accent-soft: #1d2a47; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 16px/1.55 var(--body); margin: 0; }
.page { max-width: 980px; margin: 0 auto; padding-inline: 20px; padding-block: 32px 64px; }
h1, h2 { font-family: var(--display); text-wrap: balance; line-height: 1.15; }
h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 12px; letter-spacing: -0.01em; }
h2 { font-size: 1.45rem; font-weight: 650; margin: 0; }
p { max-width: 68ch; margin: 0; }
.lede { color: var(--muted); }
.intro { display: grid; gap: 14px; }
.settings { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0;
  font: 13.5px/1.5 var(--mono); background: var(--surface); border: 1px solid var(--rule);
  border-radius: 6px; padding: 14px 16px; overflow-x: auto; }
.settings dt { color: var(--muted); } .settings dd { margin: 0; min-width: 0; }
nav { display: flex; flex-wrap: wrap; gap: 6px 18px; padding-block: 18px; border-bottom: 1px solid var(--rule); }
nav a { color: var(--accent); text-decoration: none; font-weight: 600; }
nav a:hover, nav a:focus-visible { text-decoration: underline; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
section { display: grid; gap: 14px; padding-block: 34px 8px; border-bottom: 1px solid var(--rule); }
.tag { font: 600 12px/1 var(--body); text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); }
.note { color: var(--muted); }
ol.parts { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.part { display: grid; gap: 6px; min-width: 0; }
.idx { font: 12px/1 var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 13.5px/1.55 var(--mono);
  background: var(--text-part); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; min-width: 0; }
figure { margin: 0; display: grid; gap: 6px; justify-items: start; background: var(--surface);
  border: 1px solid var(--rule); border-left: 3px solid var(--accent); border-radius: 6px; padding: 10px 12px; }
figcaption { font: 13.5px/1.5 var(--mono); overflow-wrap: anywhere; }
figure img { image-rendering: pixelated; height: auto; max-width: 100%; border: 1px solid var(--rule); }
.dims { font: 12px/1 var(--mono); color: var(--muted); }
.strip { display: flex; flex-wrap: wrap; gap: 10px; }
footer { color: var(--muted); font-size: 14px; padding-block: 24px 0; }
"""
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700'
         '&family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+3:wght@400;600&display=swap">')


def png_size(url: str) -> tuple[int, int]:
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size


def render_parts(parts: list[dict]) -> str:
    """Text parts as prose; a short text part followed by an image becomes its caption."""
    items, i, n = [], 0, 1
    while i < len(parts):
        part = parts[i]
        nxt = parts[i + 1] if i + 1 < len(parts) else None
        if part["type"] == "text" and nxt and nxt["type"] == "image_url" and len(part["text"]) < 240:
            url = nxt["image_url"]["url"]
            w, h = png_size(url)
            shown = 300 if w >= 512 else w
            items.append(
                f'<li class="part"><span class="idx">parts {n}–{n + 1} · caption + image</span>'
                f'<figure><figcaption>{html.escape(part["text"].strip())}</figcaption>'
                f'<img src="{url}" width="{shown}" height="{shown * h // w}" alt="{html.escape(part["text"].strip())}">'
                f'<span class="dims">{w}×{h} PNG</span></figure></li>')
            i, n = i + 2, n + 2
        elif part["type"] == "text":
            items.append(f'<li class="part"><span class="idx">part {n} · text</span>'
                         f'<pre>{html.escape(part["text"].strip())}</pre></li>')
            i, n = i + 1, n + 1
        else:
            url = part["image_url"]["url"]
            w, h = png_size(url)
            shown = 300 if w >= 512 else w
            items.append(f'<li class="part"><span class="idx">part {n} · image</span><figure>'
                         f'<img src="{url}" width="{shown}" height="{shown * h // w}" alt="image part {n}">'
                         f'<span class="dims">{w}×{h} PNG</span></figure></li>')
            i, n = i + 1, n + 1
    return '<ol class="parts">' + "".join(items) + "</ol>"


def page(body: str, *, fragment: bool) -> str:
    head = f"<title>One-wave frame attachments</title>\n{FONTS}\n<style>{CSS}</style>\n"
    if fragment:
        return head + body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fragment", action="store_true")
    args = ap.parse_args()

    env = bundle_env(args.bundle)
    os.environ.update(env)  # before import: tool_agent reads some settings at import time
    from inference.agent import tool_agent  # noqa: F401

    shown_keys = ["ARC3_ANIMATION", "ARC3_ANIMATION_IMAGE", "ARC3_ANIMATION_DEATH_IMAGE",
                  "MULTIMODAL_INTERMEDIATE_FRAMES", "MULTIMODAL_CONTEXT", "MULTIMODAL_UPSCALE",
                  "MULTIMODAL_ANIMATION_UPSCALE", "ARC3_DIFF_IMAGE", "ARC3_GAMEOVER_DIFF_IMAGE",
                  "ARC3_MEMORY_SECTIONS", "LOCAL_ANALYZER_YIELD_TOKENS"]
    settings = "".join(f"<dt>{k}</dt><dd>{html.escape(str(env.get(k, '(unset)')))}</dd>" for k in shown_keys)

    sections = []
    with tempfile.TemporaryDirectory() as tmp:
        cases = scenarios(Path(tmp))
        # The same ordinary turn as currently run (animation frames off), for comparison.
        baseline = dict(cases[0], id="baseline", title="Ordinary turn with animation frames off",
                        note="The first turn again with ARC3_ANIMATION_IMAGE and ARC3_ANIMATION_DEATH_IMAGE off, "
                             "as in the runs so far: the prompt text comes first, then the current board and diff image.")
        for case in cases + [baseline]:
            saved = {k: os.environ.get(k) for k in ("ARC3_ANIMATION_IMAGE", "ARC3_ANIMATION_DEATH_IMAGE")}
            if case["id"] == "baseline":
                os.environ["ARC3_ANIMATION_IMAGE"] = os.environ["ARC3_ANIMATION_DEATH_IMAGE"] = "off"
            try:
                parts = render_turn(new_agent(Path(tmp) / case["id"]), case["entries"], case["summary"],
                                    case["chain"], changed=case["changed"])
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            images = sum(p["type"] == "image_url" for p in parts)
            sections.append(
                f'<section id="{case["id"]}"><span class="tag">{len(parts)} content parts · {images} images</span>'
                f'<h2>{html.escape(case["title"])}</h2><p class="note">{html.escape(case["note"])}</p>'
                f"{render_parts(parts)}</section>")
            case["parts"] = parts

    nav = "".join(f'<a href="#{c["id"]}">{html.escape(c["title"])}</a>' for c in cases + [baseline])
    body = (
        '<main class="page"><div class="intro">'
        "<h1>One-wave frame attachments</h1>"
        '<p class="lede">What the model receives each turn on branch <code>allocation-vc-one-wave</code> '
        "with animation frames in context, rendered by the harness's own prompt and image builders. "
        "Boards are synthetic 64×64 grids. Images are the request PNGs before model preprocessing.</p>"
        f'<dl class="settings">{settings}</dl></div>'
        f"<nav>{nav}</nav>{''.join(sections)}"
        '<footer>Generated by docs/multimodal/scripts/render_frame_attachments.py from the '
        "thinking-budget-priority-one-wave v3 dry-run bundle.</footer></main>"
    )
    args.out.write_text(page(body, fragment=args.fragment))
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
