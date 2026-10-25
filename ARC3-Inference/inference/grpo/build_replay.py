"""Replay games for the rollout benchmark: real level 1-2 play as GRPO would send it, untrimmed.

Each game is one transcript rebuilt with inference/rft/context_rebuild.py. Request k sends the system
prompt + every stream message before its reply (no trimming, as in a GRPO rollout), and asks for as many
tokens as the recorded reply had (estimate within ~3%). A game ends at its first level-3 request or
before the context passes CAP, the GRPO budget's cap. Images are drawn again from the run's frames
(inference/rft/rft_images.py); one that cannot be drawn becomes a blank board of the same size, so the
token count stays right.

One JSON line per game: {id, run, game, tools, system, stream, images, requests: [[stream_end, max_tokens,
est_context], ...]}. Image parts carry {"url": "img:<sha1>"}; `images` maps sha1 -> base64 PNG.

    PYTHONPATH=. .venv/bin/python -m inference.grpo.build_replay OUT.jsonl RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from inference.rft import rft_images
from inference.rft.context_rebuild import TOKENS_FIXED, estimate_tokens, rebuild_file, run_tokens_per_image
from inference.rft.rft_data import GameFrames, level_after_actions, read_events, run_tools

CAP = 130_000  # GRPO budget cap, context tokens (docs/RL/decisions.md)
FALLBACK_TOOLS = Path(__file__).resolve().parent.parent / "rft" / "data" / "python_tool.json"
BLANK = 640  # a 64x64 board at MULTIMODAL_UPSCALE 10


def png(image: Image.Image) -> tuple[str, str]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return hashlib.sha1(data).hexdigest(), base64.b64encode(data).decode()


def levels_by_step(events: list[dict]) -> dict[int, int]:
    return {e["analysis_step"]: e["level"] for e in events if e.get("type") == "analysis"}


def one_run(run: str) -> tuple[list[dict], dict]:
    run_dir = Path(run)
    env = json.loads((run_dir / "taaf_setup_env.json").read_text())
    scales = rft_images.Scales.from_env(env)
    frames_by_game = GameFrames(run_dir)
    try:
        tools = run_tools(run_dir)
    except (FileNotFoundError, ValueError):  # a run saved without its source snapshot
        tools = json.loads(FALLBACK_TOOLS.read_text())
    blank_sha, blank_b64 = png(Image.new("RGB", (BLANK, BLANK), (255, 255, 255)))
    per_image = run_tokens_per_image(run_dir)  # vision tokens of one board image at this run's upscale
    games, stats = [], {"blank_images": 0, "images": 0}
    for path in sorted((run_dir / "transcripts").glob("*_p0.txt")):
        events_path = run_dir / "artifacts" / f"{path.stem}_events.jsonl"
        states = frames_by_game.get(path.stem)
        if not events_path.exists() or states is None:
            continue
        events = read_events(events_path)
        _, _, completing, _, _ = level_after_actions(events)
        frames = rft_images.Frames.from_run(states, events)
        at_step = levels_by_step(events)
        game = rebuild_file(path)
        system = {"role": "system", "content": game.system}
        requests = []
        for request in sorted(game.requests, key=lambda r: r.stream_end):
            if (at_step.get(request.step) or 1) >= 3:
                break
            context = estimate_tokens([system] + game.stream[:request.stream_end], per_image)
            if context > CAP:
                break
            generated = max(1, round(estimate_tokens([request.reply]) - TOKENS_FIXED))
            requests.append([request.stream_end, generated, round(context)])
        if not requests:
            continue
        end = requests[-1][0]
        images: dict[str, str] = {}
        stream = []
        for message in game.stream[:end]:
            clean = {k: v for k, v in message.items() if not k.startswith("_")}
            if message.get("role") == "user" and isinstance(message.get("content"), list) and "_action" in message:
                drawn = iter(rft_images.message_images(message, message["_action"] - 1, frames, scales, completing))
                parts = []
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        image = next(drawn, None)
                        stats["images"] += 1
                        if image is None:
                            stats["blank_images"] += 1
                            sha, b64 = blank_sha, blank_b64
                        else:
                            sha, b64 = png(image)
                        images[sha] = b64
                        part = {"type": "image_url", "image_url": {"url": f"img:{sha}"}}
                    parts.append(part)
                clean["content"] = parts
            stream.append(clean)
        games.append({"id": f"{run_dir.name}|{run_dir.parent.name}|{path.stem}", "run": run, "game": path.stem[:4],
                      "tools": tools, "system": game.system, "stream": stream, "images": images,
                      "requests": requests})
    return games, stats


def main(argv: list[str]) -> int:
    out, runs = Path(argv[0]), argv[1:]
    blank = total = 0
    count = 0
    with ProcessPoolExecutor(max_workers=min(6, len(runs))) as pool, open(out, "w") as f:
        for games, stats in pool.map(one_run, runs):
            for g in games:
                f.write(json.dumps(g) + "\n")
            count += len(games)
            blank, total = blank + stats["blank_images"], total + stats["images"]
    print(f"{count} games from {len(runs)} runs -> {out}; images {total}, blank {blank}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
