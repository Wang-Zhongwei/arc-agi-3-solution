"""Check rft_images against runs that logged their requests (`*_requests.jsonl` keep the PNGs).

Every rebuilt user message with images is matched to the logged user message with the same
text, and each image is drawn from the run's frames and compared pixel by pixel. Counts are
reported per label kind (digits replaced by N).

    python -m inference.rft.validate_images runs/kaggle/<owner>/<notebook>/vNN [...] --out summary.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from inference.rft import rft_images
from inference.rft.context_rebuild import rebuild_file
from inference.rft.validate_rebuild import user_text


def kind(label: str) -> str:
    label = re.sub(r"\(after [^)]*\)", "(after X)", label)
    return re.sub(r"\d+", "N", label)[:70]


def logged_images(path: Path) -> dict[str, list[np.ndarray]]:
    """Normalized user text -> its images, from a request log."""
    out: dict[str, list[np.ndarray]] = {}
    for line in open(path):
        request = json.loads(line)
        for m in request.get("messages") or []:
            content = m.get("content")
            if m.get("role") != "user" or not isinstance(content, list):
                continue
            key = user_text(m)
            if key in out:
                continue
            images = []
            for part in content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
                    images.append(np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")))
            if images:
                out[key] = images
    return out


def load_frames(run: Path) -> dict[str, list]:
    """game id -> list of per-pass states (intermediate_states.pkl holds one state list per game run)."""
    states = pickle.load(open(run / "intermediate_states.pkl", "rb"))
    games: dict[str, list] = defaultdict(list)
    for game in states:
        if game:
            games[game[0].raw.game_id].append(game)
    return games


def completed_actions(events: Path) -> list[int]:
    out = []
    for line in open(events):
        e = json.loads(line)
        if e.get("type") == "action" and e.get("level_completed"):
            out.append(e["action_num"])
    return out


def check_run(run: Path, counts: dict[str, Counter], examples: dict[str, list]) -> None:
    env_path = run / "taaf_setup_env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    games = load_frames(run)
    for log in sorted(list(run.glob("*_requests.jsonl")) + list(run.glob("artifacts/*_requests.jsonl"))):
        stem = log.name[: -len("_requests.jsonl")]
        transcript = run / "transcripts" / f"{stem}.txt"
        events = run / "artifacts" / f"{stem}_events.jsonl"
        game_id, pass_index = re.match(r"(.+)_p(\d+)$", stem).groups()
        if not transcript.exists() or not events.exists() or game_id not in games:
            continue
        runs = games[game_id]
        frames = rft_images.Frames.from_run(runs[min(int(pass_index), len(runs) - 1)],
                                            (json.loads(line) for line in open(events)))
        logged = logged_images(log)
        scales = rft_images.Scales.from_env(env)
        if not env and logged:  # no env file: take the board scale from a logged image
            first = next(iter(logged.values()))[0]
            scales = rft_images.Scales(first.shape[0] // 64, first.shape[0] // 64, first.shape[0] // 64)
        completed = completed_actions(events)
        game = rebuild_file(transcript)
        messages = [m for m in game.stream if m.get("role") == "user"]
        messages += [r.messages[1] for r in game.requests[:1] if len(r.messages) > 1 and "_action" in r.messages[1]
                     and r.messages[1] not in messages]
        for m in messages:
            if not isinstance(m.get("content"), list) or "_action" not in m:
                continue
            labels, previous = [], ""
            for part in m["content"]:
                if part.get("type") == "image_url":
                    labels.append(rft_images.label_before(previous))
                previous = part.get("text") or "" if part.get("type") != "image_url" else ""
            if not labels:
                continue
            drawn = rft_images.message_images(m, m["_action"] - 1, frames, scales, completed)
            reference = logged.get(user_text(m))
            for i, (label, image) in enumerate(zip(labels, drawn)):
                k = kind(label)
                if reference is None:
                    counts[k]["not logged"] += 1
                elif i >= len(reference):
                    counts[k]["extra image"] += 1
                elif image is None:
                    counts[k]["unresolved"] += 1
                else:
                    ok = np.asarray(image).shape == reference[i].shape and np.array_equal(np.asarray(image), reference[i])
                    counts[k]["match" if ok else "mismatch"] += 1
                    if not ok and len(examples[k]) < 3:
                        examples[k].append(dict(run=str(run), game=stem, action=m["_action"], label=label,
                                                drawn=list(np.asarray(image).shape), logged=list(reference[i].shape)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    counts: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    for run in args.runs:
        check_run(run, counts, examples)
        print("checked", run, flush=True)
    summary = {k: dict(v) for k, v in sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))}
    for k, v in summary.items():
        print(f"{sum(v.values()):7d}  {k!r}: {v}")
    if args.out:
        args.out.write_text(json.dumps(dict(counts=summary, mismatches=examples), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
