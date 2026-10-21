"""Phone-friendly reader for saved ARC-AGI-3 runs: per-game model reasoning, step by step.

Reads Kaggle run outputs straight from runs/kaggle/<kernel>/v<N>/ (plain or .xz)
plus any extra run folders, and serves them as small mobile pages. Kaggle runs carry no
*viewer_data.json, so the repo's `make view` cannot open them. Every run version is listed;
per-game model calls come from whichever of these the run saved:

- transcripts/<game>.txt: Franzen-harness transcripts, every model call with its thinking,
  tool calls and tool results.
- prompts/<game>.log: Fable F001 prompt logs, which keep only the latest model call per game
  (its full input, reasoning and tool calls), shown as one step.
- tycho-results/game_<game>.json: Tycho per-game results; each committed action's model
  calls (prompt, reasoning, response).

Runs with none of these (failed or log-only versions) show only their summary and logs.

    python tools/phone_viewer.py --host <tailscale-ip> --port 8012 [--extra DIR ...]

Bind to the Tailscale IP (not 0.0.0.0) so only the tailnet can reach it.
"""
from __future__ import annotations

import argparse
import html
import json
import lzma
import os
import re
import urllib.parse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# In the repo (tools/) the runs sit next door; in the project folder they are under the repo.
DEFAULT_ROOT = next((p for p in (HERE.parent / "runs" / "kaggle",
                                 HERE.parent / "daniel-franzen-sol" / "runs" / "kaggle") if p.is_dir()),
                    HERE.parent / "runs" / "kaggle")
STEPS_PER_PAGE = 15
STEP_RE = re.compile(r"^--- (analysis_step=.*?) ---$", re.M)
SECTION_RE = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|ANALYZER STATUS|MODEL RESPONSE META|THINKING|"
                        r"ASSISTANT|ASSISTANT TEXT|TOOL CALL: [^\]]+|TOOL RESULT: [^\]]+)\]$", re.M)
# F001's prompt logs add their own markers; kept apart so Franzen transcripts parse as before.
SNAPSHOT_SECTION_RE = re.compile(
    r"^\[(SYSTEM PROMPT|USER PROMPT|ANALYZER STATUS|AVAILABLE TOOLS|MODEL INPUT|TURN TRANSCRIPT SO FAR|"
    r"SYSTEM|USER|REASONING|ASSISTANT|ASSISTANT TOOL CALL: [^\]]+|TOOL RESULT: [^\]]+)\]$", re.M)
# Section -> (css class, open by default)
STYLE = {
    "THINKING": ("think", True),
    "REASONING": ("think", True),
    "ASSISTANT": ("say", True),
    "ASSISTANT TEXT": ("say", True),
    "USER PROMPT": ("user", False),
    "USER": ("user", False),
    "SYSTEM PROMPT": ("sys", False),
    "SYSTEM": ("sys", False),
    "ANALYZER STATUS": ("meta", False),
    "MODEL RESPONSE META": ("meta", False),
}
# Folders inside a run that hold per-game model calls, in the order they are looked for.
GAME_DIRS = ("transcripts", "prompts", "tycho-results")
CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--think:#eef4ff;--tool:#f4f1ea;--say:#eaf7ee;--bd:#d9d9de}
@media (prefers-color-scheme:dark){:root{--bg:#111113;--fg:#ececf1;--mut:#9a9aa3;--card:#1c1c20;--think:#18233a;--tool:#28241b;--say:#15281c;--bd:#33333a}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif;margin:0 auto;max-width:900px;padding:12px 16px 60px}
a{color:#2f6fde}h1{font-size:19px;margin:8px 0}h2{font-size:16px;margin:16px 0 6px}
.mut{color:var(--mut);font-size:13px}ul{padding-left:18px}li{margin:4px 0}
details{background:var(--card);border:1px solid var(--bd);border-radius:10px;margin:8px 0;padding:6px 10px}
details>summary{cursor:pointer;font-weight:600}
details.sec{margin:6px 0;padding:4px 8px;border-radius:8px}
.think{background:var(--think)}.tool{background:var(--tool)}.say{background:var(--say)}
pre{white-space:pre-wrap;word-break:break-word;font:12.5px/1.4 ui-monospace,Menlo,monospace;margin:6px 0}
.nav{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}
"""


def _read(path: Path) -> str:
    """Text of a plain or .xz file. A truncated .xz raises EOFError, a corrupt one LZMAError."""
    if path.suffix == ".xz":
        with lzma.open(path) as f:
            data = f.read()
    else:
        data = path.read_bytes()
    return data.decode("utf-8", errors="replace")


def _game_name(path: Path) -> str:
    name = path.name.removesuffix(".xz")
    for suffix in (".txt", ".log", ".json"):
        name = name.removesuffix(suffix)
    return name.removeprefix("game_") if path.parent.name == "tycho-results" else name


class Runs:
    def __init__(self, roots: list[Path]):
        self.roots = roots

    def list(self) -> dict[str, Path]:
        """run label -> run folder (newest first): every <kernel>/v<N> version under a root,
        plus any folder holding transcripts/, prompts/ or tycho-results/."""
        found = {}
        for root in self.roots:
            runs = {d for d in root.glob("*/v*") if d.is_dir()}
            for dirpath, dirnames, _ in os.walk(root):
                if any(name in dirnames for name in GAME_DIRS):
                    runs.add(Path(dirpath))
                    dirnames.clear()  # a run's own subfolders hold no further runs
            for run in runs:
                found[str(run.relative_to(root)) if run != root else run.name] = run
        return dict(sorted(found.items(), key=lambda kv: kv[1].stat().st_mtime, reverse=True))

    @staticmethod
    def games(run: Path) -> dict[str, Path]:
        """game -> file of its model calls, from the first of GAME_DIRS the run has."""
        for folder in GAME_DIRS:
            if (run / folder).is_dir():
                pattern = "game_*.json*" if folder == "tycho-results" else "*"
                out = {}
                for p in sorted((run / folder).glob(pattern)):
                    if p.is_file():
                        out.setdefault(_game_name(p), p)
                return out
        return {}


def _tycho_steps(text: str) -> list[tuple[str, str]]:
    """Each model call in a Tycho game result, under the action its turn committed."""
    steps = []
    for turn in json.loads(text).get("trace", []):
        reasoning = turn.get("reasoning") or {}
        calls = reasoning.get("llm_calls") or []
        action = turn.get("action", "?")
        if turn.get("x") is not None or turn.get("y") is not None:
            action += f"(x={turn.get('x')},y={turn.get('y')})"
        for i, call in enumerate(calls, 1):
            header = (f"turn={turn.get('turn')} | level={reasoning.get('level')} | call {i}/{len(calls)} | "
                      f"then {action} | in={call.get('tokens_in')} out={call.get('tokens_out')} "
                      f"cached={call.get('cache_read')}")
            body = (f"[USER PROMPT]\n{call.get('prompt') or ''}\n"
                    f"[THINKING]\n{call.get('reasoning_text') or call.get('reasoning_summary') or ''}\n"
                    f"[ASSISTANT]\n{(call.get('response') or '').strip()}\n")
            steps.append((header, body))
    return steps


@lru_cache(maxsize=16)
def _steps(path: str, mtime: float) -> list[tuple[str, str]]:
    folder = Path(path).parent.name
    text = _read(Path(path))
    if folder == "tycho-results":
        return _tycho_steps(text)
    if folder == "prompts":
        # One snapshot of the latest call; its step header line sits at the end.
        found = STEP_RE.findall(text)
        return [(f"latest model call only: {found[-1] if found else 'no step header'}", text)]
    parts = STEP_RE.split(text)
    # parts = [preamble, header1, body1, header2, body2, ...]
    return [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def _sections(body: str, section_re: re.Pattern = SECTION_RE) -> str:
    parts = section_re.split(body)
    out = []
    if parts[0].strip():
        out.append(f"<pre>{html.escape(parts[0].strip())}</pre>")
    for i in range(1, len(parts) - 1, 2):
        name, content = parts[i], parts[i + 1].strip()
        if name.startswith(("TOOL CALL", "TOOL RESULT", "ASSISTANT TOOL CALL")):
            cls, is_open = "tool", name.startswith(("TOOL CALL", "ASSISTANT TOOL CALL"))
        else:
            cls, is_open = STYLE.get(name, ("meta", False))
        size = f" <span class=mut>({len(content):,} chars)</span>" if not is_open else ""
        out.append(f"<details class='sec {cls}'{' open' if is_open else ''}><summary>{html.escape(name)}{size}"
                   f"</summary><pre>{html.escape(content)}</pre></details>")
    return "".join(out)


def _page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
            f"initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}"
            f"</body></html>").encode()


def make_handler(runs: Runs):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, payload: bytes, ctype: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            try:
                if url.path == "/":
                    return self._send(200, self.index())
                if url.path == "/run":
                    return self._send(200, self.run(q["r"][0]))
                if url.path == "/game":
                    return self._send(200, self.game(q["r"][0], q["g"][0], int(q.get("p", ["0"])[0])))
                if url.path == "/file":
                    return self._send(200, self.file(q["r"][0], q["f"][0]).encode(), "text/plain; charset=utf-8")
            except (KeyError, ValueError, OSError) as exc:
                return self._send(404, _page("Not found", f"<p>{html.escape(str(exc))}</p><a href='/'>Runs</a>"))
            except (EOFError, lzma.LZMAError) as exc:
                # A partly downloaded or corrupt .xz: say so instead of dropping the connection.
                return self._send(500, _page("Unreadable file", f"<p>The file is truncated or corrupt: "
                                                                f"{html.escape(repr(exc))}</p><a href='/'>Runs</a>"))
            except Exception as exc:  # anything else still gets a page, not an empty reply
                return self._send(500, _page("Error", f"<p>{html.escape(repr(exc))}</p><a href='/'>Runs</a>"))
            self._send(404, _page("Not found", "<a href='/'>Runs</a>"))

        def index(self) -> bytes:
            items = "".join(
                f"<li><a href='/run?r={urllib.parse.quote(label)}'>{html.escape(label)}</a></li>"
                for label in runs.list()
            )
            return _page("Runs", f"<h1>Runs</h1><p class=mut>Newest first.</p><ul>{items}</ul>")

        def _run_dir(self, label: str) -> Path:
            run = runs.list().get(label)
            if run is None:
                raise KeyError(f"unknown run {label}")
            return run

        def run(self, label: str) -> bytes:
            run = self._run_dir(label)
            summary = next((p for p in (run / "summary.txt", run / "summary.txt.xz") if p.exists()), None)
            head = f"<pre>{html.escape(_read(summary)[:6000])}</pre>" if summary else ""
            r = urllib.parse.quote(label)
            games = "".join(f"<li><a href='/game?r={r}&g={urllib.parse.quote(g)}'>{html.escape(g)}</a></li>"
                            for g in runs.games(run))
            if not games:
                games = ("<li class=mut>No per-game model calls saved (no transcripts/, prompts/ or "
                         "tycho-results/ in this run).</li>")
            logs = "".join(f"<li><a href='/file?r={r}&f={urllib.parse.quote(p.name)}'>{html.escape(p.name)}</a></li>"
                           for p in sorted(run.glob("*.log*")))
            return _page(label, f"<div class=nav><a href='/'>Runs</a></div><h1>{html.escape(label)}</h1>"
                                f"<h2>Games</h2><ul>{games}</ul><h2>Summary</h2>{head}<h2>Logs</h2><ul>{logs}</ul>")

        def file(self, label: str, name: str) -> str:
            path = (self._run_dir(label) / name).resolve()
            if path.parent != self._run_dir(label).resolve():
                raise KeyError("outside run")
            text = _read(path)
            return text if len(text) < 400_000 else "[last 400k chars]\n" + text[-400_000:]

        def game(self, label: str, game: str, page: int) -> bytes:
            path = runs.games(self._run_dir(label))[game]
            steps = _steps(str(path), path.stat().st_mtime)
            section_re = SNAPSHOT_SECTION_RE if path.parent.name == "prompts" else SECTION_RE
            pages = max(1, -(-len(steps) // STEPS_PER_PAGE))
            page = min(max(page, 0), pages - 1)
            r, g = urllib.parse.quote(label), urllib.parse.quote(game)
            nav = [f"<a href='/run?r={r}'>{html.escape(label)}</a>"]
            for p, text in ((0, "« first"), (page - 1, "‹ prev"), (page + 1, "next ›"), (pages - 1, "last »")):
                if 0 <= p < pages and p != page:
                    nav.append(f"<a href='/game?r={r}&g={g}&p={p}'>{text}</a>")
            nav_html = f"<div class=nav>{''.join(nav)}<span class=mut>page {page + 1}/{pages}</span></div>"
            body = []
            for i, (header, content) in enumerate(steps[page * STEPS_PER_PAGE:(page + 1) * STEPS_PER_PAGE]):
                n = page * STEPS_PER_PAGE + i + 1
                body.append(f"<details open><summary>#{n} · {html.escape(header)}</summary>"
                            f"{_sections(content, section_re)}</details>")
            return _page(f"{game} · {label}",
                         f"{nav_html}<h1>{html.escape(game)}</h1><p class=mut>{len(steps)} model calls</p>"
                         f"{''.join(body)}{nav_html}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="bind address, e.g. the Tailscale IP")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--extra", action="append", default=[], help="more folders containing run dirs")
    args = parser.parse_args()
    runs = Runs([DEFAULT_ROOT, *map(Path, args.extra)])
    print(f"{len(runs.list())} runs; serving http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(runs)).serve_forever()


if __name__ == "__main__":
    main()
