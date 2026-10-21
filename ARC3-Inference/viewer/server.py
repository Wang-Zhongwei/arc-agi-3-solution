"""HTTP server for the ARC3 run viewer (desktop and phone).

Serves index.html plus a small JSON API over saved runs, including
runs/kaggle/<kernel>/v<N>/ folders whose large files are stored as .xz:

    inference-view --host <tailscale-ip> --port 8011 --runs-dir ../runs/kaggle

Endpoints (all GET, gzip when the client accepts it):
    /api/runs                         every run, newest first, with mean score
    /api/run?run=ID                   one run's games
    /api/game?run=ID&index=I[&lite=1] one game's turns + per-action rows (lite: no per-turn boards)
    /api/game-step?...&step=S[&lite=1] one turn's reasoning (lite: drop the full model window)
    /api/frames?run=ID&index=I        every board of the game, compactly encoded
    /api/timeline?run=ID[&game=KEY]   score / levels / tokens vs wall-clock since run start
    /api/file?run=ID&path=P           a run's solver_analysis/*.html, transcripts/*.txt or summary.txt
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from viewer import runfiles
from viewer.data import (
    list_runs_payload,
    load_game_frames_payload,
    load_game_payload,
    load_game_shell_payload,
    load_game_step_payload,
    load_run_summary,
    load_run_timeline_payload,
    strip_step_boards,
    strip_step_context,
)


log = logging.getLogger(__name__)
_GZIP_MIN_BYTES = 1024


@dataclass(frozen=True)
class _ResponseBody:
    body: bytes
    is_gzipped: bool


def _index_html_path() -> Path:
    return Path(__file__).resolve().parent / "index.html"


def _load_index_html() -> str:
    return _index_html_path().read_text(encoding="utf-8")


def _index_html_version() -> int:
    return _index_html_path().stat().st_mtime_ns


_SERVABLE_RUN_FILES = (
    ("solver_analysis/", ".html", "text/html; charset=utf-8"),
    ("transcripts/", ".txt", "text/plain; charset=utf-8"),
    ("", "summary.txt", "text/plain; charset=utf-8"),
)


def _is_within(path: Path, root: Path) -> bool:
    resolved, base = path.resolve(), root.resolve()
    return resolved == base or base in resolved.parents


def _requested_run_dir(*, runs_dir: Path, default_run_dir: Path | None, requested_run: str | None) -> Path | None:
    """Map a run id from the URL to a folder, refusing anything outside ``runs_dir``.

    Ids are paths relative to ``runs_dir`` (``fz-ledger-on-trim-r2/v1``); the
    ``--run-dir`` run is also reachable by its name. The viewer may be served
    on a tailnet address, so ``..`` and absolute paths are not followed."""
    requested_name = str(requested_run or "").strip()
    if not requested_name:
        return default_run_dir

    if default_run_dir is not None and requested_name in {
        default_run_dir.name,
        runfiles.run_id(default_run_dir, runs_dir),
        str(default_run_dir),
    }:
        return default_run_dir

    if runs_dir.is_dir() and requested_name == runs_dir.name and runfiles.is_run_dir(runs_dir):
        return runs_dir

    candidate = runs_dir / requested_name
    if not _is_within(candidate, runs_dir):
        raise FileNotFoundError(f"Run {requested_name!r} is not under {runs_dir}")
    return candidate


def _run_file_path(run_dir: Path, relative: str) -> tuple[Path, str] | None:
    """Logical path + content type of an allowed run file, or None."""
    relative = str(relative or "").strip().lstrip("/")
    if not relative or ".." in Path(relative).parts:
        return None
    for prefix, suffix, content_type in _SERVABLE_RUN_FILES:
        if relative.startswith(prefix) and relative.endswith(suffix) and (prefix or relative == suffix):
            candidate = run_dir / relative
            if _is_within(candidate, run_dir) and runfiles.exists(candidate):
                return candidate, content_type
    return None


def _flag(params: dict[str, list[str]], name: str) -> bool:
    return params.get(name, ["false"])[0].lower() in {"1", "true", "yes", "on"}


class _ViewerHandler(BaseHTTPRequestHandler):
    """Serve the viewer shell and run payload API."""

    runs_dir: Path
    run_dir: Path | None

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route(urlparse(self.path))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 - one bad run file must not kill the request silently
            log.exception("viewer request failed: %s", self.path)
            try:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            except OSError:
                pass

    def _route(self, parsed) -> None:
        if parsed.path in {"/", "/index.html"}:
            self._send_html(_load_index_html())
            return
        if parsed.path == "/api/viewer-version":
            self._send_json({"version": _index_html_version()})
            return
        if parsed.path == "/api/run":
            self._handle_run_api(parsed.query)
            return
        if parsed.path == "/api/game":
            self._handle_game_api(parsed.query)
            return
        if parsed.path == "/api/game-step":
            self._handle_game_step_api(parsed.query)
            return
        if parsed.path == "/api/runs":
            self._send_json(list_runs_payload(runs_dir=self.runs_dir, run_dir=self.run_dir))
            return
        if parsed.path == "/api/frames":
            self._handle_frames_api(parsed.query)
            return
        if parsed.path == "/api/timeline":
            self._handle_timeline_api(parsed.query)
            return
        if parsed.path == "/api/file":
            self._handle_file_api(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _run_dir_from(self, params: dict[str, list[str]]) -> Path | None:
        return _requested_run_dir(
            runs_dir=self.runs_dir,
            default_run_dir=self.run_dir,
            requested_run=params.get("run", [None])[0],
        )

    def _handle_frames_api(self, query: str) -> None:
        params = parse_qs(query)
        try:
            game_index = int(str(params.get("index", [None])[0]))
        except (TypeError, ValueError):
            self._send_json({"error": "Missing or invalid game index."}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = load_game_frames_payload(
                runs_dir=self.runs_dir, run_dir=self._run_dir_from(params), game_index=game_index
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(payload)

    def _handle_timeline_api(self, query: str) -> None:
        params = parse_qs(query)
        try:
            payload = load_run_timeline_payload(
                runs_dir=self.runs_dir,
                run_dir=self._run_dir_from(params),
                game_key=params.get("game", [None])[0],
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc), "available": False}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(payload)

    def _handle_file_api(self, query: str) -> None:
        params = parse_qs(query)
        try:
            run_dir = self._run_dir_from(params)
        except FileNotFoundError:
            run_dir = None
        found = _run_file_path(run_dir, params.get("path", [""])[0]) if run_dir is not None else None
        if found is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        path, content_type = found
        content = self._maybe_gzip(runfiles.read_bytes(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        if content.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(content.body)))
        self.end_headers()
        self.wfile.write(content.body)

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _handle_run_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        try:
            payload = load_run_summary(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc), "games": []}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}", "games": []}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(payload)

    def _handle_game_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        raw_index = params.get("index", [None])[0]
        try:
            game_index = int(str(raw_index))
        except (TypeError, ValueError):
            self._send_json({"error": "Missing or invalid game index."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            full_payload = _flag(params, "full")
            loader = load_game_payload if full_payload else load_game_shell_payload
            payload = loader(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
                game_index=game_index,
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if _flag(params, "lite") and not full_payload:
            payload = strip_step_boards(payload)
        self._send_json(payload)

    def _handle_game_step_api(self, query: str) -> None:
        params = parse_qs(query)
        requested_run = params.get("run", [None])[0]
        raw_game_index = params.get("index", [None])[0]
        raw_step_index = params.get("step", [None])[0]
        try:
            game_index = int(str(raw_game_index))
            step_index = int(str(raw_step_index))
        except (TypeError, ValueError):
            self._send_json({"error": "Missing or invalid game/step index."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            payload = load_game_step_payload(
                runs_dir=self.runs_dir,
                run_dir=_requested_run_dir(
                    runs_dir=self.runs_dir,
                    default_run_dir=self.run_dir,
                    requested_run=requested_run,
                ),
                game_index=game_index,
                step_index=step_index,
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid viewer artifact JSON: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if _flag(params, "lite"):
            payload = strip_step_context(payload)
        self._send_json(payload)

    def _send_html(self, html: str) -> None:
        content = html.encode("utf-8")
        content = self._maybe_gzip(content)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if content.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(content.body)))
        self.end_headers()
        self.wfile.write(content.body)

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        content = self._maybe_gzip(content)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        if content.is_gzipped:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(content.body)))
        self.end_headers()
        self.wfile.write(content.body)

    def _maybe_gzip(self, content: bytes) -> "_ResponseBody":
        accept_encoding = self.headers.get("Accept-Encoding", "")
        if len(content) < _GZIP_MIN_BYTES or "gzip" not in accept_encoding.lower():
            return _ResponseBody(body=content, is_gzipped=False)
        return _ResponseBody(body=gzip.compress(content, compresslevel=6), is_gzipped=True)


def build_handler(*, runs_dir: Path, run_dir: Path | None) -> type[_ViewerHandler]:
    """Bind configuration into the request handler class."""
    handler_cls = type("ViewerHandler", (_ViewerHandler,), {})
    handler_cls.runs_dir = runs_dir
    handler_cls.run_dir = run_dir
    return handler_cls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the ARC3 run viewer (works on phones too).",
        epilog="Phone over Tailscale: inference-view --host <tailscale-ip> --port 8011 --runs-dir ../runs/kaggle",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind. Use your Tailscale IP to reach it from a phone on the tailnet; "
        "avoid 0.0.0.0 on untrusted networks (no auth).",
    )
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Folder searched (3 levels deep) for runs, e.g. ../runs/kaggle for <kernel>/v<N>/ folders.",
    )
    parser.add_argument("--run-dir", default=None, help="Optional run directory to open first.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    runs_dir = Path(args.runs_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None
    handler = build_handler(runs_dir=runs_dir, run_dir=run_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    target = run_dir if run_dir is not None else runs_dir
    log.info("Viewer serving %s at http://%s:%d", target, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Viewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
