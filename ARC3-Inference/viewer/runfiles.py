"""Run-folder access for the viewer: transparent .xz files and run discovery.

Saved Kaggle runs (runs/kaggle/<kernel>/v<N>/) keep every file larger than
64 KB as ``<name>.xz``. Everything here takes the *logical* path (the name
without ``.xz``) and reads whichever of ``<name>`` or ``<name>.xz`` exists, so
the viewer can open those runs without unpacking anything to disk.

Run discovery walks ``--runs-dir`` a few levels deep and treats any folder
holding run output (summary.txt, benchmark.json, artifacts/, transcripts/,
passes/ or seeds/) as a run. Its id is the path relative to ``--runs-dir``
(``fz-ledger-on-trim-r2/v1``), or the folder name for the legacy
timestamp-named runs that sit directly under it.
"""
from __future__ import annotations

import json
import lzma
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from inference.utils.run_artifacts import is_selectable_run_dir_name, run_dir_sort_key


XZ_SUFFIX = ".xz"
_RUN_MARKER_FILES = ("summary.txt", "benchmark.json")
_RUN_MARKER_DIRS = ("artifacts", "transcripts", "passes", "seeds")
# Sub-folders of a run that never hold another run; skipping them keeps the scan cheap.
_NON_RUN_DIRS = {
    "artifacts", "transcripts", "prompts", "movies", "solver_analysis", "passes", "seeds",
    "__pycache__", "node_modules",
}
_MAX_SCAN_DEPTH = 3


# --- transparent .xz -------------------------------------------------------------------------

def logical_name(name: str) -> str:
    """``foo.json.xz`` -> ``foo.json``; other names unchanged."""
    return name[: -len(XZ_SUFFIX)] if name.endswith(XZ_SUFFIX) else name


def resolve(path: str | Path) -> Path | None:
    """Return the file that stores ``path``: the plain file, else ``path.xz``, else None."""
    plain = Path(path)
    if plain.name.endswith(XZ_SUFFIX):
        return plain if plain.is_file() else None
    if plain.is_file():
        return plain
    packed = plain.with_name(plain.name + XZ_SUFFIX)
    if packed.is_file():
        return packed
    return None


def exists(path: str | Path) -> bool:
    return resolve(path) is not None


def read_bytes(path: str | Path) -> bytes:
    """Read ``path`` (or ``path.xz``, decompressed). Raises FileNotFoundError if neither exists."""
    actual = resolve(path)
    if actual is None:
        raise FileNotFoundError(f"{path} (or {path}{XZ_SUFFIX}) not found")
    if actual.name.endswith(XZ_SUFFIX):
        with lzma.open(actual) as handle:
            return handle.read()
    return actual.read_bytes()


def read_text(path: str | Path) -> str:
    return read_bytes(path).decode("utf-8", errors="replace")


def read_json(path: str | Path) -> Any:
    return json.loads(read_bytes(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a .jsonl file; a torn last line (run still writing) is skipped."""
    lines = read_text(path).splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if isinstance(parsed, dict):
            yield parsed


def stat_key(path: str | Path) -> tuple[str, int, int] | None:
    """(actual path, size, mtime) of the stored file, for cache keys."""
    actual = resolve(path)
    if actual is None:
        return None
    stat = actual.stat()
    return (str(actual), stat.st_size, stat.st_mtime_ns)


def glob_logical(directory: Path, pattern: str) -> list[Path]:
    """Logical paths in ``directory`` whose logical name matches ``pattern`` (plain wins over .xz)."""
    if not directory.is_dir():
        return []
    names: set[str] = set()
    for path in directory.glob(pattern):
        names.add(path.name)
    for path in directory.glob(pattern + XZ_SUFFIX):
        names.add(logical_name(path.name))
    return [directory / name for name in sorted(names)]


# --- small thread-safe LRU cache -------------------------------------------------------------

class LruCache:
    """Bounded memo for parsed run files (decompressing a big events file once, not per click)."""

    def __init__(self, max_items: int):
        self._max_items = max_items
        self._items: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_load(self, key: Any, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = loader()
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_BENCHMARK_CACHE = LruCache(8)
_RUN_META_CACHE = LruCache(512)


def load_benchmark(run_dir: Path) -> dict[str, Any] | None:
    """Parsed benchmark.json(.xz) of a run, or None when the run has none / it is unreadable."""
    path = run_dir / "benchmark.json"
    key = stat_key(path)
    if key is None:
        return None

    def _load() -> dict[str, Any] | None:
        try:
            payload = read_json(path)
        except (OSError, ValueError, lzma.LZMAError):
            return None
        return payload if isinstance(payload, dict) else None

    return _BENCHMARK_CACHE.get_or_load(key, _load)


# --- run discovery ---------------------------------------------------------------------------

def is_run_dir(path: Path) -> bool:
    """A folder with run output in it (or a legacy timestamp-named run folder)."""
    if not path.is_dir():
        return False
    if is_selectable_run_dir_name(path.name):
        return True
    if any(exists(path / name) for name in _RUN_MARKER_FILES):
        return True
    return any((path / name).is_dir() for name in _RUN_MARKER_DIRS)


def discover_run_dirs(runs_dir: str | Path) -> list[Path]:
    """Every run folder under ``runs_dir`` (or ``runs_dir`` itself when it is one run)."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    if is_run_dir(root) and not _has_child_runs(root):
        return [root]
    found: list[Path] = []
    _scan(root, 0, found)
    return found


def _has_child_runs(path: Path) -> bool:
    try:
        children = [child for child in path.iterdir() if child.is_dir()]
    except OSError:
        return False
    return any(child.name not in _NON_RUN_DIRS and is_run_dir(child) for child in children)


def _scan(directory: Path, depth: int, found: list[Path]) -> None:
    if depth >= _MAX_SCAN_DEPTH:
        return
    try:
        children = sorted(child for child in directory.iterdir() if child.is_dir())
    except OSError:
        return
    for child in children:
        if child.name.startswith(".") or child.name in _NON_RUN_DIRS:
            continue
        if is_run_dir(child):
            found.append(child)
        else:
            _scan(child, depth + 1, found)


def run_id(run_dir: Path, runs_dir: Path) -> str:
    """Stable id used in URLs: path relative to ``runs_dir`` (posix), else the folder name."""
    try:
        relative = run_dir.resolve().relative_to(runs_dir.resolve())
    except ValueError:
        return run_dir.name
    text = relative.as_posix()
    return text if text not in ("", ".") else run_dir.name


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    path: Path
    started: str | None
    mean_score: float | None
    games: int | None
    duration: str | None
    has_viewer_data: bool
    has_benchmark: bool
    has_transcripts: bool
    sort_key: tuple[Any, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.run_id,
            "started": self.started,
            "mean_score": self.mean_score,
            "games": self.games,
            "duration": self.duration,
            "has_viewer_data": self.has_viewer_data,
            "has_benchmark": self.has_benchmark,
            "has_transcripts": self.has_transcripts,
        }


_SUMMARY_LINE_RE = re.compile(r"^(?P<key>[a-z][a-z /]*?):\s+(?P<value>.+?)\s*$")


def parse_summary_txt(text: str) -> dict[str, str]:
    """``key: value`` header lines of a run's summary.txt (started, mean score, games, ...)."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if fields:
                break
            continue
        match = _SUMMARY_LINE_RE.match(line)
        if match:
            fields[match.group("key").strip()] = match.group("value")
    return fields


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_started(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def has_viewer_data(run_dir: Path) -> bool:
    artifacts = run_dir / "artifacts"
    if artifacts.is_dir() and glob_logical(artifacts, "*viewer_data.json"):
        return True
    for split in ("passes", "seeds"):
        split_dir = run_dir / split
        if split_dir.is_dir() and any(split_dir.glob("*/artifacts/*viewer_data.json*")):
            return True
    return False


def run_meta(run_dir: Path, runs_dir: Path) -> RunMeta:
    """Header facts for the run picker; summary.txt first, benchmark.json as the fallback."""
    marker_keys = tuple(
        stat_key(run_dir / name) for name in ("summary.txt", "benchmark.json")
    )
    try:
        dir_mtime = run_dir.stat().st_mtime_ns
    except OSError:
        dir_mtime = 0
    cache_key = (str(run_dir.resolve()), str(runs_dir.resolve()), marker_keys, dir_mtime)
    return _RUN_META_CACHE.get_or_load(cache_key, lambda: _build_run_meta(run_dir, runs_dir))


def _build_run_meta(run_dir: Path, runs_dir: Path) -> RunMeta:
    summary: dict[str, str] = {}
    if exists(run_dir / "summary.txt"):
        try:
            summary = parse_summary_txt(read_text(run_dir / "summary.txt"))
        except OSError:
            summary = {}
    started = summary.get("started")
    mean_score = _float_or_none(summary.get("mean score"))
    games = _int_or_none(summary.get("games"))
    duration = summary.get("duration")
    has_benchmark = exists(run_dir / "benchmark.json")
    if has_benchmark and (started is None or mean_score is None or games is None):
        benchmark = load_benchmark(run_dir) or {}
        game_runs = [run for run in benchmark.get("game_runs") or [] if isinstance(run, dict)]
        if started is None and benchmark.get("start_time"):
            parsed = _parse_started(benchmark.get("start_time"))
            started = parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else str(benchmark["start_time"])
        if games is None and game_runs:
            games = len({str(run.get("game_id")) for run in game_runs})
        if mean_score is None and game_runs:
            scores = [_float_or_none(run.get("final_score")) or 0.0 for run in game_runs]
            mean_score = sum(scores) / len(scores)
    started_dt = _parse_started(started)
    if started_dt is not None:
        sort_key: tuple[Any, ...] = (2, started_dt.isoformat())
    elif is_selectable_run_dir_name(run_dir.name):
        sort_key = (1, "".join(str(part) for part in run_dir_sort_key(run_dir.name)))
    else:
        try:
            sort_key = (0, str(run_dir.stat().st_mtime_ns).zfill(24))
        except OSError:
            sort_key = (0, "")
    return RunMeta(
        run_id=run_id(run_dir, runs_dir),
        path=run_dir,
        started=started,
        mean_score=round(mean_score, 2) if mean_score is not None else None,
        games=games,
        duration=duration,
        has_viewer_data=has_viewer_data(run_dir),
        has_benchmark=has_benchmark,
        has_transcripts=(run_dir / "transcripts").is_dir(),
        sort_key=sort_key,
    )


def list_runs(runs_dir: str | Path, extra_run_dir: Path | None = None) -> list[RunMeta]:
    """All runs under ``runs_dir`` (plus ``extra_run_dir``), newest first.

    "Newest" is the run's own start time (summary.txt / benchmark.json), not
    the folder mtime: downloaded Kaggle outputs all share their download time.
    """
    root = Path(runs_dir)
    run_dirs = discover_run_dirs(root)
    if extra_run_dir is not None and extra_run_dir.is_dir():
        resolved = {path.resolve() for path in run_dirs}
        if extra_run_dir.resolve() not in resolved:
            run_dirs.append(extra_run_dir)
    metas = [run_meta(path, root) for path in run_dirs]
    return sorted(metas, key=lambda meta: (meta.sort_key, meta.run_id), reverse=True)
