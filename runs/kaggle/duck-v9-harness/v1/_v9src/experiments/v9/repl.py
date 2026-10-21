"""Parent side of the read-only REPL: one child per game, JSON snapshots in, results out.

Offered to the model only in ORIENT and RECOVER (a tool round-trip is a full request); in the
other modes the harness precomputes the counts and diffs the REPL was for.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

CHILD = Path(__file__).with_name("repl_child.py")


class Repl:
    def __init__(self, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()

    def _ensure(self) -> subprocess.Popen:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen([sys.executable, "-I", "-S", str(CHILD)], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                         encoding="utf-8", start_new_session=True)
        return self.proc

    def snapshot(self, payload: dict[str, Any]) -> None:
        with self.lock:
            p = self._ensure()
            assert p.stdin is not None
            p.stdin.write(json.dumps({"type": "snapshot", **payload}, default=str) + "\n")
            p.stdin.flush()

    def run(self, code: str) -> dict[str, Any]:
        with self.lock:
            p = self._ensure()
            assert p.stdin is not None and p.stdout is not None
            p.stdin.write(json.dumps({"type": "exec", "code": code}) + "\n")
            p.stdin.flush()
            line: list[str] = []

            def reader() -> None:
                line.append(p.stdout.readline())

            t = threading.Thread(target=reader, daemon=True)
            t.start()
            t.join(self.timeout_s)
            if t.is_alive():
                p.kill()
                self.proc = None
                return {"stdout": "", "result": None, "error": f"REPL timed out after {self.timeout_s:.0f}s"}
            try:
                return json.loads(line[0]) if line and line[0] else {"stdout": "", "result": None, "error": "REPL exited"}
            except json.JSONDecodeError:
                return {"stdout": "", "result": None, "error": "REPL returned invalid output"}

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None
