import json
import os
import re
import signal
import sys
import time
import urllib.request
from pathlib import Path

CONFIG = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
SERVER_PID = int(CONFIG['server_pid'])
SERVER_PGID = CONFIG.get('server_pgid')
NOTEBOOK_PID = CONFIG.get('notebook_pid')
HEALTH_URL = str(CONFIG['health_url'])
SERVER_LOG = Path(CONFIG['server_log'])
DIED_FILE = Path(CONFIG['died_file'])
TEARDOWN_FLAG = Path(CONFIG['teardown_flag'])
LOG_FILE = Path(CONFIG['log_file'])
POLL = float(CONFIG.get('poll_seconds', 5))
HEALTH_EVERY = float(CONFIG.get('health_seconds', 30))
HEALTH_FAILURES = int(CONFIG.get('health_failures', 3))
STOP_GRACE = float(CONFIG.get('stop_grace_seconds', 240))
KILL_GRACE = float(CONFIG.get('kill_grace_seconds', 90))
FATAL_PATTERNS = tuple(CONFIG.get('fatal_patterns') or ())
TAIL_LINES = int(CONFIG.get('tail_lines', 80))
FAILFAST = bool(CONFIG.get('failfast', False))
HARNESS_LOG = Path(CONFIG['harness_log']) if CONFIG.get('harness_log') else None
STATUS_FILE = Path(CONFIG['status_file']) if CONFIG.get('status_file') else None
FF_HARNESS = [re.compile(p) for p in CONFIG.get('failfast_harness_patterns') or ()]
FF_SERVER = [re.compile(p) for p in CONFIG.get('failfast_server_patterns') or ()]
FF_ALLOW = [re.compile(p) for p in CONFIG.get('failfast_allow_patterns') or ()]
FF_WARN = [re.compile(p) for p in CONFIG.get('failfast_warn_patterns') or ()]
WARN_LOG_LIMIT = 20  # counted warnings logged one by one up to this many per pattern, then only counted
_T0 = time.monotonic()


def out(message):
    line = f'[flashnext-watchdog +{time.monotonic() - _T0:6.0f}s] {message}'
    print(line, flush=True)
    try:
        with LOG_FILE.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
    except OSError:
        pass


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f'/proc/{int(pid)}/stat').read_text(encoding='utf-8', errors='replace').rsplit(')', 1)[1].split()[0]
        return state != 'Z'
    except (OSError, IndexError):
        return True


def healthy():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=10) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def log_tail(lines=TAIL_LINES, max_bytes=512 * 1024):
    try:
        with SERVER_LOG.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode('utf-8', 'replace')
    except OSError:
        return ''
    return '\n'.join(text.splitlines()[-lines:])


def fatal_lines():
    return [line for line in log_tail(lines=4000, max_bytes=2 * 1024 * 1024).splitlines()
            if any(pattern in line for pattern in FATAL_PATTERNS)]


def teardown_started():
    return TEARDOWN_FLAG.exists()


class LineTail:
    # Complete lines appended to a file since the last call, from the start of the file. A file that
    # shrinks (recreated) is read again from the start; a partial last line waits for its newline.
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.partial = b''
        self.lines = 0

    def new_lines(self, max_bytes=8 * 1024 * 1024):
        if self.path is None:
            return []
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:
            self.offset, self.partial = 0, b''
        if size == self.offset:
            return []
        try:
            with self.path.open('rb') as handle:
                handle.seek(self.offset)
                data = handle.read(min(size - self.offset, max_bytes))
        except OSError:
            return []
        self.offset += len(data)
        parts = (self.partial + data).split(b'\n')
        self.partial = parts.pop()
        if len(self.partial) > 256 * 1024:  # a runaway line without a newline is still checked
            parts.append(self.partial)
            self.partial = b''
        self.lines += len(parts)
        return [part.decode('utf-8', 'replace').rstrip('\r') for part in parts]


WARN_COUNTS = {}
TAILS = [('harness output', LineTail(HARNESS_LOG), FF_HARNESS), ('server log', LineTail(SERVER_LOG), FF_SERVER)]


def write_status(stopped_by=None):
    if STATUS_FILE is None:
        return
    status = {
        'failfast': FAILFAST,
        'warning_counts': WARN_COUNTS,
        'lines_checked': {source: tail.lines for source, tail, _ in TAILS},
        'stopped_by': stopped_by,
        'updated': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }
    try:
        STATUS_FILE.write_text(json.dumps(status, indent=2), encoding='utf-8')
    except OSError:
        pass


def classify(line, stop_patterns):
    # 'allow' and 'warn' win over the stop patterns (the xgrammar line is also an ERROR line).
    if any(pattern.search(line) for pattern in FF_ALLOW):
        return None, None
    for pattern in FF_WARN:
        if pattern.search(line):
            return 'warn', pattern.pattern
    if any(pattern.search(line) for pattern in stop_patterns):
        return 'stop', None
    return None, None


def failfast_error():
    # The first stop-pattern line appended to the harness output or the server log since the last call.
    for source, tail, stop_patterns in TAILS:
        for line in tail.new_lines():
            kind, pattern = classify(line, stop_patterns)
            if kind == 'warn':
                count = WARN_COUNTS[pattern] = WARN_COUNTS.get(pattern, 0) + 1
                if count <= WARN_LOG_LIMIT:
                    out(f'counted warning #{count} ({source}, not a stop): {line.strip()[:300]}')
                write_status()
            elif kind == 'stop':
                return source, line.strip()
    return None


def file_tail(path, lines):
    if path is None:
        return ''
    try:
        with path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 512 * 1024))
            text = handle.read().decode('utf-8', 'replace')
    except OSError:
        return ''
    return '\n'.join(text.splitlines()[-lines:])


def wait_gone(pid, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid) or teardown_started():
            return True
        time.sleep(min(POLL, 1.0))
    return not alive(pid)


def main():
    out(f'watching vLLM pid {SERVER_PID}: {HEALTH_URL} every {HEALTH_EVERY:.0f}s ({HEALTH_FAILURES} failures in a row = dead), '
        f'log patterns {list(FATAL_PATTERNS)}; notebook pid {NOTEBOOK_PID}')
    if FAILFAST:
        out(f'fail-fast ON: the first line in {HARNESS_LOG} matching {[p.pattern for p in FF_HARNESS]} or in {SERVER_LOG} '
            f'matching {[p.pattern for p in FF_SERVER]} stops the run (ignored: {[p.pattern for p in FF_ALLOW]}; '
            f'counted only: {[p.pattern for p in FF_WARN]})')
    write_status()
    failures = 0
    last_health = time.monotonic()
    reason = None
    failfast_hit = None
    while reason is None:
        if teardown_started():
            out('teardown started; exiting')
            return
        if NOTEBOOK_PID and not alive(NOTEBOOK_PID):
            out('the notebook process is gone; exiting')
            return
        if not alive(SERVER_PID):
            reason = f'the vLLM server process (pid {SERVER_PID}) is gone'
        elif time.monotonic() - last_health >= HEALTH_EVERY:
            last_health = time.monotonic()
            if healthy():
                failures = 0
            else:
                failures += 1
                out(f'health check {HEALTH_URL} failed ({failures}/{HEALTH_FAILURES})')
                if failures >= HEALTH_FAILURES:
                    reason = f'{HEALTH_URL} failed {failures} times in a row'
        if reason is None:
            fatal = fatal_lines()
            if fatal:
                reason = 'fatal error in the server log: ' + fatal[-1].strip()[:300]
        if reason is None and FAILFAST:
            failfast_hit = failfast_error()
            if failfast_hit:
                reason = f'first error in the {failfast_hit[0]}: {failfast_hit[1][:400]}'
        if reason is None:
            time.sleep(POLL)
    tail = log_tail()
    headline = 'RUN STOPPED (FAIL-FAST): ' if failfast_hit else 'FLASHNEXT SERVER DIED: '
    message = (headline + reason + '\n'
               f'detected by the watchdog at {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}; the harness stops on this file '
               f'(ARC3_RUN_STOP_FILE={DIED_FILE}); the notebook is interrupted if it is still running in {STOP_GRACE:.0f}s.\n'
               f'--- {SERVER_LOG.name}, last {TAIL_LINES} lines ---\n' + tail + '\n')
    if failfast_hit and HARNESS_LOG is not None:
        message += f'--- {HARNESS_LOG.name}, last {TAIL_LINES} lines ---\n' + file_tail(HARNESS_LOG, TAIL_LINES) + '\n'
    write_status(stopped_by=headline + reason)
    try:
        DIED_FILE.write_text(message, encoding='utf-8')
    except OSError as exc:
        out(f'could not write {DIED_FILE}: {exc!r}')
    out(message)
    if SERVER_PGID:
        try:
            os.killpg(int(SERVER_PGID), signal.SIGKILL)  # leftover engine/worker processes hold GPU memory
            out(f'sent SIGKILL to the server process group {SERVER_PGID}')
        except OSError:
            pass
    if not NOTEBOOK_PID:
        out('no notebook process known: the stop file is the only signal; not terminating anything')
        return
    if wait_gone(NOTEBOOK_PID, STOP_GRACE):
        out('the notebook run ended by itself' if not alive(NOTEBOOK_PID) else 'teardown started; exiting')
        return
    for sig, name in ((signal.SIGINT, 'SIGINT'), (signal.SIGTERM, 'SIGTERM'), (signal.SIGKILL, 'SIGKILL')):
        out(f'the notebook process {NOTEBOOK_PID} is still running {time.monotonic() - _T0:.0f}s after the server died; sending {name}')
        try:
            os.kill(int(NOTEBOOK_PID), sig)
        except OSError as exc:
            out(f'{name} failed: {exc!r}')
            return
        if wait_gone(NOTEBOOK_PID, KILL_GRACE):
            out('the notebook process ended' if not alive(NOTEBOOK_PID) else 'teardown started; exiting')
            return
    out(f'the notebook process {NOTEBOOK_PID} survived SIGKILL; giving up')


main()
