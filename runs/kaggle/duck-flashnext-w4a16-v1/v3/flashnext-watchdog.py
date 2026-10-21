import json
import os
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
    failures = 0
    last_health = time.monotonic()
    reason = None
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
        if reason is None:
            time.sleep(POLL)
    tail = log_tail()
    message = ('FLASHNEXT SERVER DIED: ' + reason + '\n'
               f'detected by the watchdog at {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}; the harness stops on this file '
               f'(ARC3_RUN_STOP_FILE={DIED_FILE}); the notebook is interrupted if it is still running in {STOP_GRACE:.0f}s.\n'
               f'--- {SERVER_LOG.name}, last {TAIL_LINES} lines ---\n' + tail + '\n')
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
