"""Periodic Python stack dumps for the SGLang server's processes (loader diagnosis).

On the server's PYTHONPATH only when the config sets kaggle.sglang.STACK_DUMP
(pennyroyal_launch.py). Each
server or multiprocessing child writes all thread stacks every
ARC3_STACK_DUMP_INTERVAL seconds to <ARC3_STACK_DUMP_DIR>/stacks-<pid>.txt, and
stops after ARC3_STACK_DUMP_SECONDS. Uses faulthandler, so it needs no packages
and works while a thread holds the GIL.
"""
import faulthandler
import os
import sys
import threading

_dir = os.environ.get("ARC3_STACK_DUMP_DIR")
_argv = " ".join(sys.argv)
if _dir and ("sglang" in _argv or "multiprocessing" in _argv):
    os.makedirs(_dir, exist_ok=True)
    _file = open(os.path.join(_dir, f"stacks-{os.getpid()}.txt"), "w", buffering=1)
    _file.write(f"# pid {os.getpid()} argv {_argv[:300]}\n")
    faulthandler.dump_traceback_later(float(os.environ.get("ARC3_STACK_DUMP_INTERVAL", "15")), repeat=True, file=_file)
    _stop = threading.Timer(float(os.environ.get("ARC3_STACK_DUMP_SECONDS", "2400")), faulthandler.cancel_dump_traceback_later)
    _stop.daemon = True
    _stop.start()
