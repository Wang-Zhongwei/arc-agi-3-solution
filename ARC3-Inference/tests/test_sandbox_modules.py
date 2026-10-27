"""The sandbox's helper modules work and its I/O modules stay out."""

import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.grid_utils import format_grid_ascii

GRID = [[0, 0], [0, 9]]


def _run(code):
    frame = {"ascii": format_grid_ascii(GRID), "step": 0, "level": 1, "shape": [2, 2], "grid": GRID}
    return run_sandboxed_python(
        code=code, timeout_seconds=20,
        initial_state={"current_frame": frame, "history": [{"action": "", "frame": frame}]},
        action_handler=lambda *a, **k: {},
    )


def test_helper_modules_import_and_work():
    out = _run(
        "import abc, array, cmath, dataclasses, decimal, enum, hashlib, inspect, numbers, time, types, typing\n"
        # The sandbox has no __build_class__, so no class statements: use the
        # functional forms.
        "P = dataclasses.make_dataclass('P', [('r', int), ('c', int)])\n"
        "Dir = enum.Enum('Dir', 'UP')\n"
        "def f(a, b=2):\n    return a\n"
        "result = [P(1, 2).c, Dir.UP.value, hashlib.sha1(b'x').hexdigest()[:4],\n"
        "          str(inspect.signature(f)), isinstance(time.time(), float),\n"
        "          str(decimal.Decimal('1.5') * 2), abs(cmath.sqrt(-4)), array.array('i', [1, 2])[1],\n"
        "          isinstance(3, numbers.Integral), isinstance(f, types.FunctionType),\n"
        "          typing.get_origin(typing.List[int]) is list, isinstance(abc.ABC, type)]\n"
    )
    assert out.get("result") == [2, 1, "11f6", "(a, b=2)", True, "3.0", 2.0, 2, True, True, True, True], out


@pytest.mark.parametrize("module", ["os", "sys", "pathlib", "glob", "pickle", "importlib", "builtins", "io", "shutil"])
def test_io_modules_stay_blocked(module):
    out = _run(f"import {module}\nresult = 1")
    assert "not allowed in the sandbox" in str(out.get("error") or out), out


def test_open_is_not_a_builtin():
    out = _run("open('x.txt', 'w')\nresult = 1")
    assert "open" in str(out.get("error") or out) and out.get("result") != 1, out



def test_listed_modules_and_collections_helpers_are_preloaded():
    # a forgotten import cost 2-10 NameErrors a run (Counter, deque, collections, itertools, math)
    out = _run(
        "result = [Counter('aab')['a'], deque([1, 2]).pop(), defaultdict(int)['x'],\n"
        "          math.isqrt(10), len(list(itertools.permutations(range(3)))), collections.OrderedDict is not None,\n"
        "          heapq.nsmallest(1, [3, 1]), json.dumps([1]), re.sub('a', 'b', 'a'), statistics.mean([1, 3])]\n"
    )
    assert out.get("result") == [2, 2, 0, 3, 6, True, [1], "[1]", "b", 2], out


def test_preloaded_modules_still_import_and_can_be_shadowed():
    out = _run("import math\nfrom collections import Counter\nre = 5\nresult = [math.pi > 3, Counter('a')['a'], re]")
    assert out.get("result") == [True, 1, 5], out
