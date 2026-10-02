"""Execute viewer helpers and loading behavior in Node without a browser."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest


VIEWER = Path(__file__).parents[1] / "viewer"


def _function(html: str, name: str) -> str:
    match = re.search(rf"^    (?:async )?function {name}\(.*?^    }}", html, re.M | re.S)
    assert match is not None
    return match.group()


def _run_js(source: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for viewer JavaScript checks")
    subprocess.run([node, "-e", source], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("page", ["index.html", "comparison.html"])
def test_medians_ignore_missing_values_and_preserve_zero(page: str) -> None:
    html = (VIEWER / page).read_text()
    helpers = _function(html, "median")
    if page == "index.html":
        helpers = _function(html, "finiteValues") + helpers
    _run_js(helpers + """
      const assert = require('node:assert/strict');
      assert.equal(median([9, 0, 1]), 1);
      assert.equal(median([9, 0, 1, 3]), 2);
      assert.equal(median([null, undefined, '', NaN, Infinity]), null);
      assert.equal(median([null, '', 0, 10, 2]), 2);
    """)


def test_comparison_medians_use_games_and_available_trials() -> None:
    html = (VIEWER / "comparison.html").read_text()
    helpers = "\n".join(_function(html, name) for name in ("median", "medianGameScore", "metricValue"))
    _run_js(helpers + """
      const assert = require('node:assert/strict');
      assert.equal(medianGameScore({games:{a:{score:0},b:{score:2},c:{score:40}}}), 2);
      const definition = {median:true, field:'trial_scores'};
      assert.equal(metricValue({trial_scores:{a:0,b:0,c:30}}, definition), 0);
      assert.ok(Number.isNaN(metricValue({score:10}, definition)));
    """)


def test_home_load_defers_trace_until_trace_view() -> None:
    html = (VIEWER / "index.html").read_text()
    _run_js(_function(html, "loadPayload") + """
      const assert = require('node:assert/strict');
      let currentRunName = 'run', isLoadingPayload = false, currentGameLoadToken = 0;
      let activeStepIndex = 0, currentGameIndex = 0, payload = null;
      let currentSteps = ['old'], currentGameSignature = 'old', currentView = 'results';
      const runSubtitleEl = {}, window = {location:{origin:'http://localhost'}};
      let gameLoads = 0;
      const fetch = async () => ({ok:true, json:async () => ({selected_run:'run',games:[{}]})});
      const renderHeader = () => {};
      const loadGameDetails = async () => { gameLoads++; };
      (async () => {
        await loadPayload();
        assert.equal(gameLoads, 0);
        assert.deepEqual(currentSteps, []);
        assert.equal(isLoadingPayload, false);
        currentView = 'trace';
        await loadPayload();
        assert.equal(gameLoads, 1);
      })().catch(error => { console.error(error); process.exitCode = 1; });
    """)
