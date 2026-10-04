// Run with: node tests/test_curves_levels.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, '../viewer/curves.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context = {URLSearchParams, location: {hash: ''}, history: {
  replaceState(_state, _title, hash) { context.location.hash = hash; },
}};
vm.createContext(context);
vm.runInContext(script.slice(0, script.indexOf('  fetch("/api/curves"')) + `
  globalThis.api = {runSeries, stepValue, state, saveHash, loadHash,
    setRuns(runs) { RUNS = runs; ALL_GAMES = ['aa01']; GAME_INFO = new Map([['aa01', {}]]); }};
})();`, context);
const {api} = context;
const run = {run_id: 'kaggle/owner/run/v01', games: {aa01: {
  score: 40, trial_scores: {p0: 30, p1: 50}, levels_completed_mean: 1.5,
  generated_tokens_mean: 20, actions_mean: 6,
}}, lanes: {
  tokens: {aa01: {p0: [[0, 0], [10, 30]], p1: [[0, 0], [20, 1], [30, 50]]}},
}, level_lanes: {
  tokens: {aa01: {p0: [[0, 0], [10, 1]], p1: [[0, 0], [20, 1], [30, 2]]}},
  actions: {aa01: {p0: [[0, 0], [4, 1]], p1: [[0, 0], [5, 1], [8, 2]]}},
}};
const copy = value => JSON.parse(JSON.stringify(value));
const score = api.runSeries(run, 'tokens', ['aa01']);
const levels = api.runSeries(run, 'tokens', ['aa01'], 'levels');
assert.deepEqual(copy(score.curve.mean.at(-1)), [20, 40]);
assert.deepEqual(copy(levels.curve.mean.at(-1)), [20, 1.5]);
assert.deepEqual(copy(levels.endpoint), {x: 20, mean: 1.5, median: 1.5});
assert.deepEqual(copy(api.runSeries(run, 'actions', ['aa01'], 'levels').curve.mean.at(-1)), [6, 1.5]);
assert.equal(api.stepValue([[0, 0], [10, 1], [10, 2]], 10), 2);

// Missing a pass must fall back to the recorded endpoint, without inventing a curve.
const incomplete = {...run, run_id: 'incomplete', level_lanes: {
  tokens: {aa01: {p0: run.level_lanes.tokens.aa01.p0}},
}};
assert.equal(api.runSeries(incomplete, 'tokens', ['aa01'], 'levels').curve, null);
assert.equal(api.runSeries(incomplete, 'tokens', ['aa01'], 'levels').endpoint.mean, 1.5);
const unknown = {...incomplete, run_id: 'unknown', games: {
  aa01: {...run.games.aa01, levels_completed_mean: null},
}};
assert(Number.isNaN(api.runSeries(unknown, 'tokens', ['aa01'], 'levels').endpoint.mean));

api.setRuns([run]);
api.state.view = 'games';
api.state.axis = 'actions';
api.state.metric = 'levels';
api.state.pins = [{id: run.run_id, slot: 3}];
api.saveHash();
assert(context.location.hash.includes('metric=levels'));
api.state.metric = 'score';
api.loadHash();
assert.equal(api.state.metric, 'levels');
assert.equal(api.state.axis, 'actions');
assert.equal(api.state.pins[0].slot, 3);
context.location.hash = '#view=games&pins=3%3Akaggle%2Fowner%2Frun%2Fv01';
api.loadHash();
assert.equal(api.state.metric, 'score', 'Existing links retain score as the default');
console.log('Score/levels pooling, action costs, missing histories, and URL state passed.');
