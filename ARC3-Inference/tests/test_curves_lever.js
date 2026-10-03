// Run with: node tests/test_curves_lever.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, '../viewer/curves.html'), 'utf8');
const script = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).find(value => value.trim());

function element(tag) {
  return {tag, children:[], listeners:{}, attributes:{}, disabled:false,
    setAttribute(key, value) { this.attributes[key] = value; },
    addEventListener(key, listener) { this.listeners[key] = listener; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    querySelector(tag) {
      for (const child of this.children) {
        if (child.tag === tag) return child;
        const match = child.querySelector(tag);
        if (match) return match;
      }
      return null;
    },
    focus() { this.focused = true; },
    click() { if (!this.disabled) return this.listeners.click?.(); },
  };
}

async function main() {
  const requests = [];
  const run = {run_id:'kaggle/owner/notebook/v01', annotation:{leaderboard_score:47.2, notes:'Original lever'}};
  const context = {run, redraws:0, document:{createElement:element},
    renderTable() { context.redraws++; },
    async fetch(url, options) {
      const body = JSON.parse(options.body);
      requests.push({url, options, body});
      return {ok:true, json:async () => ({annotation:{...body.annotation, updated_at:'saved'}})};
    },
  };
  vm.createContext(context);
  vm.runInContext(script.slice(script.indexOf('  function el('), script.indexOf('  function svg(')), context);
  vm.runInContext(script.slice(script.indexOf('  async function saveLever('), script.indexOf('  function renderTable(')), context);
  const makeCell = () => vm.runInContext('leverCell(run)', context);
  const editor = cell => ({input:cell.children[0], save:cell.children[1].children[0],
    cancel:cell.children[1].children[1], error:cell.children[1].children[2]});

  let cell = makeCell();
  assert.equal(cell.children[0].textContent, 'Original lever');
  let stopped = false;
  cell.listeners.click({stopPropagation() { stopped = true; }});
  assert(stopped, 'Clicking the lever must not pin the row');
  cell.children[0].click();
  assert.equal(editor(cell).input.value, 'Original lever');
  editor(cell).input.value = 'Discard this';
  editor(cell).cancel.click();
  assert.equal(cell.children[0].textContent, 'Original lever');
  assert.equal(requests.length, 0);

  cell.children[0].click();
  const notes = 'Changed budget\n<example> & extra details';
  editor(cell).input.value = notes;
  await editor(cell).save.click();
  assert.equal(requests[0].url, '/api/run-annotation');
  assert.equal(requests[0].options.method, 'POST');
  assert.deepEqual(requests[0].body, {run_id:run.run_id, annotation:{leaderboard_score:47.2, notes}});
  assert.equal(run.annotation.notes, notes);
  assert.equal(context.redraws, 1);
  cell = makeCell();
  assert.equal(cell.children[0].textContent, notes, 'Notes must render as text');

  context.fetch = async () => ({ok:false, json:async () => ({error:'Unable to write notes'})});
  cell.children[0].click();
  const failed = editor(cell);
  failed.input.value = 'Unsaved change';
  await failed.save.click();
  assert.equal(failed.error.textContent, 'Unable to write notes');
  assert.equal(run.annotation.notes, notes);
  assert.equal(failed.save.disabled, false);
  assert.equal(failed.input.disabled, false);
  assert.equal(context.redraws, 1);
  failed.input.listeners.keydown({key:'Escape', preventDefault() {}});
  assert.equal(cell.children[0].textContent, notes);

  context.fetch = async (url, options) => ({ok:true,
    json:async () => ({annotation:JSON.parse(options.body).annotation})});
  cell.children[0].click();
  editor(cell).input.value = '';
  await editor(cell).save.click();
  assert.equal(makeCell().children[0].textContent, 'Add lever…');
  assert.equal(run.annotation.leaderboard_score, 47.2);
  console.log('Lever edit, cancel, save, clear, error recovery, and row-click isolation passed.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
