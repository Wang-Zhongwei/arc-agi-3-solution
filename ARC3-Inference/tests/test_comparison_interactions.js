// Run with: node tests/test_comparison_interactions.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(path.join(__dirname, '../viewer/comparison.html'), 'utf8');
const script = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
  .map(match => match[1]).find(value => value.trim());
function element() {
  return {style:{}, attributes:{}, listeners:{}, innerHTML:'', offsetWidth:180, offsetHeight:100,
    clientWidth:1000, setAttribute(key,value){this.attributes[key]=value;},
    addEventListener(key,listener){this.listeners[key]=listener;},
    getBoundingClientRect(){return {left:0,top:0,width:1000,height:375};},
    setPointerCapture(){}};
}
const root=element(), overlay=element(), cross=element(), dots=element(), tip=element();
const parts={'svg':root,'.progress-overlay':overlay,'.progress-crosshair':cross,
  '.progress-dots':dots,'.progress-tooltip':tip};
const wrap=element(); wrap.dataset={progressChart:'mean'}; wrap.querySelector=selector=>parts[selector];
const context={console,redraws:0,fetch:()=>new Promise(()=>{}),
  document:{getElementById:()=>element(),querySelectorAll:selector=>selector==='[data-progress-chart]'?[wrap]:[]}};
vm.createContext(context);vm.runInContext(script,context);
vm.runInContext(`
  function score(points) {return {scores:{public:{score:{score_progression:{status:'complete',points,median_points:points}}}}};}
  registry={runs:{a:score([[0,0],[100,10],[200,20]]),b:score([[0,0],[100,5],[150,12]])}};
  baselineId='a'; candidateId='b'; render=()=>{redraws++;};
`,context);
assert.equal(vm.runInContext('progressionValueAt([[0,0],[100,10],[200,20]],99)',context),0);
assert.equal(vm.runInContext('progressionValueAt([[0,0],[100,10],[200,20]],100)',context),10);
assert.equal(vm.runInContext('progressionValueAt([[0,0],[100,10],[200,20]],300)',context),20);
const markup=vm.runInContext("scoreProgressionSection('mean')",context);
assert(markup.includes('progress-overlay'));
assert(!markup.includes('Generated tokens only.'));
vm.runInContext('bindProgressionCharts()',context);
const event=x=>({clientX:x,clientY:120,button:0,pointerId:1});
overlay.listeners.pointermove(event(512));
assert(tip.innerHTML.includes('Baseline: <strong>10.00'));
assert(tip.innerHTML.includes('Candidate: <strong>5.00'));
assert(tip.innerHTML.includes('Difference: -5.00'));
assert.equal(cross.attributes.visibility,'visible');
overlay.listeners.pointermove(event(952));
assert(tip.innerHTML.includes('(finished)'));
overlay.listeners.pointerleave();
assert.equal(tip.style.display,'none');
assert.equal(overlay.listeners.pointerdown,undefined);
assert.equal(overlay.listeners.pointerup,undefined);
assert.equal(overlay.listeners.dblclick,undefined);
assert(!markup.includes('Reset zoom'));
assert(!markup.includes('progress-brush'));
overlay.listeners.pointercancel();
assert.equal(tip.style.display,'none');
let prevented=false;
wrap.listeners.keydown({key:'ArrowRight',preventDefault(){prevented=true;}});
assert(prevented && tip.innerHTML.includes('100 tokens'));
wrap.listeners.keydown({key:'Escape'});
assert.equal(tip.style.display,'none');
assert.equal(context.redraws,0);
console.log('Comparison hover, step values, endpoints, cancellation, and keyboard checks passed; zoom removed.');
