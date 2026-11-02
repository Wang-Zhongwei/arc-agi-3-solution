export const meta = {
  name: 'cold-read-rules-section',
  description: 'Cold readers judge whether each pinned confirmed/ruled-out line is meaningful without its trimmed context',
  phases: [{ title: 'Cold read', detail: 'two readers per game: one placed at a later level, one at the same level as each line' }],
}
const FILE = '/Users/wang/.claude/jobs/910e124b/tmp/verify/cold_args.json'
const LINE = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    comprehensible: { type: 'string', enum: ['clear', 'partial', 'unclear'] },
    paraphrase: { type: 'string' },
    missing_referents: { type: 'string' },
    info_type: { type: 'string', enum: ['game_mechanic', 'level_fact', 'state_snapshot', 'plan_or_progress', 'empty_or_meta'] },
    actionable: { type: 'string', enum: ['yes', 'maybe', 'no'] },
    meaningful: { type: 'boolean' },
    reason: { type: 'string' },
  },
  required: ['id', 'comprehensible', 'paraphrase', 'missing_referents', 'info_type', 'actionable', 'meaningful', 'reason'],
}
const OUT = { type: 'object', properties: { lines: { type: 'array', items: LINE } }, required: ['lines'] }

const COMMON = `You are an AI agent playing an ARC-AGI-3 game: a turn-based puzzle on a 64x64 grid of 16 colours with several levels. Actions are UP/DOWN/LEFT/RIGHT, ACTION5 (often "space"/interact), clicks at (x,y), and RESET. Rules are never stated; you learn them by experimenting. Your earlier turns have been trimmed from your context, and the harness pinned the section below: sentences from your own earlier reasoning where you wrote that something was "confirmed" or "ruled out", grouped by the level on which you wrote them. You see the CURRENT board each turn, but nothing else from the trimmed turns.

Judge each line ONLY from what is on the page: do not look up files or transcripts. The point is to test whether the line carries meaning without its lost context.

For every line id give:
- comprehensible: clear (you know what it asserts and about which objects), partial (the gist is clear but key referents are undefined), unclear (you cannot tell what it asserts).
- paraphrase: what you take it to assert, in plain words ("?" if unclear).
- missing_referents: names, labels, coordinates or "the plan/model/map" that the line uses but never defines ("" if none).
- info_type: game_mechanic (how objects/actions behave, likely to hold on other levels), level_fact (layout/goal/solution detail for that level), state_snapshot (positions/counts at a moment, stale after the next moves), plan_or_progress (what you were doing or had achieved), empty_or_meta (says only that something was confirmed, with no usable content).
- actionable: would reading it change what you do or try (yes/maybe/no)?
- meaningful: true only if the line is comprehensible enough AND carries information that could still help you at the position described below.
- reason: one short sentence.
Return exactly one entry per line id, using the ids in square brackets.`

const LENS = {
  later: 'YOUR POSITION: you are now on a LATER level than every line in the section (one level past the highest level shown). Lines are useful to you only if they still hold or transfer.',
  same: 'YOUR POSITION: for each line, imagine you are still on THAT line\'s level, a while after it was written (its turn has been trimmed). Level facts can still help you there; stale snapshots usually cannot.',
}
function load(game) {
  return `LOAD THE SECTION FIRST:\npython3 -c "import json;a=[x for x in json.load(open('${FILE}')) if x['game']=='${game}'][0];print(a['section'])"\nOnly use that command; do not open any other file.`
}
const res = await pipeline(args, item => parallel(['later', 'same'].map(k => () =>
  agent(COMMON + '\n\n' + LENS[k] + '\n\n' + load(item.game), { label: k + ':' + item.game, phase: 'Cold read', schema: OUT, effort: 'medium' })
    .then(r => ({ lens: k, lines: (r && r.lines) || [] })))),
  (pair, item) => item.ids.map(id => {
    const o = { id, game: item.game }
    for (const p of pair.filter(Boolean)) o[p.lens] = p.lines.find(x => x.id === id) || null
    return o
  }))
const flat = res.filter(Boolean).flat()
log(`cold-read ${flat.length} lines; missing later ${flat.filter(x => !x.later).length}, missing same ${flat.filter(x => !x.same).length}`)
return { lines: flat }
