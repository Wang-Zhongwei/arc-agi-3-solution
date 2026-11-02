export const meta = {
  name: 'noise-rule-design',
  description: 'Four independent rule designers search for regex filters for useless confirmed-rule lines on v02; a critic attacks the survivors',
  phases: [{ title: 'Design', detail: 'one designer per lens, v02 labels only' }, { title: 'Critique', detail: 'attack overfit and lost useful lines' }],
}
const ROOT = '/Users/wang/.claude/jobs/910e124b/tmp'
const SHARED = `Context: an ARC-AGI-3 game agent keeps a pinned section of "Confirmed rules": sentences from its own thinking containing the word "confirmed", harvested when old turns are trimmed. A frozen filter (${ROOT}/scripts/rule_v8.py) already removes hedged, agitated, generic and impossibility lines. Lines it keeps were audited by readers, each labelled useful, bad (wrong or misleading), or noise (true or harmless but useless to a reader). Lines have two display phases: 'current' (the level is still being played; reader is on that level) and 'past' (the level was cleared; reader is on a later level).

Data: ${ROOT}/verify/noise/dev_lines_v02.json has {current:[...],past:[...]} with id, game, level, label, info_type, usefulness, verdict, reader_reason, text, next1. Read it in full before designing.
Scorer: python3 ${ROOT}/scripts/noise_harness.py <your_candidate.py> [--list]. Read the scorer first. Candidate module defines CANDIDATES = {name: (phase, fn)} with phase current|past|both; fn(line) returns True to drop. line has ONLY: text, next1, level, prior (earlier kept lines of the same game, each with text and level). Using id, game, label, or anything else is forbidden: the rule must run live in the agent.
Acceptance bar, fixed in advance, per phase: at least 4 lines caught, at least 3 games, at least 70% of caught lines noise or bad.

Hard rules:
- Do NOT read anything under ${ROOT}/verify/v01/ and do NOT open any run under .../main-tuned-drafter/v01. That run is the held-out test and must remain unseen.
- Write only to ${ROOT}/verify/noise/. Do not modify the repo, rule_v8.py or the scorer. Do not use git.
- Rules must be general language patterns. Never encode a game name, a colour, a specific object word from one game (like "plug" or "peg"), or a specific coordinate. Never tune a regex by adding words seen only in a caught line and absent from the concept. A rule with more than about 12 alternatives is suspect.
- Before finalising, read the USEFUL lines each candidate catches (--list) and say honestly whether the rule costs too much. Report failures too: a concept you tried that did not separate from useful lines is a finding.
Be precise and report exact counts from the scorer, not estimates.`
const LENSES = [
  { key: 'snapshot', brief: 'Lens: momentary state snapshots. Lines reporting where things are or what value something has at one instant (positions, counts, colours at a step), which go stale and which the live board already shows. Coordinates alone do not separate these from useful lines: many useful lines have coordinates because they state a cause and effect. Find what does separate them, such as a statement with no action or cause, a check against a prediction, or an observation tied to a step.' },
  { key: 'fragment', brief: 'Lens: fragments and private shorthand. Headings that lost their content, lines whose subject was never stated, undefined labels and single letters, one-off notation, and bare references to something gone ("confirmed earlier", "level 2 confirmed it", "as predicted"). Find what can be recognised from the text alone: unresolved references, symbol density, sentence structure, absence of a verb.' },
  { key: 'plan_progress', brief: 'Lens: plan and progress lines. Steps of a lost plan, branch alternatives, records that one step or check happened, statements of what was just done. Also look at lines that verify the agent\'s own model rather than stating what the model says.' },
  { key: 'past_stale_redundant', brief: 'Lens: staleness after the level is cleared, and redundancy. For the past phase, level-specific facts are useless to a reader on a later level while game mechanics stay useful; find language patterns separating the two (the past phase has the highest junk rate). Also try redundancy using line["prior"]: a line repeating an earlier kept line. Earlier word-overlap dedup attempts were unreliable, so measure rather than assume.' },
]
const SCHEMA = { type: 'object', required: ['file', 'candidates', 'failed_concepts', 'noise_description'], properties: {
  file: { type: 'string' },
  noise_description: { type: 'string', description: 'What this kind of useless line is, with 2-3 quoted examples' },
  candidates: { type: 'array', items: { type: 'object', required: ['name', 'phase', 'n', 'useful', 'bad', 'noise', 'games', 'junk_pct', 'passes', 'what_it_catches', 'useful_lost_assessment'], properties: {
    name: { type: 'string' }, phase: { type: 'string' }, n: { type: 'integer' }, useful: { type: 'integer' }, bad: { type: 'integer' }, noise: { type: 'integer' }, games: { type: 'integer' }, junk_pct: { type: 'number' }, passes: { type: 'boolean' }, what_it_catches: { type: 'string' }, useful_lost_assessment: { type: 'string' } } } },
  failed_concepts: { type: 'array', items: { type: 'string' } },
} }
phase('Design')
const designs = (await parallel(LENSES.map(l => () => agent(`${SHARED}\n\n${l.brief}\n\nWrite your candidates to ${ROOT}/verify/noise/cand_${l.key}.py, score them, iterate at most a few rounds, and return the final scorer counts.`, { label: `design:${l.key}`, phase: 'Design', schema: SCHEMA })))).filter(Boolean)
log(`${designs.length} designers returned ${designs.reduce((a, d) => a + d.candidates.length, 0)} candidates`)
phase('Critique')
const critique = await agent(`${SHARED}\n\nYou are the critic. Four designers produced candidates in ${ROOT}/verify/noise/cand_*.py (ignore cand_example.py). Their summaries:\n${JSON.stringify(designs, null, 1)}\n\nFor every candidate that passes the bar, in each phase: read the code, run the scorer with --list, and try to break it. Look for overfitting (patterns that only make sense for these lines; would it catch a good rule stated in slightly different words? invent 5 plausible useful rule sentences across different game styles and test them against the code by running it), for useful lines caught that are actually the most valuable kind, and for overlap between candidates. Then write ${ROOT}/verify/noise/cand_shortlist.py, a module whose CANDIDATES holds only the survivors, each simple and general, simplified where you can without lowering the score; score it. Say which concepts are NOT filterable by rule and why.`, { label: 'critic', phase: 'Critique', schema: { type: 'object', required: ['survivors', 'rejected', 'not_filterable', 'adversarial_tests'], properties: {
  survivors: { type: 'array', items: { type: 'object', required: ['name', 'phase', 'n', 'useful', 'bad', 'noise', 'games', 'junk_pct', 'why'], properties: { name: { type: 'string' }, phase: { type: 'string' }, n: { type: 'integer' }, useful: { type: 'integer' }, bad: { type: 'integer' }, noise: { type: 'integer' }, games: { type: 'integer' }, junk_pct: { type: 'number' }, why: { type: 'string' } } } },
  rejected: { type: 'array', items: { type: 'object', required: ['name', 'why'], properties: { name: { type: 'string' }, why: { type: 'string' } } } },
  not_filterable: { type: 'array', items: { type: 'string' } },
  adversarial_tests: { type: 'string' } } } })
return { designs, critique }