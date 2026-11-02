export const meta = {
  name: 'verify-confirmed-refuted-claims',
  description: 'Verify every confirmed/refuted claim of main-tuned-drafter v02 plus its following sentence, two lenses + judge',
  phases: [
    { title: 'Verify', detail: 'two independent verifiers per batch: in-run evidence and cross-run evidence' },
    { title: 'Judge', detail: 'resolve fields where the two verifiers disagree' },
  ],
}

const FILE = '/Users/wang/.claude/jobs/910e124b/tmp/verify/batches_v02.json'
const VERDICTS = ['correct', 'partially_correct', 'incorrect', 'not_a_claim', 'unverifiable']
const ROLES = ['consequence', 'elaboration', 'doubt_about_claim', 'moves_on', 'none']
const CONSEQ = ['correct', 'partially_correct', 'incorrect', 'unverifiable', 'na']
const DOUBT = ['claim_was_wrong', 'claim_stood', 'unresolved', 'na']
const USE = ['game_rule', 'level_rule', 'momentary_detail', 'vague', 'misleading']

const ASSESS = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    claim_verdict: { type: 'string', enum: VERDICTS },
    next_role: { type: 'string', enum: ROLES },
    consequence_verdict: { type: 'string', enum: CONSEQ },
    doubt_outcome: { type: 'string', enum: DOUBT },
    usefulness: { type: 'string', enum: USE },
    confidence: { type: 'number' },
    evidence: { type: 'string' },
  },
  required: ['id', 'claim_verdict', 'next_role', 'consequence_verdict', 'doubt_outcome', 'usefulness', 'confidence', 'evidence'],
}
const BATCH_SCHEMA = {
  type: 'object',
  properties: { assessments: { type: 'array', items: ASSESS } },
  required: ['assessments'],
}

const COMMON = `You are verifying claims an AI agent made in its private reasoning while playing an ARC-AGI-3 game (a turn-based 64x64 grid puzzle with several levels; actions are moves like UP/DOWN/LEFT/RIGHT, ACTION5, or clicks at (x,y)). Run: Kaggle notebook main-tuned-drafter v02.

Each claim is one sentence from the model's THINKING where it said something was "confirmed" (kind=confirmed) or that a hypothesis was "refuted / ruled out / disproved / falsified" (kind=refuted). in_section=true means the harness would carry the sentence (shown = the text it would show) across context trims; in_section=false (refuted only) means one of the harness filters dropped it, with dropped_by naming the filter. We need to know whether each claim is TRUE about the game, and how the sentence right after it relates to it.

Files (paths contain spaces: always quote them):
- context.this_run_transcript: sections start with lines like "--- analysis_step=N | action=M | time | tool-agent ---", "[USER PROMPT]" (contains "Current state: step S, level L." and the current board), "[THINKING]", "[TOOL CALL: python]" (the model's code), "[TOOL RESULT: python]" (what actually happened: action effects, printed frames, errors), "[ANALYZER STATUS]". Each claim's thinking_line is the line where its [THINKING] block starts.
- context.this_run_events: JSONL, one record per action with keys level, action_num, analysis_step, board_ascii (the real board after the action), action_display.
- context.this_run_levels_completed / this_run_actions_per_level: the run's outcome for this game.
- Other runs of the same game: context.best_other_runs (transcripts sorted by levels_completed) and the full index file context.all_other_runs_index (JSON: game -> list of {run, levels_completed, transcript}). other_runs_clearing_level[L] = how many other runs cleared level L.
The transcripts are 1-3 MB. Never print whole files: use grep -n -i, sed -n 'A,Bp', or short python one-liners.

For EACH claim decide:
1. claim_verdict: is the claim TRUE about the game at the level it was made? For kind=refuted: is the refuted hypothesis really FALSE?
   - correct: evidence supports it.
   - partially_correct: the core is right but a stated detail is wrong, or it overgeneralizes.
   - incorrect: evidence contradicts it.
   - not_a_claim: the sentence does not actually assert a settled confirmation/refutation (a plan, question, conditional, quoted hypothesis, or a verdict word with no identifiable object).
   - unverifiable: you searched and found no evidence either way.
   Evidence order: real game behaviour (TOOL RESULT action effects, board_ascii frames, deaths, level-ups) > what runs that cleared that level relied on or observed > this model's later statements backed by new observations. The model calling something "confirmed" is NOT evidence. A later reversal by the model counts only when an observation backs it.
2. next_role: what next1 (the sentence right after the claim; next2 is the one after that, for context) does relative to THIS claim: consequence (states an implication or inference drawn from the claim), elaboration (adds detail, evidence or list items for the same claim), doubt_about_claim (questions, qualifies or contradicts this claim), moves_on (switches to planning, acting or another topic without building on the claim), none (no next sentence). The given next_class is only a keyword heuristic: judge the meaning.
3. consequence_verdict: only when next_role is consequence or elaboration (else na): is what next1 states TRUE about the game? correct / partially_correct / incorrect / unverifiable.
4. doubt_outcome: only when next_role is doubt_about_claim (else na): claim_was_wrong (the doubt was justified), claim_stood (the claim held up), unresolved.
5. usefulness: if this line were shown to the model later, after the context that produced it was trimmed: game_rule (a general mechanic that stays true across levels), level_rule (true and useful, but only for that level), momentary_detail (a coordinate/state/step fact that goes stale within the level), vague (not understandable without the trimmed context), misleading (wrong, or would push toward a wrong belief).
Also give evidence (a short concrete citation: file + line, or step/action number, and what it shows) and confidence in [0,1] for the claim_verdict.

Be skeptical and concrete; a few precise greps beat broad reading. Return exactly one assessment per claim id in the batch, none skipped, using the ids given.`

const LENS_A = `YOUR LENS: IN-RUN EVIDENCE. For each claim start at its thinking_line, read the reasoning around it, then follow THIS run forward: the TOOL RESULTs and board_ascii frames after the claim, later statements about the same mechanic, and whether the level was later cleared with or without relying on the claim. Use other runs only as a tiebreaker.`
const LENS_B = `YOUR LENS: CROSS-RUN EVIDENCE. For each claim first read its thinking_line context in this run to understand it, then check it against OTHER runs of the same game, especially runs that cleared that level (best_other_runs; grep their transcripts for the same objects, colours and mechanic words, and for "level L" with L the claim's level). Then check this run's own TOOL RESULT / board evidence. A rule that clearing runs relied on is strong support; a run that cleared the level while observing the opposite is strong contradiction.`

const FIELDS = ['claim_verdict', 'next_role', 'consequence_verdict', 'doubt_outcome']

function loader(batchId) {
  return `LOAD YOUR BATCH FIRST (it holds the context paths and the claims):\npython3 -c "import json;b=[x for x in json.load(open('${FILE}')) if x['batch_id']=='${batchId}'][0];print(json.dumps(b,indent=1,ensure_ascii=False))"\nYour batch is ${batchId}.`
}

const results = await pipeline(
  args,
  item => parallel([
    () => agent(COMMON + '\n\n' + LENS_A + '\n\n' + loader(item.batch_id), { label: 'in-run:' + item.batch_id, phase: 'Verify', schema: BATCH_SCHEMA }),
    () => agent(COMMON + '\n\n' + LENS_B + '\n\n' + loader(item.batch_id), { label: 'cross-run:' + item.batch_id, phase: 'Verify', schema: BATCH_SCHEMA }),
  ]),
  async (pair, item) => {
    const A = new Map((((pair && pair[0]) || {}).assessments || []).map(x => [x.id, x]))
    const B = new Map((((pair && pair[1]) || {}).assessments || []).map(x => [x.id, x]))
    const disputed = item.ids.filter(id => {
      const a = A.get(id), b = B.get(id)
      return !a || !b || FIELDS.some(f => a[f] !== b[f])
    })
    let J = new Map()
    if (disputed.length) {
      const both = disputed.map(id => ({ id, in_run_verifier: A.get(id) || null, cross_run_verifier: B.get(id) || null }))
      const judged = await agent(
        COMMON + '\n\nYOU ARE THE JUDGE. Two independent verifiers (one using in-run evidence, one using cross-run evidence) assessed the claims below and disagreed on at least one of claim_verdict, next_role, consequence_verdict, doubt_outcome (or one of them is missing). Check the evidence each cites and look yourself where needed, then return the final assessment (all fields) for exactly these claim ids: ' + JSON.stringify(disputed) + '. Do not split the difference: pick what the evidence shows.\n\n' + loader(item.batch_id) + '\n\nverifier assessments = ' + JSON.stringify(both, null, 1),
        { label: 'judge:' + item.batch_id, phase: 'Judge', schema: BATCH_SCHEMA })
      J = new Map(((judged && judged.assessments) || []).map(x => [x.id, x]))
    }
    return item.ids.map(id => {
      const a = A.get(id) || null, b = B.get(id) || null, j = J.get(id) || null
      return { id, batch: item.batch_id, disputed: disputed.includes(id), in_run: a, cross_run: b, judge: j, final: j || (a && b ? a : (a || b)) }
    })
  },
)
const flat = results.filter(Boolean).flat()
log(`verified ${flat.length} claims; ${flat.filter(r => r.disputed).length} disputed; ${flat.filter(r => !r.final).length} without any assessment`)
return { claims: flat }
