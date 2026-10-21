# Harness memory upgrade: harness-built facts, shown every turn as deltas (extends memory_ledger.py; base arm/stack-v2)
Decision. The data card's largest loss is evidence already visible but unused (9.0% exact (state,action) repeats, 97.8% of them still in window; 27% of actions on net-zero loops; 15% of
tokens in dead attempts; next level cheaper in 34%), not trim forgetting (1-2% of tokens). The model-written channel is nearly empty (2.7-3.9 tags/100 calls) and 60-87% of ledger entries
are harness fragments of thinking, cut at 200 chars, 61% starting with filler. So the HARNESS extracts facts from game data; model tags shrink to a "beliefs" section; delivery is small
deltas every turn plus the full card after a trim. Everything is flag-gated; all flags off = byte-identical to today.
## 0. Flags and files
ARC3_MEM_SOLVED (A) | ARC3_MEM_CARD (B) | ARC3_MEM_EFFECTS (C) | ARC3_MEM_SEEN (D) | ARC3_MEMORY_PLACEMENT=trim|delta|both (E; default trim = today's ledger-on-trim).
Memory is active if ARC3_LEDGER_ON_TRIM or any of A-D is set; then self._ledger = MemoryLedger() (tool_agent.py ctor ~L3563, new-game reset ~L3662 on stack-v2). Knobs: ARC3_MEM_STATE_BORDER=4,
ARC3_MEM_CARD_CHARS=3900, ARC3_MEM_DELTA_CHARS=480, ARC3_MEM_DELTA_GAME_CHARS=12000, ARC3_MEM_SEEN_GAME_CHARS=8000. Tokens = chars/3 (data card: 1,107 chars = 370 tokens).
NEW inference/agent/memory_facts.py (pure python, no I/O): state_hash, classify, ActionEffects, AttemptTracker, SeenStates, loop_free, rle_path, distill_think. EXTEND memory_ledger.py:
MemoryLedger gains a version counter, a harness-item store {id: (version, section, text)}, last_think, render_card(), render_delta(), mark_shown(). solver.py is untouched.
## 1. Observation pipeline (one feed, no new solver hooks)
ToolAgent._mem_observe(history_entries, valid_actions) runs in _build_user_prompt right after _death_ledger_observe (~L5153), unconditionally when memory is active (death-ledger lesson: a
recorder inside a display path stays empty when only the display flag is off). Cursor self._mem_cursor = entries consumed; reset on a new game. solver.py _execute_action appends one
HistoryEntry(action, frame, result) per executed action, batches included. For entry i>=1: pre = entries[i-1].frame, post = entry.frame, r = entry.result (gameplay_changed, board_changed,
game_over, level_completed, automatic, action_data, animation), sig = action_signature(entry.action), level = pre.level (the clearing action's post.level is already L+1).
Effect class, first match: clear (level_completed) > death (game_over) > noop (gameplay_changed False and board_changed False) > hud (gameplay_changed False, board_changed True: bar tick)
> anim (animation present, no interior change) > shift (interior colour histogram unchanged) > change (histogram changed: recolour, appear, disappear).
State identity for A/C/D = _state_hash(grid, border=ARC3_MEM_STATE_BORDER, default 4 = ARC3_NOOP_GUARD_BORDER, so it agrees with gameplay_changed). The crop is required: the death ledger
default (ARC3_STATE_IDENTITY_BORDER=0) hashes the shrinking budget bar, so a revisit never matches whenever a bar or counter ticks and loops would be invisible. Refactor
_interior_state_hash(grid) to call _state_hash(grid, _state_identity_border()) (no behaviour change). Our notes are advisory and block nothing, so the self-sealing risk behind the
border-0 default does not apply.
Attempt = actions since level start or the last RESET (automatic after game_over, or manual); closed by death, RESET or clear; keep per attempt [(sig, pre_h, post_h, cls)].
## 2. (A) Solved-level memory
On the entry with level_completed, close the attempt and write one record, never evicted:
 path = loop_free(winning attempt): walk the chain, when post_h equals an earlier chain hash (start included) cut back to it, drop noop/hud/anim; rle_path: U D L R S, clicks M(r,c).
 spent = all actions on the level incl. failed attempts; deaths = count; end = compute_frame_diff(segment_layer(level-start grid), segment_layer(last pre-clear grid)) via
 _render_auto_frame_diff(budget_tokens=30), first 2 entries joined by ";" <= 90 chars (omitted on shape change or any exception); how = distill_think(MemoryLedger.last_think) <= 160.
 Format, <= 165 chars/level (end/how text below is illustrative):
 <solved>L1: 13 acts [L3 U4 R3 U3], spent 40, 0 deaths | end: block moved (10,4)->(34,20) | how: take the key before the door</solved>
 Worked example (r2 ls20 L1 replay): 40 actions spent, 7 of them no-ops, loop-free path 13. Data card: 78.5% of final-attempt actions are needed (median 21), RLE path median 75 chars.
 distill_think replaces summarize_think here: split sentences, strip leading filler (so/then/hmm/but/ok/wait/now/let me), score +2 action name or click, +1 digit or coordinate, +1 finding
 keyword; take the best-scoring LAST whole sentence <= 160 chars; never cut mid-sentence, no "..."; none qualifies -> omit "how" (absent beats a fragment).
 last_think = raw reasoning of the last assistant message (note_think stores it) = the call that issued the clearing batch. A model <level_note> for that level appends " | note: ...".
 Over the card cap: from solved lines older than the newest 3 drop "end", then "how", then the path, leaving "L1-L4: 13/22/9/31 acts". Spent and deaths stay.
## 3. (C) Action-effect table
ActionEffects: Counter[(scope, key, cls)], scope = level L or game. key = action name; click key = "MOUSE@<letter>", letter = ARC_COLOR_CHARS[pre.grid[row][col]] (colour of what was clicked;
no segmentation needed). Per key also the median changed cells and the set of fatal / clearing board hashes. Rendered, counts "this level | game" (UP/RIGHT level counts are the r2 ls20 L1
replay; game counts and the SPACE row are illustrative):
 <actions>
 UP shift 13 noop 2 clear 1 | game shift 31 noop 6 clear 2
 RIGHT shift 6 noop 2 | game shift 12 noop 4 death 1 (fatal in 1 board)
 SPACE untested this level | game noop 2/2 (inert so far)
 </actions>
Keyboard: <= 6 rows of ~90 chars (~540). Click games (17 of 25; median 27 distinct targets): group by letter, <= 11 rows, ~1k chars, clear/death rows first, then by count:
 MOUSE@R 7 targets: change 5 noop 2, median 12 cells, e.g. (16,22) change, (38,38) noop | MOUSE@Y 2 targets: clear 1 (54,38) | MOUSE@G 3 targets: death 1 (46,54)
 untried colours: B P (on the board, never clicked this level)
Rules: omit zero classes; "inert" needs >= 3 noops on >= 2 distinct boards and 0 effects; deaths cite boards, never a cause (the harness does not guess budget vs hazard); valid_actions never
taken this level are listed untested. Counts live only in the full card; deltas (section 6) carry class changes, not increments.
## 4. (D) "You have been here before"
SeenStates, per level, LRU 512 boards x 16 actions: hash -> {visits [(attempt, action_index)] <= 4, tried {sig: (cls, next_hash, attempt)}}, fed by the same pre/post pairs.
Fires when hash(current_frame) has an EARLIER visit (not the start board after a RESET or level change; the last action's own noop/hud self-visit does not count). Worked example (r2 ls20 L1:
L L L U U U D U, the last U returns to the board after action 6):
 Seen before (board interior matches): after action #6 and #8 of this attempt. From it you tried: DOWN -> back to a seen board (loop). Untried: UP, LEFT, RIGHT.
Outcome words: new board | loop | noop | FATAL (attempt n) | CLEAR. <= 4 tried items then "+N more"; <= 240 chars; clicks as M(r,c). Descriptive wording, no "do not" (data: 56% of actions
after a loop closes repeat a tried action; 47% of actions from visited boards repeat one). With ARC3_DEATH_LEDGER lines on, FATAL items are omitted (already printed there).
Rides the same non-resume gate as deltas but is appended EVERY turn it applies (it describes the current board), so it ignores ARC3_MEMORY_PLACEMENT; it goes before <memory_update>. ~19% of
turns, ~60 tokens each.
## 5. (B) World-model card = harness facts + model beliefs
<actions> and <solved> are the mechanics layer of the world model; <world_model> tags are left for what only the model can infer (goal, rules, why). Card = the existing
<memory>...</memory> wrapper (so _LEDGER_BLOCK_RE and _with_ledger_head work unchanged; "</memory>" is stripped from every entry). Render order, also truncation-safe order: <solved> (A, L
ascending) | <actions> (C) | <hazard>L2: RIGHT from the board first reached at action 14 killed attempt 2 (31 acts)</hazard> (<= 3 newest, current level, <= 120 chars) | <world_model> |
<hypothesis status="confirmed"> | <hypothesis status="refuted"> (oldest first). Hard cap ARC3_MEM_CARD_CHARS=3900 (~1.3k tokens); soft caps solved 1300, actions 1000, hazard 200,
beliefs 1400 (data card: ~1.5k beliefs block). Over cap: drop oldest model entries, then compress solved (section 2); never refuted entries or hazards.
Model entry caps (only with ARC3_MEM_CARD): world_model 200 -> 400 chars (covers 73%; 88% were truncated), hypothesis 250, level_note 200; cut at a word boundary + "...". Count caps
unchanged (6 / 8 per status / 6). note_think fragments are no longer rendered; _pending_think keeps only "open:" hypotheses.
Dedup, newest wins. Harness items have stable ids (solved:L, actions, hazard:L:n): replace in place, bump version, never duplicate. Model items: normalise (lowercase, drop punctuation and
stop words); token Jaccard >= 0.6 vs the same list -> newest replaces and moves last; a confirmed/refuted pair at >= 0.6 removes the older (0 flips in 154 tags, cheap anyway); a tag
starting "Correction:", "Fix:" or "Actually," drops the prefix and supersedes the best match at Jaccard >= 0.35 across world_model + hypotheses (10 of 154 tags self-revise).
Hypothesis ledger, verdict: no separate model-written ledger (0.6-0.95 hypotheses per 100 calls, 2-4 refuted per run: nothing to ledger). Its objective half is built by the harness: C rows
with n/n effects are the "confirmed" side; inert rows and <hazard> lines the "refuted" side ("SPACE does something: 0/4"; "that RIGHT is safe: killed attempt 2"). Model <hypothesis> tags are
still absorbed and shown in beliefs; nothing auto-judges them.
Lost channel: in _dispatch_tool absorb memory tags found inside python `code` args and strip them before sandbox exec (42 tags lost in r1+r2, plus the SyntaxError case).
System prompt: one static paragraph MEMORY_CARD_ADDENDUM (cache-safe): <memory> lines are harness-observed facts (counts, solved paths, deaths) or your notes; do not re-record facts the
harness shows; write <world_model> only for what it cannot see.
## 6. (E) ARC3_MEMORY_PLACEMENT
trim: today's path. _note_history_evicted sets _ledger_due; _apply_due_ledger renders the card once, puts it at the head via _with_ledger_head, then mark_shown(version).
delta: in _build_user_message, non-resume openers only, add a final text part after the last image (string content: suffix):
 <memory_update>
 + solved L2: 21 acts [R3 U2 D4 ...], spent 52, 1 death | end: ... | how: ...
 + RIGHT: first death (attempt 2, after 31 acts); UP: first noop
 + world_model: ...   (tag absorbed since the last opener)
 </memory_update>
 Delta = items with version > shown_version, oldest first. C adds only game-first (key, cls) pairs and every death/clear, never counter increments. <= 480 chars per opener; overflow is
 carried; on the 3rd consecutive carry the cap doubles to flush. Empty delta = no text. Per-game cap 12000 chars, after which only head cards continue.
both (recommended; delta alone loses whatever lived only in dropped openers): head card at each trim + deltas every opener. Invariant: head card at version V plus all later deltas == the
 full card now. A trim landing on a request whose opener already holds a delta repeats a few lines (harmless).
Why not the literal full card every turn. Kept in history it is cache-safe but the data card prices a 370-token block at ~5k tokens/window (14 retained turns); our typical 0.9k-token card
 would be ~12k (~20% of the 59k window). Stripped from older messages it rewrites a message ahead of the reply that follows it, so the prefix diverges every turn and the card plus
 that reply (~2k tokens of retained thinking) are re-prefilled: ~+2-3k prefill/turn on ~9k new (estimate from data-card sizes, not measured; test on arm/serving-bench before trying it).
Cache safety. trim: the prefix is already invalid from the first retained message after a trim, so the head edit rides the same re-prefill; between trims nothing re-renders. delta and D:
 text exists only in the new suffix, is composed once and persisted verbatim in _history_messages. Only existing rewrites remain (trim, _apply_history_image_window, control pruning);
 image stripping keeps text parts, so deltas survive it. Resume openers (_mark_control_message "resume") get none; items stay pending for the next full opener. Rollback: analyze()
 snapshots (shown_version, pending queue) next to previous_history_messages and the abandoned-turn branch restores them, so a dropped opener never swallows a delta.
## 7. Level change
Carries (game scope): solved records, model beliefs, C game counters and inert/fatal verdicts, pending deltas. Resets at level start: this-level counters, SeenStates, attempts, <hazard>
(the count survives as "deaths" in the solved line), open hypotheses, "untested this level". The first opener of the new level carries the new <solved> delta, so L1's path is on screen
when L2 starts. note_think fragments no longer become level_notes when A is on. A game that ends mid-level writes nothing.
## 8. Interaction with the existing ledger (extend, do not duplicate)
absorb() keeps its signature and still strips/records tags from content, reasoning and (new) python code; render() is untouched when no new flag is set (the card is a second renderer,
render_card()); on_level_change() skips the note_think fold when A is on; note_think() additionally stores last_think. LEDGER_ON_TRIM_ADDENDUM stays for ledger-only runs; with B on,
MEMORY_CARD_ADDENDUM replaces it (never both). Valid actions come from _build_user_prompt's valid_actions argument (also self._current_valid_actions).
## 9. Budgets
Card <= 3900 chars (typical 1.5-2.6k), once per trim (~21 calls): ~0.9k tokens, ~1.5% of the ~59k post-trim window. Delta <= 480 chars (~160 tokens), typically 0-1 lines. D <= 240 chars.
Per game: head cards <= ~5k tokens, deltas <= 4k, D <= 2.7k, total <= ~12k, ~1% of ~90 calls x ~9k new tokens. No per-turn full re-render anywhere.
## 10. Tests (tests/test_memory_facts.py, tests/test_memory_placement.py; extend test_ledger_on_trim.py and the flags-off identity snapshot)
1 flags off: MemoryLedger.render(), _build_user_message and the system prompt are byte-identical; replayed r2 transcripts give the same ledger blocks.  2 classify: synthetic grids for
clear/death/noop/hud/anim/shift/change, keyboard and click keys.  3 loop_free + rle_path: fixture gives "L3 U4 R3 U3"; a ticking HUD bar does not split states at border 4 but does at 0.
4 distill_think: whole sentence, filler stripped, <= 160, omitted when nothing scores; solved line <= 165.  5 SeenStates: no fire on the start board after RESET, tried/untried lists,
FATAL suppressed with the death ledger on, <= 240.  6 delta: only newer items; C emits no increments; carry and flush; resume opener gets none.  7 property test: random event stream,
head card at V + later deltas == full card; serialize(messages_t) is a prefix of serialize(messages_t+1) in delta mode between trims.  8 rollback of an abandoned turn.  9 dedup: Jaccard 0.6
newest wins, flip removal, "Correction:" supersession, "</memory>" stripped, _LEDGER_BLOCK_RE still matches.  10 level-change matrix.  11 python-code tags absorbed and stripped.
12 caps: card <= 3900 and click rows <= 11 under adversarial input.  13 replay anchor on cached ev__*.json (ih hashes): revisit rate 14.2% +-0.5, exact repeats 9.0% +-0.5.
## 11. Arms and kill rule
kaggle/arms.json: mem-solved (A, both), mem-facts (B+C+D, both), mem-all, each against ledger-on-trim and control on matched seeds. Read-outs reuse the data-card scripts (m3_revisit,
m4_deaths, m5): exact repeats 9.0% -> <= 6%, loop actions 27%, no-op rate 5.9%, dead-attempt token share 15%, next level cheaper 34%. Score decides (control ~45-46, ledger 49-53); drop an
arm whose repeat rate does not fall and whose score gain over control is inside control noise. Risks: facts can anchor exploration (so "inert" needs 3 noops on 2 boards and the wording
stays descriptive); border-4 hashes can merge boards that differ in hidden state (wording says "board interior matches"); the ledger gain is not explained by trim recovery, so expect A to
move little.