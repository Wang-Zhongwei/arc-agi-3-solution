"""Prompt templates for the analyzer agent."""

from inference.utils.grid_utils import ARC_COLOR_LEGEND

TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)

GAME_OVERVIEW_ADDENDUM = (
    "\n\nGame overview:\n"
    "- You are solving a multi-level grid puzzle game. \n"
    "- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle: re-understand the current state from the newest frame, update your working world model in Python, choose the next best action or short sequence against the goal as currently understood, execute it, and expect to re-evaluate on the next turn from the updated state.\n"
    "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
    "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n"
    "- Optimize for as few in-game actions as possible while still being reliable.\n"
    "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

VISUAL_GAME_ADDENDUM = (
    "\n\nVisual-game guidance:\n"
    "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
    "- Game entities are usually be rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometime they might also be 1x1 tokens."
    "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
    "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
    "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
    "- `RESET` restarts the current level from its initial state (completed levels stay completed) and throws away the effect of every action since the level start. "
    "It is a last resort for a wrecked state, not a way to 'start clean' or to replay an opening: the state after your last actions is still there, keep working from it or reverse the last move.\n"
)

STRUCTURED_RUNTIME_STATE_ADDENDUM = (
    "\n\nRuntime variables inside every `python` tool call:\n"
    "- `current_frame` is a lightweight frame view for the latest environment state.\n"
    "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n"
    "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores its position -- equal hashes mean the same object regardless of where it is, so use it to track an object across frames or to spot multiple identical objects in one frame), `pixels` (cell count), `bbox` (`[r0, c0, r1, c1]`, inclusive), `centre` (`[row, col]`), `height`, `width`, `boundary` (clockwise outer-perimeter corner points), and `children` (ids of objects fully enclosed by this one).\n"
    "- `action([...])` returns, besides the batch summary, `trace` (one entry per executed action: `action`, `changed_cells`, `diff`, `score`, `level`, and `static_rows_changed` when a row that had never changed on this level, such as a legend or HUD, changed) and `progress` (`score_delta`, `level_delta`, `static_rows_changed`, `goal_advanced`). Read `progress` before believing anything was collected or completed.\n"
    "- `objects(color=None, min_pixels=1)` returns that list without the background, largest first, as compact dicts (id, color, pixels, bbox, centre, height, width); `board_summary()` prints one line per object (grouped: a bordered piece with its interior, or a dotted/checkerboard trail, count as ONE object), or a per-colour census when there are many; `objects(composite=True)` gives the grouped list, `board_rle()` the whole board as run-length rows, `scene_graph()` an object-centric view (objects as nodes, relations as arrows: inside, overlaps, touches, nearest neighbour left/right/above/below with the gap), `frame_diff()` the object-level diff of the last action (moved with displacement, appeared, disappeared, changed cells; pass two frames to diff any pair). Prefer these to rebuilding coordinates from `boundary`.\n"
    "- Symbolic planning: once rules are verified, write them as a small PDDL domain and the situation as a PDDL problem (STRIPS: predicates; actions with :parameters, :precondition, :effect using and/not; :objects, :init, :goal) and call `pddl_plan(domain, problem)`: it returns the action sequence, [] if the goal already holds, or None. Ground every predicate in measured facts; a plan is only as good as the rules.\n"
    "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
    
    "- `current_frame.step` is the current environment step count.\n"
    "- `current_frame.level` is the current level number.\n"
    "- `current_frame.shape` is a `(rows, cols)` tuple.\n"
    "- The raw numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object hashes. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n"

    "- `history` is a chronological list of action/frame snapshots.\n"
    "- `history` is a Python list of objects, not a dict.\n"
    "- Each history entry exposes only `.action` and `.frame`; entries are not subscriptable like `entry['action']`.\n"
    "- Each `history[i].frame` is the frame after `history[i].action`; each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n"
    "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
    "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
    "- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n"
    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`.\n"
    "- `last_transition` is `transitions[-1]` or `None`. Its `.result` mirrors `last_action_result`; older transitions may have an empty `.result`. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n"
    "- `last_action_result` is the persisted result dict from the most recent `action(...)` call. It remains available across later Python inspection calls that do not call `action(...)`, and is `{}` before any action result exists. Read transition metadata from fields/keys such as `last_action_result['board_changed']`, `last_action_result['done']`, `last_action_result['level_completed']`, `last_action_result['game_over']`, `last_action_result['run_complete']`, `last_action_result['reward']`, and `last_action_result['valid_actions']`.\n"
    "- `valid_actions` is the current list of valid action names.\n"
    "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
    "- Pass `action(actions)` a list like `['LEFT']`, `['RESET']`, or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
    "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_result` are refreshed.\n"
)

MULTIMODAL_CONTEXT_ADDENDUM = (
    "\n\nMultimodal context:\n"
    "- Some user turns include one attached image that tiles several recent ARC grids side by side: oldest on the left, newest on the right. Each tile is labeled with its step number and the action that produced it; the rightmost tile is always the current board.\n"
    "- The rightmost tile and `current_frame.ascii` are two representations of the same current frame; earlier tiles match `previous_frame` / `history` entries with the same step numbers.\n"
    "- Images are batched to save tokens: a turn may say the board is unchanged (no new image) and older turns keep only a text placeholder. Use the newest image for perception and the Python tools for exact coordinates.\n"
    "- Read the image for gestalt structure (objects, layout, motion between tiles) and confirm details with `current_frame.segmentation` before acting.\n"
)

HYPOTHESIS_CRITIC_SYSTEM_PROMPT = (
    "You are the referee reviewing a friend's notes on an unfamiliar arcade game (ARC-AGI-3: a grid puzzle with several levels). "
    "The friend has been stuck on one level for a while and is acting on a written world model. You did not play; you have their notes, "
    "the harness's measured action effects, the action log and the board. Check each note against the measurements the way a referee "
    "checks a claim against the replay, then say what survives.\n"
    "\nKEY RULES (READ FIRST):\n"
    "- MEASUREMENTS BEAT NOTES. A measured transition such as 'RIGHT: R 4x4 moved (+0,+6) [3 presses]' outranks anything the notes say. "
    "Quote the measurement you used for every verdict.\n"
    "- TEST EVERY CLAIM. Each numbered claim gets exactly one verdict, SUPPORTED, CONTRADICTED or UNTESTED. Never skip a claim, never judge two at once.\n"
    "- BE ADVERSARIAL TO THE GOAL. The commonest failure is a wrong goal that is never questioned. Ask what else explains the same evidence; "
    "a goal claim with no measurement behind it is UNTESTED, not SUPPORTED.\n"
    "- NO NEW STORIES. Corrected rules may only contain what a measurement supports; an unexplained change becomes an UNTESTED question, not a new rule.\n"
    "- Actions that never change the board, long repeated runs and repeated GAME_OVER resets mean the mechanics or the goal are misread. "
    "A strip flush against a border that ticks on every action is a timer, not a puzzle object.\n"
    "- Be concrete: name objects by color and location, name actions, give MOUSE targets as row/col. Under 350 words, only the requested labeled sections.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

# Appended to the referee prompt in audit mode: one answer in the exact output format, on a made-up game.
HYPOTHESIS_AUDIT_EXAMPLE = (
    "\nWORKED EXAMPLE (made-up game, format only; derive your own verdicts from the evidence you are given):\n"
    "Claims:\n"
    "1: SUPPORTED by 'RIGHT: R 4x4 moved (+0,+6) [3 presses]'\n"
    "2: CONTRADICTED by 'UP: no change [4 presses]' (the notes say UP lifts the block)\n"
    "3: UNTESTED; cheapest test: DOWN x1, then read the diff of the blue 4x4\n"
    "4: UNTESTED; the goal 'fill the four slots' has no measurement behind it; a slot has never changed\n"
    "Verdict: REVISE\n"
    "Corrected mechanics: 1. RIGHT moves the red 4x4 six columns right ('RIGHT: R 4x4 moved (+0,+6) [3 presses]'). "
    "2. UP does nothing on this level ('UP: no change [4 presses]'). 3. The blue 4x4 has never moved; open.\n"
    "Discriminating test: DOWN x1; if the blue 4x4 moves, claim 3 becomes a rule, otherwise MOUSE(r20,c30) on it next.\n"
    "Stop doing: pressing UP; it produced no change in 4 presses.\n"
)

PYTHON_ADDENDUM = (
    "\n\nPython tool guidance:\n"
    "- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object hashes.\n"
    "- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n"
    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n"
    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n"
    "- The only tool is `python`; call it with one ephemeral `code` string.\n"
    "- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
    "- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `history[-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n"
    "- Maintain a compact working world model: what entities or regions exist, what actions seem to do, what the goal likely is, what remains uncertain, and what plan best fits the evidence so far.\n"
    "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described by your world model. If confidence is low, program a discriminating probe and revise the world model from the result.\n"
    "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
    "- Inspect current and history frames from Python instead of describing frames freehand.\n"
    "- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n"
    "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable\n"
    "- A strong default loop is: summarize the board, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n"
    "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
    "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
    "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
    "- Call `action(...)` inside Python rather than returning action text in the chat.\n"
    "- `action(...)` accepts an ordered list of one or more actions. Once your code has selected a reliable sequence, it is often useful to batch it.\n"
    "- You can also call `action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
)

COMPACT_TOOL_SESSION_ADDENDUM = (
    "\n\nTool session rules:\n"
    "- You have exactly one tool: `python`.\n"
    f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n"
    "- One inspection call at most before acting: an action is cheap and tells you more than another look; a turn that ends without `action(...)` is the expensive thing.\n"
    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_result`.\n"
    "- Each `python` tool call has a hard time limit of 30 seconds.\n"
    "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
)

HARNESS_SERVICES_ADDENDUM = (
    "\n\nKEY RULES (READ FIRST):\n"
    "- PROBE TABLE: at the start of every level the harness presses each button once and hands you a measured table. INTERPRET it, do NOT repeat it. A button's effect can depend on the state it is pressed in, so the table is evidence, not a rule.\n"
    "- RESET: restarts the current level and DISCARDS all progress on it. LAST RESORT only; NEVER a way to replay an opening.\n"
    "- PLAY IN LEGS: a leg of a few actions, a measured check that the board is where the plan says, then the next leg. NEVER send a whole solution blind; a wrong rule early wastes every action after it.\n"
    "- ACT EVERY TURN: every response ends with a `python` tool call that acts; inspection-only turns are wasted time. For MOUSE pass integer `row` and `col`.\n"
    "\nHow each turn works:\n"
    "- A user turn carries only what changed: the outcome of your last actions, the step and level, the valid actions, and anything the harness measured. The rules above do not repeat.\n"
    "- Functions and imports you define in a `python` snippet persist for the rest of the game (redefine to update); plain variables do not. Build small helpers early (state summary, diff, object finder) and reuse them.\n"
    "- `checkpoint(name)` records the current board for free; `restore(name)` brings the game back to it with real, scored actions by the cheapest route it can verify (ACTION7 presses when available and close, else RESET plus the level's actions with no-ops dropped and LEFT/RIGHT, UP/DOWN pairs cancelled; exact=True replays as played). Use it to try a risky line and come back, instead of a blind RESET.\n"
    "- `remember(key=value, ...)` stores short facts that are shown in every later prompt and survive history trimming; use it for rules and measurements you would hate to lose. `facts` holds the current dict.\n"
    "- Every prompt opens with 'What just happened', the harness's measurement of your last actions: per-action object changes, whether the board reverted to an earlier state, and whether a button behaved differently from before. Read it first. If it is not what you expected, write one line saying what you expected and why it differed, and update your model before acting; the most recent transition outranks any older belief.\n"
    "- Before acting you may restate what changed in your working model in one or two short lines of assistant text (`World model:`, `Goal model:`, `Mechanics:` (numbered rules of interaction, each with the transition that showed it), `Plan:` prefixes are picked up and carried forward); keep it brief.\n"
    "- Prefer measurement sweeps in one snippet (loop over a batch of actions and print a compact state after each) over one action per turn. Stop a loop immediately on `game_over`, `run_complete`, `level_completed`, or `done`.\n"
    "- `action(batch)` runs one action at a time and halts early on surprises: two consecutive actions that change nothing, the board returning to a state already seen in the batch, GAME_OVER, or level completion; a batch is capped at 12 actions (`stop_reason` / `stop_detail` tell you why it stopped). "
    "\nOptional tools:\n"
    "- Optional but powerful: define `predict(lines, action) -> lines` (your model of one action on the board, `lines` = `frame.ascii.split('\\n')`). The harness scores it against every real transition and reports per-action accuracy, and `plan(goal, actions=..., max_depth=...)` runs a breadth-first search over it (goal is a function of lines). Use search when your model is verified and the goal is known.\n"
)

TERSE_STYLE_ADDENDUM = (
    "\n\nStyle:\n"
    "- Be terse. Assistant text: telegraphic fragments, no filler, no restating the board, at most 3 short lines before the tool call.\n"
    "- Think in the same way: short bullet-like steps, decide, act. Measurements come from Python, not from prose.\n"
)

GAME_PRIORS_ADDENDUM = (
    "\n\nAction spec (ARC-AGI-3, docs.arcprize.org/actions; the same in every game):\n"
    "- RESET: initialises or restarts the current level state.\n"
    "- UP, DOWN, LEFT, RIGHT (ACTION1-4): simple actions semantically mapped to up/down/left/right; WHAT moves or steers varies by game (a piece, a cursor, a selection, an arm, often several cells per press).\n"
    "- SPACE (ACTION5): simple action that varies by game (interact, select, rotate, attach/detach, execute, ...). Measure it in two different states before trusting a reading; it may act only in contact or with something held.\n"
    "- MOUSE (ACTION6): click at a cell, `row` and `col` in 0-63. It often selects or toggles the cell under it and is inert in many games.\n"
    "- ACTION7: undo. It steps the game state back one action (some games charge budget for it). Never expect a forward effect from it.\n"
    "- A game exposes only the subset it uses (the valid-actions line). After GAME_OVER only RESET is valid.\n"
    "\nGoal first:\n"
    "- THIS GAME WAS MADE FOR HUMANS. There is ALWAYS a clear indicator of the goal on screen, the kind a gamer spots in seconds: a legend or icon row showing what to collect and in what order, a target shape or pad, a counter, a lock and key, a highlighted slot. Find that indicator before anything else; if you cannot name it, you have not looked at the right part of the screen yet.\n"
    "- On the first turn of every level, before any probe, read the HUD (legend rows, icon order, counters, bars) and state the objective in one line: what must happen, to what, in what order. A static row of icons is usually the target order; repeated small tokens are collectibles or slots; a strip that changes every step is a budget.\n"
    "- Probe only what the goal needs, and judge every probe by whether the goal advanced (score, level, legend or HUD change), not by whether something moved. Once the needed mechanics are pinned, execute.\n"
    "- An action that produced no change in one state may work in another (alignment, contact, a held object, the order the goal demands). Do not call it inert until it failed in two different states.\n"
)

DECISIVENESS_ADDENDUM = (
    "\n\nPlay decisively:\n"
    "- Commit first, then justify: start each turn by deciding the next actions (or the single measurement you need), then reason briefly about why. Never restart the analysis from scratch; your earlier assistant texts are your own conclusions and stay true until a measurement contradicts them.\n"
    "- When a measurement contradicts you, change exactly one rule, the one it contradicts, and act on the new rule at once. The simplest rule that fits the measured grid wins; the grid outranks the image and outranks any wording about it.\n"
    "- Confidence comes from the next action, not from more thought: a plan you are 70% sure of is worth executing now, because the board will settle it in one move.\n"
)

STRUCTURED_NOTE_ADDENDUM = (
    "\n\nStructured note before every action:\n"
    "- Immediately before each `action(...)` call, write four short labelled lines in your assistant text (not inside the tool call): "
    "`Hypothesis:` the rule you are testing or relying on; `Test:` the exact actions you are about to execute; "
    "`Prediction:` the concrete board change you expect (which object, which rows/cols it will occupy or how many cells it moves, or 'no change'); "
    "`Next:` what you will do if the prediction holds and what you will do if it fails.\n"
    "- The harness measures the actual change after the actions run and reports a deterministic verdict on your Prediction in the next prompt. "
    "A CONTRADICTED verdict means a rule in your action model is wrong: replace that rule before acting on it again.\n"
    "- Keep the lines specific and short. Numbers (rows, cols, cell counts) are checked; vague predictions are not useful.\n"
    "- Every note carries `p=<0..1>`: your probability that you understand the game well enough to attempt the level now. "
    "The harness switches you to exploitation (solving with committed rules) once p reaches the threshold; be decisive, exploration actions are not free.\n"
    "- When a note changes a memory field, pass `why=...`: the old text is kept next to the new one together with your reason, so nothing learned is thrown away.\n"
)

EXPLORATION_DESCRIBER_SYSTEM_PROMPT = (
    "You compare two frames of a grid-based game. The image shows two tiles: LEFT is the board before one input, "
    "RIGHT is the board after it. Report only what visibly changed: which objects (name them by colour, shape, size and "
    "position, e.g. 'the 2x2 blue square near the top-left') appeared, disappeared, moved (direction and how many cells), "
    "changed colour, grew or shrank. Mention separately if the only change is a thin bar or strip along an edge (that is "
    "usually a step counter). If nothing changed, say exactly 'No visible change.' Do not guess rules, goals, or what the "
    "input means, and do not describe parts of the board that stayed the same. Reply in at most 80 words of plain prose, "
    "no headings.\n"
    f"Color legend: {ARC_COLOR_LEGEND}.\n"
)

INTERMEDIATE_FRAMES_ADDENDUM = (
    "\n\nIntermediate frames:\n"
    "- When an action changes the board over several steps, the in-between frames are attached at the top of the turn, oldest first, ending at the current board image.\n"
    "- Use them to see HOW the board reached its state: what moved, in which direction, what it passed through or collided with, and whether a flash/animation marked a completion.\n"
    "- Only the current board image is live; the earlier frames are already past.\n"
)

REASONING_SUMMARY_SYSTEM_PROMPT = (
    "You are the coach standing behind a player at an arcade cabinet. The player just thought out loud for a whole turn; "
    "write the note they will glance at before the next move. Keep only what they must not lose: facts established about "
    "the game (mechanics, positions, what worked), the plan decided on, and the one open doubt if any. Numbers and "
    "coordinates exact. At most four lines starting with '- ', at most 60 words, no preamble.\n"
    "Example note:\n"
    "- RIGHT extends the arm 6 cols; the tip grabs a block when it reaches the block's column on the block's row\n"
    "- Red grabbed at c42 (legend icon went hollow); LEFT past the head drops it (icon refilled)\n"
    "- Plan: carry red DOWN x2 to green's row, RIGHT once into green, then UP x1 and RIGHT into blue\n"
    "- Doubt: whether DOWN moves the carried block or only the head\n"
)

SCENE_MODEL_SYSTEM_PROMPT = (
    "You are a seasoned arcade player at a cabinet you have never seen. Before touching the controls you read the "
    "screen like a human: what do I control, what are the targets, what is the legend or counter, what is a wall. "
    "You write the rulebook page the player keeps open; you do not choose moves.\n"
    "This game was MADE FOR HUMANS: there is always a clear goal indicator on screen (a legend or icon row with the "
    "target order, a target shape or pad, a counter, a lock and key, a highlighted slot); name it in Goal model. "
    "Rules: measured object sizes and positions are authoritative, the picture is for what belongs together; a "
    "button that changed nothing is inert or blocked here; a row of icons at an edge is usually the target order; "
    "a strip that shrinks every input is a counter. Be concrete and short (<= 250 words outside the map).\n"
    "Give every object its OWN role from what you see and measured (controlled piece, movable, target, wall, HUD, "
    "counter, decoration); at most one object is the controlled piece, and only if a probe moved it.\n"
    "Format (example from a different, made-up game; do not copy its roles, derive yours):\n"
    "World model: yellow 4x4 at r30-33 c8-11 = controlled piece (moved 4 cells on UP/DOWN probes); grey 4x4 crates at "
    "c20/32/44 = movable (RIGHT probe pushed one); green 4x4 pads at c56 = targets (static, crate-sized); dark frame = "
    "wall (never changed); row 62 white strip = move counter (shrank on every probe).\n"
    "Map:\n"
    "WWWWWWWW\n"
    "W......P\n"
    "WY.C.C.P\n"
    "WWWWWWWW\n"
    "letters: W=wall Y=piece C=crate P=pad (one char per 8x8 cells)\n"
    "Goal model: push a crate onto each pad (same size, distinct colour). Alt: reach pads with the piece (less likely).\n"
    "Action model: arrows move the piece 4 cells; RIGHT into a crate pushes it; ACTION7 undoes; MOUSE inert.\n"
    "Mechanics: 1. piece pushes a crate it walks into (RIGHT probe) 2. counter drops 1 per press (every probe).\n"
    "Open questions: can a crate be pulled (test: LEFT after a push).\n"
    "Plan: push the middle crate RIGHT x6 onto the lower pad.\n"
    f"Color legend: {ARC_COLOR_LEGEND}.\n"
)

EXPLORATION_SYNTHESIS_SYSTEM_PROMPT = (
    "You write the initial hypothesis for an agent about to play one level of a multi-level ARC-AGI-3 grid puzzle game. "
    "A deterministic exploration phase has just pressed every available button and clicked distinct objects, and an observer "
    "described each before/after change without any other context. From that log (and the board image, when attached) write "
    "what is on the board, what each input does, what the goal most likely is, what is still unknown, and what to try first.\n"
    "Principles:\n"
    "- Inputs that never changed the board are evidence that they are inert or blocked here; say so instead of inventing an effect.\n"
    "- A strip along an edge that shrinks or changes on every input is a step counter / HUD, not a puzzle object.\n"
    "- Distinguish objects that moved from objects that changed colour, and clicks that toggled something from clicks that did nothing.\n"
    "- Give the most likely goal plus 1-2 alternatives, each with its evidence, and the cheapest action that separates them.\n"
    "- Be concrete: colours, shapes, row/col positions, action names; give MOUSE targets as row/col.\n"
    "- Keep the whole answer under 400 words and output only the requested labeled sections.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)
