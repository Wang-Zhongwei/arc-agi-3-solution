"""Role prompts and the state renderer.

Every prompt is framed as a real job (project rule): the model is never "an AI solving a
puzzle", it is a professional with a method. The harness renders the state in eight budgeted
blocks (JOB, OBJ, LAST, VARS, RULES, MAP, MEM, ASK) and never shows the grid.

Budget: ~1200 tokens of state, enforced in characters (4 chars per token, checked in a test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import MAX_BATCH
from .types import Mode

# The board and the change map are one character per cell, so they are large in characters and
# cheap in time: the project's own measurement is that prefill is ~1% of request latency while
# decode is ~12% and queue wait the rest, so input tokens are close to free and output tokens
# are not. Showing the raw board is therefore affordable in a way that a longer answer is not.
STATE_CHAR_BUDGET = 18000
# MAP carries the harness's conclusions (the plan, the separating feature) ahead of its raw
# tallies, and needs room for both: the control listing alone runs ~380 chars on a five-action
# game, and at 340 the separator line was being clipped away entirely.
BLOCK_BUDGET = {"JOB": 240, "GRID": 4700, "DELTA": 5200, "SYM": 1200, "OBJ": 1100, "LAST": 520,
                "VARS": 480, "RULES": 820, "MAP": 700, "MEM": 820, "ASK": 400}

GRID_NOTE = (
    "GRID is the board itself, one character per cell, `.` for the background colour, with row "
    "numbers down the left and column numbers across the top. DELTA is what the last action "
    "changed: the exact cell list with the raw colour numbers, then the same board with only the "
    "changed cells drawn in the colour they became. OBJ is the harness's reading of the same "
    "board as tracked objects with stable ids; when the two disagree, the board is the truth."
)

META_PRIORS = (
    "Method: identify what you control before anything else; salience is not the goal; test opposites; "
    "separate what changes on its own from what you caused; a contradiction usually means hidden state, "
    "not a broken rule; a thing can recolour or move and still be the same thing."
)

SYSTEM = {
    # decide_lean is the same job with a smaller form to fill in; it shares the physicist prompt.
    "orient": (
        "You are a field naturalist who has just landed on an unknown island with a notebook. You do not ask how to "
        "leave; you ask what is alive here, what moves on its own, what responds to you, and what is worth writing "
        "down. The instrument readings below were taken by your assistant (the harness) and are exact; your job is "
        "interpretation. " + META_PRIORS + " Answer only with the JSON object requested."
    ),
    "interpret": (
        "You are a radar operator labelling blips. The console (the harness) has already segmented the screen into "
        "tracked objects with stable ids; you say what each one probably is, which ones matter, and what you cannot "
        "tell yet. Never invent an object that is not on the console. Be brief. Answer only with the JSON object requested."
    ),
    "decide": (
        "You are an experimental physicist choosing the next measurement. The lab record (the harness) is exact: "
        "the object table, the last outcome, the variables, the scored rules. You propose the next action batch and, "
        "for each step, what you expect to change and with what probability; the record will grade you. A rule you "
        "propose must cite the steps that support it; it gets its confidence from the record, not from you. "
        + META_PRIORS + " Answer only with the JSON object requested."
    ),
    "critic": (
        "You are an accident investigator. A prediction just failed. Before anyone rewrites the manual, you classify "
        "what kind of failure it was: a misread instrument (perception), a swapped identity, a wrong idea of what a "
        "control does, wrong dynamics, hidden state, timing, a wrong goal, or a bad plan. One kind, one sentence. "
        "Answer only with the JSON object requested."
    ),
    "goal": (
        "You are a surveyor asked for one thing: state where the work has to end up. You are given "
        "an inventory of what is on the board and the exact distances between the pieces. You do "
        "not plan the route and you do not choose actions; a solver does that, and it can only "
        "search for goals written in the algebra below. State the condition that would END THIS "
        "LEVEL, not the next thing worth doing: the solver will find the route, so a goal one "
        "step away wastes it. Answer only with the JSON object requested."
    ),
    "hypothesis": (
        "You are the mediator between colleagues who disagree about why the last result happened. You do not pick a "
        "winner; you write down each competing explanation, and for each one the single observation that would tell "
        "it apart from the others, with the action that produces it. Answer only with the JSON object requested."
    ),
}

for _role in list(SYSTEM):
    SYSTEM[_role] = SYSTEM[_role] + " " + GRID_NOTE

SYSTEM["decide_lean"] = SYSTEM["decide"]

ASK = {
    Mode.ORIENT: "Name what you can control (from the control map), label the entities, say what to track, watch or ignore, and list the questions the next few actions should answer.",
    Mode.EXPLORE: "Pick the batch that answers the most important open question or tests the least-supported rule, and predict each step. Commit as many actions as you can still predict: a turn costs several minutes of the run and an action costs almost nothing, so a single-action turn throws the turn away. Four to eight is normal once any control is known; stay short only when the very next result would change what you do.",
    Mode.VERIFY: "Two explanations disagree, or one channel blocks execution. Pick the batch on which they predict different outcomes and predict each step. Put the discriminating action first, then keep going with what you expect to follow: the batch stops itself on the first surprise, so a long batch costs nothing when you are right.",
    Mode.EXECUTE: "The world model is trusted. Commit the plan (up to {max_batch} steps) with a prediction per step; stop and re-plan on the first surprise.",
    Mode.RECOVER: "A trusted prediction failed. Use the critic's verdict and the new hypotheses; choose the batch that separates them. Put the discriminating action FIRST, then keep going with what you expect to follow from whichever explanation wins: the batch stops itself on the first surprise, so the steps after the discriminator cost nothing when you are right and are dropped when you are not. A one-action turn spends six minutes of the run to learn one thing.",
}

ACTION_SEMANTICS = (
    "Actions are keys, not clicks: ACTION1..ACTION5 and ACTION7 take no target and no coordinates; they act on "
    "whatever the game lets you control (the MAP block says what each one did so far; the OBJ header names the "
    "controlled body once it is known). Only ACTION6 is a click and takes x (column) and y (row)."
)

DECIDE_FORMAT = (
    "The batch halts by itself on the first step whose prediction fails, so a long batch is never "
    "a gamble: you lose nothing but the steps after the surprise, and you gain every step before it. "
    "memory_patch.disputes: name anything the harness is holding that the record does not support "
    "-- a rule id, a question id, a hypothesis id, a track id -- and say why. Disputing is not "
    "deleting: it sends the thing back to being a claim that has to earn its place again. Use it. "
    "A memory only ever added to is one nobody is allowed to be wrong in.\n"
    "Format rules: batch <= {max_batch} actions from the available list; predict at most the first 6 "
    "steps, and only the ones worth grading, because an answer that runs past the token limit is lost "
    "whole rather than trimmed (step numbers start at 1); expect.var is a name from VARS or obj:<tid>.pos / obj:<tid>.colour / obj:<tid>.gone; "
    "ops: delta (value like '3' or '(-1,0)'), equals, unchanged, changed; p in [0,1]. memory_patch.new_rules: type in "
    "{{invariant,conditional,autonomous,correlation,causal,temporal,goal}}, when/then as JSON strings in the rule DSL "
    "shown, because = ledger step indices. Leave arrays empty when you have nothing to add. Confidence values are your "
    "own estimate per channel; the harness recomputes rule confidences from the record."
)

DSL_HINT = (
    "Rule DSL: when = {{\"and\":[{{\"action_is\":\"ACTION1\"}},{{\"adjacent\":[\"$self\",\"class:c\",\"up\"]}}]}}; "
    "then = [{{\"move\":[\"$self\",[0,0]]}}] or [{{\"vanish\":[\"obj:t4\"]}}] or [{{\"delta\":[\"consumed[Y]\",1]}}] or "
    "[{{\"level_complete\":true}}]. Terms: $self, obj:<tid>, class:<letter>, var:<name>, cell:(r,c)."
)


@dataclass
class StateBlocks:
    """Already-rendered pieces the session assembles; prompts.py only budgets and orders them."""
    job: str = ""
    grid: str = ""
    delta: str = ""
    sym: str = ""
    obj: str = ""
    last: str = ""
    vars: str = ""
    rules: str = ""
    map: str = ""
    mem: str = ""
    ask: str = ""
    extras: dict[str, str] = field(default_factory=dict)


def _clip(text: str, budget: int) -> str:
    text = text.strip()
    if len(text) <= budget:
        return text
    return text[: budget - 3].rstrip() + "..."


def render_state(blocks: StateBlocks, *, char_budget: int = STATE_CHAR_BUDGET) -> str:
    """Eight blocks in fixed order, each clipped to its budget, the whole clipped to the total."""
    order = [("JOB", blocks.job), ("GRID", blocks.grid), ("DELTA", blocks.delta), ("SYM", blocks.sym),
             ("OBJ", blocks.obj),
             ("LAST", blocks.last), ("VARS", blocks.vars), ("RULES", blocks.rules), ("MAP", blocks.map),
             ("MEM", blocks.mem), ("ASK", blocks.ask)]
    lines = []
    for name, text in order:
        if not text:
            continue
        # A grid is only readable whole, so it goes on its own lines under its heading rather
        # than being run onto one.
        if name in ("GRID", "DELTA", "SYM"):
            lines.append(f"{name}\n{_clip(text, BLOCK_BUDGET[name])}")
            continue
        lines.append(f"{name}  {_clip(text, BLOCK_BUDGET[name])}")
    out = "\n".join(lines)
    if len(out) > char_budget:
        # Drop from the least decision-relevant block first: MEM, then MAP, then RULES tails.
        for name in ("MEM", "MAP", "RULES", "OBJ", "DELTA"):
            lines = [l for l in lines if not l.startswith(name + "  ")] + [l for l in lines if l.startswith(name + "  ")]
            lines[-1] = _clip(lines[-1], max(80, BLOCK_BUDGET[name] // 2))
            out = "\n".join(lines)
            if len(out) <= char_budget:
                break
    return out[:char_budget]


def user_prompt(role: str, mode: Mode, state_text: str, *, available: list[str], extra: str = "") -> str:
    avail = ", ".join(available)
    if role in ("decide", "decide_lean"):
        ask = ASK[mode].format(max_batch=MAX_BATCH)
        return (f"{state_text}\nAVAILABLE actions: {avail}\n{ACTION_SEMANTICS}\n{ask}\n"
                f"{DECIDE_FORMAT.format(max_batch=MAX_BATCH)}\n{DSL_HINT}\n{extra}").strip()
    if role == "interpret":
        return (f"{state_text}\nLabel the tracked objects (tid -> label, role, p), name up to 6 salient ones, and add at most "
                f"three questions the harness cannot answer from the table alone. Roles: player, wall, floor, collectable, hud, "
                f"goal, hazard, switch, door, counter, decor, unknown.\n{extra}").strip()
    if role == "orient":
        return (f"{state_text}\nAVAILABLE actions: {avail}\n{ACTION_SEMANTICS}\n{ASK[Mode.ORIENT]}\nRoles: player, wall, floor, collectable, hud, goal, "
                f"hazard, switch, door, counter, decor, unknown. Goal hypotheses are guesses with p; the harness will test them.\n{extra}").strip()
    if role == "goal":
        return (f"{state_text}\nFill in the form for THIS level's goal. The solver assembles it; you "
                f"only choose the parts.\n"
                f"  predicate all_touched -> reach every thing of a class   (target: class:<X>)\n"
                f"  predicate touched     -> reach one particular thing     (target: obj:<tid>)\n"
                f"  predicate adjacent    -> end up next to something       (target: obj:<tid> or class:<X>)\n"
                f"  predicate at          -> stand on a square              (row and col, 0..63)\n"
                f"  predicate count       -> leave at most n of a class unreached (target: class:<X>, op, n)\n"
                f"  predicate covered     -> every piece of a class ends up on a piece of another "
                f"(target: class:<X>, onto: class:<Y>)\n"
                f"`target` must name something in the SYM inventory: an id like obj:t35, or a class like "
                f"class:Y or class:Y ring. Fill row/col only for `at`, op/n only for `count`, and onto "
                f"only for `covered`; leave the others null.\n"
                f"`covered` is the only one here that is not about where your body goes. Use it when "
                f"the level looks like it ends once things are ARRANGED rather than once you have "
                f"reached them.\n"
                f"Prefer a class over one object whenever several things share a description. A "
                f"class goal stays worth solving as the board empties; a single object stops being "
                f"a goal the moment it is reached, and the level does not end because of it.\n"
                f"{extra}").strip()
    if role == "critic":
        return (f"{state_text}\nClassify the failure in LAST. kinds: perception, identity, control, dynamics, hidden_state, temporal, "
                f"goal, plan. fix: retrack, demote_control, open_question, replan, regoal, none.\n{extra}").strip()
    if role == "hypothesis":
        return (f"{state_text}\nAVAILABLE actions: {avail}\n{ACTION_SEMANTICS}\nWrite 2-3 competing explanations for the failure in LAST. For each, the "
                f"test_action and the single predicted variable/op/value that separates it from the others.\n{extra}").strip()
    raise ValueError(role)


COMPACT = "Emit compact JSON on one line (no indentation, no newlines inside strings); every token you spend on whitespace is a token not spent on thought."


def effort_sentence(mode: Mode) -> str:
    if mode in (Mode.RECOVER, Mode.VERIFY):
        return "Weigh both explanations before answering. " + COMPACT
    return "Answer directly; the observation is routine. " + COMPACT
