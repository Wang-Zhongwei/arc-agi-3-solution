"""arm/mem-repair-rules: ARC3_RULES_FILTER_V2 and ARC3_RULES_ACTION_TAG.

The sentences are stall-fix-10x's own quoted rules (the run-wide audit of
its 512 rules, hand-labelled): the ones v2 drops were shown while the model
was stuck (re86 L7, dc22 L3, wa30 L3, m0r0 L3, sc25 L6, tr87 L2).
"""
from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path

import pytest

from inference.agent import memory_layout as layout
from inference.agent.confirmed_rules import RuleLedger, quoted, rule_sentences
from inference.agent.memory_ledger import MemoryLedger

FIXTURE = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "mem_unified_states.json.gz", "rt").read())

# a marked sentence the v1 filter rejects (two coordinate pairs): the "So"
# sentence after it inherits the mark
EVIDENCE = "The token moved from (3,4) to (5,6) ✓. "


def _reply(key: str) -> str:
    reply = FIXTURE["replies"][key]
    return reply["reasoning"] + "\n" + (reply["content"] or "")


@pytest.mark.parametrize("text, rule", [
    # (c) an echo of an earlier level's note, contradicted on re86 L7
    ("Level 6: the dots were absorbed when covered by the same-colored piece ✓.",
     "Level 6: the dots were absorbed when covered by the same-colored piece ✓."),
    # (d) a plan (tr87 L2)
    ("The order list: I'll maintain as a list where consecutive entries are UP-successors, and treat it as "
     "cyclic only when I've confirmed the wrap (i.e., when the chain returns to the start).",
     "The order list: I'll maintain as a list where consecutive entries are UP-successors, and treat it as "
     "cyclic only when I've confirmed the wrap (i.e., when the chain returns to the start)."),
    # (f) unmarked conclusions stating an absolute (dc22 L3, wa30 L3, m0r0 L3)
    (EVIDENCE + "So the purple region can NEVER be entered.", "So the purple region can NEVER be entered."),
    (EVIDENCE + "So: the player can never cross the wall; the 3 left boxes can never be delivered.",
     "So: the player can never cross the wall; the 3 left boxes can never be delivered."),
    (EVIDENCE + "So the ONLY passage between the halves is blocked by a dot, and dots are impassable walls, "
     "and clicking a dot is a penalty.",
     "So the ONLY passage between the halves is blocked by a dot, and dots are impassable walls, and clicking "
     "a dot is a penalty."),
    # (e) a hedge (sc25 L6)
    (EVIDENCE + 'So the outer frame may just be a "big marker" that\'s decorative/indicating the last teleport.',
     'So the outer frame may just be a "big marker" that\'s decorative/indicating the last teleport.'),
    # (a) a range of rows, (a) a pair of ranges
    ("So with the upright mold the dye always affects rows 34-38 ✓.",
     "So with the upright mold the dye always affects rows 34-38 ✓."),
    ("Second beam went LEFT and hit the ring at (43-46, 3-6) whose core was orange → destroyed all orange ✓.",
     "Second beam went LEFT and hit the ring at (43-46, 3-6) whose core was orange → destroyed all orange ✓."),
    # (b) fragments
    ("So the deformation rule for a blocked press in direction D ✓:", "So the deformation rule for a blocked "
     "press in direction D ✓:"),
    ("no — the landing must be floor, but the path can cross walls ✓).",
     "no — the landing must be floor, but the path can cross walls ✓)."),
    ("→ the N plate must be at the gap row at the win ✓.", "→ the N plate must be at the gap row at the win ✓."),
    # (c) quoted text naming a level
    ('In level 2: "landing ON the red tile destroys it and is safe" — that was confirmed there.',
     'In level 2: "landing ON the red tile destroys it and is safe" — that was confirmed there.'),
    # (c) an echo word, (d) the model's own code
    ("Note: RESET must be the first game action in a Python snippet ✓.",
     "Note: RESET must be the first game action in a Python snippet ✓."),
    ("The loop over the solver states matches every win ✓ so the solver must be right.",
     "The loop over the solver states matches every win ✓ so the solver must be right."),
    # (e) a condition
    ("If the cycle is {0,1} only, then a rotation = swap ✓.", "If the cycle is {0,1} only, then a rotation = swap ✓."),
])
def test_v2_drops_what_v1_kept(text, rule):
    assert rule in rule_sentences(text)
    assert rule not in rule_sentences(text, v2=True)


@pytest.mark.parametrize("text, rule", [
    # marked conclusions with an absolute keep their own mark
    ("So each knot must go to the ring with the SAME color set ✓✓.",
     "So each knot must go to the ring with the SAME color set ✓✓."),
    # an unmarked conclusion without an absolute still inherits the mark
    (EVIDENCE + "So the mechanic: moving the white knot onto a colored blob absorbs the blob's colors.",
     "So the mechanic: moving the white knot onto a colored blob absorbs the blob's colors."),
    # level names without a quote are kept (35 of the audit's 186 mechanics name a level)
    ("In level 2, clicking a piece moved the active dot to it ✓.",
     "In level 2, clicking a piece moved the active dot to it ✓."),
    # short quoted labels are kept
    ("Moving LEFT moves both left ✓ (the box moves left, \"pulling\").",
     "Moving LEFT moves both left ✓ (the box moves left, \"pulling\")."),
    # a sentence opening with "and" is kept (the lab quoted one)
    ("Hmm, and the knot's path doesn't matter (teleport) ✓.", "and the knot's path doesn't matter (teleport) ✓."),
])
def test_v2_keeps_mechanics(text, rule):
    assert rule in rule_sentences(text, v2=True)


@pytest.mark.parametrize("key", ["r11l_A_94", "r11l_A_100", "r11l_A_117", "m0r0_A_63"])
def test_v2_keeps_the_rules_the_lab_quoted(key):
    v1 = rule_sentences(_reply(key))
    v2 = rule_sentences(_reply(key), v2=True)
    lab = {
        "So each knot must go to the ring with the SAME color set ✓✓.",
        "So the match is by color SET, not counts ✓.",
        "So the mechanic: moving the white knot onto a colored blob absorbs the blob's colors (the knot becomes "
        "half W/half R as the blob was), and the blob disappears.",
        "and the knot's path doesn't matter (teleport) ✓.",
        "So the selected (yellow) dot is a movable object, moving in 4-cell steps, while the blocks are frozen.",
    }
    assert set(v1) & lab and set(v1) & lab <= set(v2)
    assert set(v2) <= set(v1)


def test_ledger_filter_v2_is_off_by_default():
    text = EVIDENCE + "So the purple region can NEVER be entered."
    plain, v2 = RuleLedger(), RuleLedger(filter_v2=True)
    plain.absorb(3, text)
    v2.absorb(3, text)
    assert plain.rules_for(3) == ["So the purple region can NEVER be entered."]
    assert v2.rules_for(3) == []


# ---------------------------------------------------------------- ARC3_RULES_ACTION_TAG

SC25_OLD = "Either way, hitting any dark red destroys all dark red ✓."
SC25_NEW = "In this level, the beam hit the dark red SEGMENT (plain cells) and destroyed nothing ✓ confirmed."


def test_tags_name_the_action_and_order_oldest_first():
    ledger = RuleLedger(filter_v2=True, tag_actions=True)
    ledger.absorb(6, SC25_OLD, action=210)
    ledger.absorb(6, SC25_NEW, action=248)
    rules = ledger.rules_for(6)
    assert rules == [SC25_OLD, SC25_NEW] and [r.action for r in rules] == [210, 248]
    assert layout.this_level_rules_line(rules) == (
        f'Rules you confirmed on this level: "{SC25_OLD}" (action 210) | "{SC25_NEW}" (action 248)')


def test_tags_off_or_unknown_render_as_before():
    plain = RuleLedger()
    plain.absorb(6, SC25_OLD, action=210)
    assert quoted(plain.rules_for(6)) == f'"{SC25_OLD}"'
    untagged = RuleLedger(tag_actions=True)
    untagged.absorb(6, SC25_OLD)
    assert quoted(untagged.rules_for(6)) == f'"{SC25_OLD}"'


def test_a_restatement_takes_the_newer_action():
    ledger = RuleLedger(tag_actions=True)
    ledger.absorb(2, "Confirmed: RIGHT always stretches the block two cells to the right ✓.", action=5)
    ledger.absorb(2, "Confirmed: RIGHT always stretches the block two cells right, then it settles ✓.", action=9)
    assert [(str(r), r.action) for r in ledger.rules_for(2)] == [
        ("Confirmed: RIGHT always stretches the block two cells right, then it settles ✓.", 9)]


def test_tagged_rules_survive_copies_and_ride_in_the_frozen_entry():
    ledger = MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True, recap_ask=True)
    ledger.layout_mode = ledger.legible = True
    ledger.rules = RuleLedger(tag_actions=True)
    ledger.rules.absorb(4, "So each knot must go to the ring with the SAME color set ✓✓.", action=63)
    ledger.add_solved(4, ["MOUSE(1,2)"], 3, 0)
    block, = ledger.layout_stage([4])
    assert '  Rules you confirmed: "So each knot must go to the ring with the SAME color set ✓✓." (action 63)' in block
    rule = ledger.prev_entries[4].confirmed[0]
    for clone in (copy.deepcopy(rule), pickle.loads(pickle.dumps(rule))):
        assert clone == rule and clone.action == 63


def test_absorb_rules_passes_the_turn_action():
    from inference.agent.tool_agent import _absorb_rules

    ledger = MemoryLedger()
    ledger.rules = RuleLedger(tag_actions=True)
    ledger.on_level_change(3)
    _absorb_rules(ledger, None, "Confirmed: the purple checker always blocks the token ✓.", action=104)
    assert [r.action for r in ledger.rules.rules_for(3)] == [104]


# ---------------------------------------------------------------- SF's replies, replayed

SF = json.loads(gzip.open(Path(__file__).parent / "fixtures" / "sf_rules_replies.json.gz", "rt").read())


def _replay(game: str, **flags) -> dict[int, str]:
    """The <this_level> rules line at each of the level's turn messages."""
    data = SF["games"][game]
    ledger, lines = RuleLedger(**flags), {}
    replies = iter(data["replies"])
    reply = next(replies, None)
    for turn in data["lines"]:
        while reply is not None and reply["i"] < turn["i"]:
            if reply["absorbed"]:
                ledger.absorb(data["level"], reply["reasoning"], action=reply["action"])
                ledger.absorb(data["level"], reply["content"], action=reply["action"])
            reply = next(replies, None)
        lines[turn["i"]] = layout.this_level_rules_line(ledger.rules_for(data["level"]))
    return lines


@pytest.mark.parametrize("game", ["sc25", "re86"])
def test_flags_off_reproduces_the_logged_lines(game):
    assert _replay(game) == {turn["i"]: turn["logged"] for turn in SF["games"][game]["lines"]}


def test_sc25_level6_the_refutation_follows_the_rule():
    on = _replay("sc25", filter_v2=True, tag_actions=True)
    assert on[97] == (
        'Rules you confirmed on this level: "Either way, hitting any dark red destroys all dark red ✓." (action 210)'
        ' | "So the teleport chose the left-area marker (not the nearest)." (action 233)'
        ' | "In this level, the beam hit the dark red SEGMENT (plain cells) and destroyed nothing ✓ confirmed."'
        ' (action 248)')
    # off, the hedged "big marker" guess took the place of the refutation
    assert "big marker" in _replay("sc25")[97] and "big marker" not in on[97]


def test_re86_level7_drops_the_level6_echo():
    off, on = _replay("re86"), _replay("re86", filter_v2=True)
    stuck = [i for i in off if "Level 6: the dots were absorbed" in off[i]]
    assert len(stuck) >= 8 and not any("Level 6:" in on[i] for i in stuck)
    assert on[94] == ('Rules you confirmed on this level: "So pressing UP while blocked → u+3, d-3 ✓." | "Blocked '
                      'RIGHT press: the vertical bar must be at col 27 (left of the block) spanning the block\'s rows '
                      '→ pressing RIGHT → the vertical stays, the horizontal slides right → l-3, r+3 ✓." | "So 2 '
                      'blocked presses in the vertical direction."')
