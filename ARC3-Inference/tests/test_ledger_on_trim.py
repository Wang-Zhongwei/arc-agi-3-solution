"""Memory ledger shown only after a history trim (ARC3_LEDGER_ON_TRIM)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from inference.agent import tool_agent as tool_agent_module
from inference.agent.memory_ledger import MemoryLedger
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent, _build_system_prompt, _ChatCompletionResult, _with_ledger_head

BLOCK = "<memory>\n  <hypothesis status=\"refuted\">MOUSE on a wall opens it</hypothesis>\n</memory>"


class _Opener(dict):
    """Stands in for the harness's marked turn-opener dict subclass."""


def test_text_head_gets_the_block_once():
    first = _with_ledger_head({"role": "user", "content": "board summary"}, BLOCK)
    assert first["content"] == f"{BLOCK}\n\nboard summary"
    # A head that survives into the next trim is refreshed, not stacked.
    newer = BLOCK.replace("MOUSE", "SPACE")
    second = _with_ledger_head(first, newer)
    assert second["content"] == f"{newer}\n\nboard summary"


def test_multimodal_head_keeps_its_image_parts():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    message = {"role": "user", "content": [{"type": "text", "text": "turn"}, image]}
    once = _with_ledger_head(message, BLOCK)
    twice = _with_ledger_head(once, BLOCK)
    assert twice["content"] == [{"type": "text", "text": BLOCK}, {"type": "text", "text": "turn"}, image]
    assert message["content"][0]["text"] == "turn"  # the stored original is untouched


def test_marker_subclass_survives():
    opener = _Opener(role="user", content="x")
    assert type(_with_ledger_head(opener, BLOCK)) is _Opener


def _agent(ledger, due):
    return SimpleNamespace(_ledger=ledger, _ledger_due=due)


def test_applied_once_per_trim_and_only_when_due():
    ledger = MemoryLedger()
    ledger.absorb('<hypothesis status="refuted">MOUSE on a wall opens it</hypothesis>')
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "turn 9"}]

    idle = _agent(ledger, due=False)
    assert ToolAgent._apply_due_ledger(idle, messages) is messages

    agent = _agent(ledger, due=True)
    out = ToolAgent._apply_due_ledger(agent, messages)
    assert out[1]["content"].startswith("<memory>\n")
    assert "refuted" in out[1]["content"]
    assert agent._ledger_due is False
    # The next request in the same window leaves the (now stable) head alone.
    assert ToolAgent._apply_due_ledger(agent, out) is out


def test_empty_ledger_adds_nothing():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "turn"}]
    assert ToolAgent._apply_due_ledger(_agent(MemoryLedger(), due=True), messages) is messages


def test_system_prompt_mentions_it_only_when_enabled(monkeypatch):
    monkeypatch.delenv("ARC3_LEDGER_ON_TRIM", raising=False)
    assert "survives history trimming" not in _build_system_prompt(tool_output_tokens=1024)
    monkeypatch.setenv("ARC3_LEDGER_ON_TRIM", "1")
    assert "survives history trimming" in _build_system_prompt(tool_output_tokens=1024)


# ---------------------------------------------------------------- with ARC3_VERBATIM_HISTORY, across a trim

REASONING = "\nI look at the board first.\n\nThe player is at (3, 4).\n   \nSo I probe UP.\n"


def _reply(reasoning, content="", code=None):
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if code is not None:
        message["tool_calls"] = [{"id": f"call_{len(code)}", "type": "function",
                                  "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
    return _ChatCompletionResult(message=message, finish_reason="tool_calls" if code is not None else "stop",
                                 usage={"prompt_tokens": 1000, "completion_tokens": 100})


def _text(message):
    content = message.get("content")
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content or [])


def test_ledger_returns_after_a_trim_and_kept_replies_stay_verbatim(tmp_path, monkeypatch):
    """The tput-safe pair: the ledger absorbs the normalised reply while history keeps it verbatim
    (the fix-3 merge in tool_agent.py), and a trim then shows the ledger on the first kept user
    message without touching the kept replies. The assistant-turn cap stands in for any trim."""
    for key, value in {"ARC3_LEDGER_ON_TRIM": "1", "ARC3_VERBATIM_HISTORY": "1",
                       "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
                       "ARC3_HISTORY_ASSISTANT_TURNS": "2", "ARC3_HISTORY_TURN_DRAIN": "0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(tool_agent_module, "run_sandboxed_python", lambda **kwargs: {
        "stdout": "ok\n", "result": None, "error": "", "action_results": [], "keepable_functions": []})
    state = tmp_path / "artifacts" / "g_p0_tool_runtime_state.json"
    frame = Frame(grid=((0, 1), (2, 3)), step=0, level=1)
    write_runtime_state(state, current_frame=frame, history=[HistoryEntry(action="", frame=frame)])
    agent = ToolAgent(model="test", base_url="http://127.0.0.1:9/v1", provider="vllm")
    replies = [
        # turn 1: a finding, a reply without a tool call (the harness answers with a user
        # message), then a reply whose blank lines history must keep
        _reply("Testing the wall.", '<hypothesis status="refuted">MOUSE on a wall opens it</hypothesis>',
               code="print(1)"),
        _reply("No call this time.", "I will look first."),
        _reply(REASONING, code="print(2)"),
        # turn 2
        _reply("Done.", code="print(3)"),
    ]
    sent = []

    def chat_completion(messages, **kwargs):
        sent.append(json.loads(json.dumps(messages)))
        return replies.pop(0)

    monkeypatch.setattr(agent, "_chat_completion", chat_completion)
    for action_num, steps in ((1, 3), (2, 1)):
        agent._tool_steps = steps
        agent.analyze(state, action_num=action_num, valid_actions=["UP"], step_env=lambda arguments: {})
    assert not replies and len(sent) == 4
    # no trim during turn 1, so no ledger (the system prompt only describes it)
    assert not any("<memory>\n" in _text(m) for request in sent[:3] for m in request[1:])
    # the cap kept the last two replies, so turn 2 starts after a trim
    after_trim = sent[3]
    users = [m for m in after_trim if m["role"] == "user"]
    assert _text(users[0]).startswith("<memory>\n")
    assert "MOUSE on a wall opens it" in _text(users[0])
    assert sum("<memory>\n" in _text(m) for m in after_trim[1:]) == 1
    kept = [m for m in after_trim if m["role"] == "assistant"]
    assert kept[-1]["reasoning_content"] == REASONING.strip()
    assert "\n\n" in kept[-1]["reasoning_content"]
    assert "<memory>" not in json.dumps(kept)


# ---------------------------------------------------------------- memory-v2 card

import random  # noqa: E402
import re  # noqa: E402

from inference.agent.memory_ledger import jaccard  # noqa: E402
from inference.agent.tool_agent import _LEDGER_BLOCK_RE  # noqa: E402

HARNESS_BANNED = re.compile(r"\b(inert|refuted|impossible|useless|cannot)\b", re.I)


def _card_ledger(**kw):
    return MemoryLedger(card_mode=True, solved_mode=True, delta_mode=True, **kw)


def test_default_ledger_still_folds_think_fragments():
    ledger = MemoryLedger()
    ledger.note_think("So the key must come first because the door stayed shut.")
    ledger.on_level_change(2)
    assert ledger.level_notes == ["level 1: So the key must come first because the door stayed shut."]
    assert ledger.last_think  # stored, never rendered by render()
    assert "So the key" not in ledger.render().replace("level 1: So the key", "")


def test_card_mode_keeps_raw_think_out_of_notes():
    ledger = _card_ledger()
    ledger.note_think("So the key must come first because the door stayed shut.")
    ledger.absorb('<hypothesis status="open">SPACE toggles the wall</hypothesis>')
    ledger.on_level_change(2)
    assert ledger.level_notes == []
    assert ledger._pending_think == []


def test_solved_alone_keeps_the_stack_v2_think_fold():
    # ARC3_MEM_SOLVED without the card adds solved lines to stack-v2's ledger:
    # the think fold and open-hypothesis carry-over stay, as level notes
    ledger = MemoryLedger(solved_mode=True, delta_mode=True)
    ledger.note_think("So the key must come first because the door stayed shut.")
    ledger.absorb('<hypothesis status="open">SPACE toggles the wall</hypothesis>')
    ledger.on_level_change(2)
    folded = "level 1: So the key must come first because the door stayed shut. open: SPACE toggles the wall"
    assert ledger.level_notes == [folded]
    # the solved line carries only the model's own note, and the fold stays on the card
    ledger.add_solved(1, ["UP", "UP"], 4, 0)
    assert "note:" not in ledger.items("solved")[0].text
    assert f"<level_note>{folded}</level_note>" in ledger.render_card()
    ledger.absorb("<level_note>the key opens the door</level_note>")  # now on level 2
    ledger.on_level_change(3)
    ledger.add_solved(2, ["LEFT"], 1, 0)
    assert ledger.items("solved")[-1].text.endswith("| note: the key opens the door")
    card = ledger.render_card()
    assert "<level_note>level 2: the key opens the door</level_note>" not in card
    assert f"<level_note>{folded}</level_note>" in card


def test_overlap_dedup_newest_wins_and_moves_last():
    ledger = _card_ledger()
    ledger.absorb("<world_model>The blue block is the player piece</world_model>")
    ledger.absorb("<world_model>Grey cells are walls</world_model>")
    ledger.absorb("<world_model>The blue block is the player piece now</world_model>")
    assert ledger.world_model == ["Grey cells are walls", "The blue block is the player piece now"]
    assert jaccard("The blue block is the player piece", "The blue block is the player piece now") >= 0.6
    ledger.absorb("<world_model>Grey cells are walls</world_model>")  # a restatement moves last
    assert ledger.world_model[-1] == "Grey cells are walls"


_DRAFT = 'maybe <hypothesis status="confirmed">the red cell kills the player</hypothesis>. Wait, no.'
_REPLY = 'Recording: <hypothesis status="refuted">the red cell kills the player on contact</hypothesis>'


def test_reasoning_draft_overwrites_the_reply_by_default(monkeypatch):
    # what memory-v2-both ran with: the reply is absorbed first, so the draft wins
    from inference.agent.tool_agent import _absorb_assistant_texts

    monkeypatch.delenv("ARC3_MEM_REPLY_WINS", raising=False)
    ledger = _card_ledger()
    _absorb_assistant_texts(ledger, _REPLY, _DRAFT)
    assert ledger.confirmed == ["the red cell kills the player"] and ledger.refuted == []


def test_reply_wins_absorbs_in_generation_order(monkeypatch):
    from inference.agent.tool_agent import _absorb_assistant_texts

    monkeypatch.setenv("ARC3_MEM_REPLY_WINS", "1")
    ledger = _card_ledger()
    _absorb_assistant_texts(ledger, _REPLY, _DRAFT)
    assert ledger.confirmed == []
    assert ledger.refuted == ["the red cell kills the player on contact"]
    plain = MemoryLedger()
    _absorb_assistant_texts(plain, "<scratchpad>final</scratchpad>", "<scratchpad>draft</scratchpad>")
    assert plain.scratchpad == "final"


def test_confirmed_refuted_flip_removes_the_older():
    ledger = _card_ledger()
    ledger.absorb('<hypothesis status="confirmed">SPACE opens the red door</hypothesis>')
    ledger.absorb('<hypothesis status="refuted">SPACE opens the red door</hypothesis>')
    assert ledger.confirmed == [] and ledger.refuted == ["SPACE opens the red door"]


def test_correction_supersedes_the_best_match():
    ledger = _card_ledger()
    ledger.absorb("<world_model>The player moves two cells per arrow press</world_model>")
    ledger.absorb("<world_model>Yellow is the goal</world_model>")
    ledger.absorb("<world_model>Correction: the player moves one cell per press</world_model>")
    assert ledger.world_model == ["Yellow is the goal", "The player moves one cell per press"]


def test_memory_close_tag_is_stripped_and_the_block_still_matches():
    ledger = _card_ledger()
    ledger.absorb("<world_model>a </memory> trap</world_model>")
    ledger.set_item("hazard:1:1", "hazard", "L1: UP killed </memory> attempt 1")
    card = ledger.render_card()
    assert card.count("</memory>") == 1
    assert _LEDGER_BLOCK_RE.fullmatch(card + "\n\n")


def test_card_entry_caps():
    ledger = _card_ledger()
    ledger.absorb("<world_model>" + "word " * 200 + "</world_model>")
    ledger.absorb('<hypothesis status="refuted">' + "claim " * 100 + "</hypothesis>")
    assert len(ledger.world_model[0]) <= 400 and ledger.world_model[0].endswith("...")
    assert len(ledger.refuted[0]) <= 250


def _fill_adversarial(ledger):
    for level in range(1, 11):
        ledger.last_think = "So pressing UP 4 times from (5,5) opens the door at (3,3). " * 3
        ledger.absorb(f"<level_note>level {level} note " + "x" * 300 + "</level_note>")
        ledger.add_solved(level, ["LEFT", "UP", "MOUSE(12,40)"] * 40, 300 + level, level)
    for i in range(12):
        ledger.absorb(f"<world_model>fact number {i} " + "alpha beta gamma " * 30 + f"{i}</world_model>")
        ledger.absorb(f'<hypothesis status="confirmed">claim {i} ' + "delta " * 50 + f"{i}</hypothesis>")
        ledger.absorb(f'<hypothesis status="refuted">idea {i} ' + "epsilon " * 40 + f"{i}</hypothesis>")
    rows = [f"MOUSE@{c} 30 targets: changed 10, no change 20; median 4 cells; e.g. (1,2) changed" for c in "RGBYWPOSMcNp"]
    ledger.set_item("actions", "actions", "\n".join(rows))
    for n in range(6):
        ledger.set_item(f"hazard:10:{n}", "hazard", f"L10: RIGHT from the board first reached at action {n} killed attempt {n} (31 actions)", order=n)


def test_card_hard_cap_under_adversarial_input():
    ledger = _card_ledger()
    _fill_adversarial(ledger)
    card = ledger.render_card()
    assert len(card) <= 3900
    assert card.count("<hazard>") == 3  # never dropped while anything else can go
    assert card.count('status="refuted"') >= 1
    assert "<solved>L1-L" in card and " acts, spent " in card  # older levels compressed to counts
    small = _card_ledger(card_chars=1200)
    _fill_adversarial(small)
    assert len(small.render_card()) <= 1200


def test_card_order_and_harness_wording():
    ledger = _card_ledger()
    ledger.absorb("<world_model>blue is the player</world_model>")
    ledger.last_think = "So UP 3 times reaches the goal at (5,5)."
    ledger.add_solved(1, ["UP"] * 3, 9, 0)
    ledger.set_item("actions", "actions", "UP: moved 3\nSPACE: no interior change on 4 presses across 2 boards")
    ledger.set_item("hazard:2:1", "hazard", "L2: RIGHT from its start board killed attempt 1 (1 action)")
    card = ledger.render_card()
    order = [card.index(tag) for tag in ("<solved>", "<actions>", "<hazard>", "<world_model>")]
    assert order == sorted(order)
    harness = "\n".join(line for line in card.splitlines() if "<world_model>" not in line)
    assert not HARNESS_BANNED.search(harness)


def test_actions_section_names_its_level():
    # a head card can outlive its level; the counts say which level they are for
    ledger = _card_ledger()
    ledger.set_item("actions", "actions", "UP: moved 3", data={"level": 3})
    assert '  <actions level="3">\n  UP: moved 3\n  </actions>' in ledger.render_card()


# ---------------------------------------------------------------- deltas

def test_delta_takes_only_newer_lines_and_groups_effects():
    ledger = _card_ledger()
    ledger.add_event("fx:UP:shift", "UP: first moved")
    ledger.add_event("fx:SPACE:noop", "SPACE: first no change")
    ledger.absorb("<world_model>blue is the player</world_model>")
    block = ledger.take_delta()
    assert block == (
        "<memory_update>\n+ UP: first moved; SPACE: first no change\n"
        "+ world_model: blue is the player\n</memory_update>"
    )
    assert ledger.take_delta() == ""


def test_actions_counts_are_not_deltas():
    ledger = _card_ledger()
    ledger.set_item("actions", "actions", "UP: moved 1")
    v = ledger.version
    ledger.set_item("actions", "actions", "UP: moved 2", bump=False)
    assert ledger.version == v and ledger.take_delta() == ""


def test_delta_carry_then_flush():
    ledger = _card_ledger()
    for i in range(30):
        ledger.add_event(f"hazard:1:{i}", f"L1: RIGHT from the board first reached at action {i} killed attempt {i} (31 actions)")
    sizes = []
    while ledger.pending():
        sizes.append(len(ledger.take_delta(cap=480)))
    assert sizes[0] <= 480 + 40 and sizes[1] <= 480 + 40
    assert max(sizes[2:]) > 520  # the third carry doubles the cap


def test_hold_lets_only_bypass_lines_through():
    ledger = _card_ledger()
    ledger.add_event("fx:UP:shift", "UP: first moved")
    ledger.add_solved(1, ["UP"], 1, 0)
    block = ledger.take_delta(hold=True)
    assert "solved L1" in block and "UP: first moved" not in block
    assert "UP: first moved" in ledger.take_delta(hold=False)


def test_game_cap_stops_deltas():
    ledger = _card_ledger()
    for i in range(50):
        ledger.add_event(f"x{i}", "y" * 100)
    total = 0
    for _ in range(50):
        total += len(ledger.take_delta(game_cap=1000))
    assert 500 < total <= 1000


def test_mark_shown_drops_what_the_card_covers():
    ledger = _card_ledger()
    ledger.add_event("fx:UP:shift", "UP: first moved")
    ledger.render_card()
    ledger.mark_shown(ledger.version)
    ledger.add_event("fx:UP:noop", "UP: first no change")
    assert ledger.take_delta() == "<memory_update>\n+ UP: first no change\n</memory_update>"


def test_rollback_restores_taken_lines_and_keeps_new_ones():
    ledger = _card_ledger()
    ledger.add_event("fx:UP:shift", "UP: first moved")
    snap = ledger.snapshot_display()
    assert "UP: first moved" in ledger.take_delta()
    ledger.add_event("fx:SPACE:noop", "SPACE: first no change")  # absorbed during the failed turn
    ledger.mark_shown(ledger.version)  # a head card in the failed turn
    ledger.restore_display(snap)
    block = ledger.take_delta()
    assert "UP: first moved" in block and "SPACE: first no change" in block
    assert ledger.shown_version == 0


def test_a_new_level_drops_the_last_levels_queued_lines():
    ledger = _card_ledger()
    ledger.add_event("fx:LEFT:shift", "LEFT: first moved")
    ledger.set_item("hazard:1:2", "hazard", "L1: attempt 2 ended in GAME OVER after 2 actions",
                    event="L1: attempt 2 ended in GAME OVER after 2 actions")
    ledger.add_solved(1, ["UP"] * 7, 11, 2)
    ledger.absorb("<world_model>blue is the player</world_model>")
    snap = ledger.snapshot_display()
    ledger.clear_level_facts()
    ledger.add_event("fx:UP:noop", "UP: first no change")  # on the new level
    assert not ledger.items("hazard")
    block = ledger.take_delta()
    assert "solved L1" in block and "world_model: blue" in block and "UP: first no change" in block
    assert "LEFT: first moved" not in block and "GAME OVER" not in block
    # a rolled-back turn does not bring them back
    ledger.restore_display(snap)
    texts = [e.text for e in ledger.pending()]
    assert "LEFT: first moved" not in texts and not any("GAME OVER" in t for t in texts)


_CARD_LINE_RE = re.compile(r"<(?:solved|hazard|world_model|hypothesis[^>]*)>(.*?)</")


def _random_memory_ops(ledger, rng, steps, on_card):
    """Random notes, solved levels, hazards, deltas, abandoned turns and head
    cards; ``on_card()`` renders a head card. Returns the deltas shown since."""
    shown_after = []
    for _ in range(steps):
        op = rng.random()
        if op < 0.2:
            ledger.absorb(f"<world_model>fact {rng.randint(0, 30)} about colour {rng.randint(0, 9)}</world_model>")
        elif op < 0.3:
            status = rng.choice(["confirmed", "refuted"])
            ledger.absorb(f'<hypothesis status="{status}">claim {rng.randint(0, 20)} on {rng.randint(0, 5)}</hypothesis>')
        elif op < 0.4:
            level = rng.randint(1, 6)
            ledger.add_solved(level, ["UP"] * rng.randint(1, 9), rng.randint(5, 40), rng.randint(0, 3))
        elif op < 0.5:
            n = rng.randint(0, 99)
            text = f"L1: attempt {n} ended in GAME OVER after 3 actions"
            ledger.set_item(f"hazard:1:{n}", "hazard", text, order=n, event=text)
        elif op < 0.75:
            block = ledger.take_delta(cap=480)
            if block:
                shown_after.append(block)
        elif op < 0.85:
            snap = ledger.snapshot_display()
            ledger.take_delta(cap=480)
            ledger.restore_display(snap)  # an abandoned turn
        else:
            on_card()
            ledger.mark_shown(ledger.version)
            shown_after = []
    while ledger.pending():
        shown_after.append(ledger.take_delta(cap=480))
    return shown_after


def test_property_head_card_plus_later_deltas_cover_the_card():
    rng = random.Random(7)
    for trial in range(30):
        ledger = _card_ledger(card_chars=100000)
        head = {"text": ""}

        def on_card():
            head["text"] = ledger.render_card()

        shown_after = _random_memory_ops(ledger, rng, 80, on_card)
        seen = head["text"] + "\n".join(shown_after)
        lines = _CARD_LINE_RE.findall(ledger.render_card())
        assert lines
        for text in lines:
            assert text in seen, (trial, text)


def test_property_capped_card_hides_only_what_the_cap_cut():
    # under the hard cap a head card leaves lines out, and mark_shown treats
    # them as shown: the only lines the model never saw are ones the cap cut
    rng = random.Random(11)
    cut_cards = 0
    for trial in range(30):
        cap = rng.choice([400, 700, 1200])
        ledger = _card_ledger(card_chars=cap)
        head = {"text": "", "cut": set()}
        cuts = [0]

        def on_card():
            text = ledger.render_card()
            assert len(text) <= cap
            ledger.card_chars = 10**6
            whole = set(_CARD_LINE_RE.findall(ledger.render_card()))
            ledger.card_chars = cap
            head["text"], head["cut"] = text, whole - set(_CARD_LINE_RE.findall(text))
            cuts[0] += bool(head["cut"])

        shown_after = _random_memory_ops(ledger, rng, 120, on_card)
        seen = head["text"] + "\n".join(shown_after)
        final = ledger.render_card()
        assert len(final) <= cap
        ledger.card_chars = 10**6
        whole = set(_CARD_LINE_RE.findall(ledger.render_card()))
        # shortened solved lines ("L1-L4: 13/22/9/31 acts ...") are derived, not items
        checked = [text for text in _CARD_LINE_RE.findall(final) if text in whole]
        assert checked
        for text in checked:
            assert text in seen or text in head["cut"], (trial, cap, text)
        cut_cards += cuts[0]
    assert cut_cards >= 30  # the cap really cut head cards
