"""ARC3_CONFIRMED_RULES: "confirmed" thinking sentences from trimmed replies, pinned by level."""

from inference.agent import confirmed_rules, tool_agent
from inference.agent.confirmed_rules import HEADER, RULES_MESSAGE_KEY, SEQ_KEY, ConfirmedRules, confirmed_sentences

SYSTEM = {"role": "system", "content": "sys"}
RULE_1 = "The recoloring mechanic is confirmed: a shape touching a slot takes its colour."
RULE_2 = "The rule is confirmed: clicks within distance 12 move the square."


def _agent(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setenv("ARC3_CONFIRMED_RULES", "1" if enabled else "0")
    monkeypatch.setenv("ARC3_STUCK_CONTEXT_RESET", "0")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.delenv("ARC3_CONTEXT_DRAIN_TOKENS", raising=False)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    return agent


def _turn(agent, text, reasoning="", level=1, size=2000):
    agent._last_step_summary = {"level": level}
    messages = []
    for message in (
        {"role": "user", "content": text},
        {"role": "assistant", "content": "x" * size, "reasoning": reasoning},
        {"role": "tool", "tool_call_id": "t", "content": "y" * size},
    ):
        agent._append_context_message(messages, message)
    return messages


def _over_budget_by_one_turn(agent, history):
    agent._context_budget_tokens = agent._estimate_request_input_tokens([SYSTEM, *history]) - 10


def _cleared_up_to(agent, level):
    """The game has moved past `level`, so its confirmed rules can be shown."""
    agent._last_step_summary = {"level": level + 1}


def test_sentence_filter():
    text = (
        "Let me look. " + RULE_1 + " Confirmed. Confirmed again.\n"
        "If the plus turns orange, the mechanic is confirmed. "
        "Is the goal confirmed by this frame? - " + RULE_2 + "\n"
        "The wall is not confirmed as solid yet."
    )
    assert confirmed_sentences(text) == [RULE_1, RULE_2]
    long = "The wall is confirmed to block the red piece at " + "a" * 900 + "."
    assert len(confirmed_sentences(long)[0]) == confirmed_rules.MAX_SENTENCE_CHARS


def test_off_adds_no_tags_and_no_message(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path, enabled=False)
    history = [*_turn(agent, "first", RULE_1), *_turn(agent, "second"), *_turn(agent, "third")]
    assert not any(SEQ_KEY in m for m in history)
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert trimmed[1]["content"] == "second"


def test_no_trim_shows_no_section(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first", RULE_1), *_turn(agent, "second", RULE_2)]
    assert agent._trim_messages_for_context([SYSTEM, *history]) == [SYSTEM, *history]
    assert agent._persistent_history_messages([SYSTEM, *history]) == history
    assert agent._rules_ledger().message() is None


def test_request_trim_pins_the_dropped_rules_before_the_history(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [
        *_turn(agent, "first", RULE_1, level=2),
        *_turn(agent, "second", RULE_2, level=3),
        *_turn(agent, "third"),
    ]
    _cleared_up_to(agent, 3)
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    rules = trimmed[1]
    assert rules["role"] == "user" and RULES_MESSAGE_KEY in rules
    assert rules["content"].startswith(HEADER)
    assert "Level 2:\n- " + RULE_1 in rules["content"]
    assert RULE_2 not in rules["content"]  # its reply is still in context
    assert trimmed[2]["content"] == "second"
    # the tags never reach the server
    wire = tool_agent._strip_control_keys(trimmed)
    assert not any(RULES_MESSAGE_KEY in m or SEQ_KEY in m for m in wire)
    # stored history never holds the pinned message, and the next request
    # carries the identical one (prefix cache holds between trims)
    stored = agent._persistent_history_messages(trimmed)
    assert not any(RULES_MESSAGE_KEY in m for m in stored)
    agent._context_budget_tokens = 10**9
    again = agent._trim_messages_for_context([SYSTEM, *stored, *_turn(agent, "fourth")])
    assert again[1] == rules


def test_end_of_turn_trim_groups_by_level_and_dedupes(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [
        *_turn(agent, "first", RULE_2 + " " + RULE_1, level=3),
        *_turn(agent, "second", RULE_1.upper(), level=1),
        *_turn(agent, "third"),
        *_turn(agent, "fourth"),
    ]
    _cleared_up_to(agent, 3)
    agent._context_budget_tokens = agent._estimate_request_input_tokens([SYSTEM, *history[6:]]) + 50
    stored = agent._persistent_history_messages([SYSTEM, *history])
    assert stored[0]["content"] in ("third", "fourth")
    text = agent._rules_ledger().text()
    # level order, the duplicate (case-insensitive) kept once from its first reply
    assert text.index("Level 3:") < text.index(RULE_1)
    assert "Level 1:" not in text
    assert text.count(RULE_2) == 1


def test_only_thinking_is_read(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    turn = _turn(agent, "first")
    turn[1]["content"] = RULE_1  # reply text, not thinking
    history = [*turn, *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert trimmed[1]["content"] == "second"


def test_cap_drops_the_oldest_rules(monkeypatch):
    sentence = "Rule {} is confirmed by three probes in a row."
    # room for two of the three rules (each line costs len + 3)
    monkeypatch.setattr(confirmed_rules, "MAX_SECTION_CHARS", 2 * (len(sentence.format(0)) + 3))
    ledger = ConfirmedRules()
    for index, level in enumerate((1, 2, 3)):
        ledger.tag({"role": "assistant"}, level, sentence.format(index))
    assert ledger.harvest([], 4)
    text = ledger.text()
    assert "Rule 0" not in text and "Rule 1" in text and "Rule 2" in text


def test_transcript_logs_each_change_once(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first", RULE_1), *_turn(agent, "second"), *_turn(agent, "third")]
    _cleared_up_to(agent, 1)
    _over_budget_by_one_turn(agent, history)
    agent._trim_messages_for_context([SYSTEM, *history])
    notes = []
    agent._log_confirmed_rules(lambda label, text: notes.append((label, text)))
    agent._log_confirmed_rules(lambda label, text: notes.append((label, text)))
    assert [label for label, _ in notes] == ["CONFIRMED RULES"]
    assert RULE_1 in notes[0][1]


def test_new_game_starts_empty(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first", RULE_1), *_turn(agent, "second"), *_turn(agent, "third")]
    _cleared_up_to(agent, 1)
    _over_budget_by_one_turn(agent, history)
    agent._trim_messages_for_context([SYSTEM, *history])
    assert agent._rules_ledger().message() is not None
    agent._ensure_session(tmp_path / "other" / "state.json")
    assert agent._rules_ledger().message() is None


def test_control_pruning_alone_releases_nothing(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first"), *_turn(agent, "second")]
    spin_out = [
        tool_agent._mark_control_message({"role": "user", "content": "nudge"}, "nudge"),
        {"role": "assistant", "content": "z", "reasoning": RULE_1},
    ]
    agent._tag_confirmed_rules(spin_out[1])
    monkeypatch.setattr(tool_agent, "_any_control_pruning_enabled", lambda: True)
    monkeypatch.setattr(tool_agent, "_prune_control_messages", lambda messages: [messages[0], *history])
    assert agent._persistent_history_messages([SYSTEM, *history, *spin_out]) == history
    assert agent._rules_ledger().message() is None


REFUTED_1 = "The notch-relative model is refuted by the rotate test: the plug moved right."
REFUTED_2 = "U is ruled out for all three bars."


def test_refuted_filter_needs_strong_words_and_no_hedge():
    text = (
        REFUTED_1 + " So " + REFUTED_2 + "\n"
        "My model of the slide was wrong: the white cell stops the player.\n"  # weak word
        "That contradicts my earlier reading of the board entirely.\n"  # weak word
        "So the wrap hypothesis is NOT ruled out by level 1!\n"
        "Let me click A again to rule out a coordinate issue.\n"
        "No, that wouldn’t disprove the all-eight rule at all.\n"
        "Hmm, but it may rule out valid plans for this level.\n"
        "That will reveal the structure and support/refute the hypothesis.\n"
        "So the moves are absolute \u2713\u2713 REFUTED definitively.\n"  # verdict of what?
        "The trail model is strongly supported (28 steps, including a refutation of greedy).\n"
        "So rule (b') is refuted."  # a label only
    )
    assert confirmed_rules.refuted_sentences(text) == [REFUTED_1, REFUTED_2]


def test_both_lists_by_level(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_REFUTED_RULES", "1")
    agent = _agent(monkeypatch, tmp_path)
    both = "I refuted the dynamic model, and the absolute moves are confirmed by two probes."
    history = [
        *_turn(agent, "first", RULE_1 + " " + REFUTED_1 + " " + both, level=2),
        *_turn(agent, "second", REFUTED_2, level=1),
        *_turn(agent, "third"),
        *_turn(agent, "fourth"),
    ]
    _cleared_up_to(agent, 2)
    agent._context_budget_tokens = agent._estimate_request_input_tokens([SYSTEM, *history[6:]]) + 50
    agent._persistent_history_messages([SYSTEM, *history])
    assert agent._rules_ledger().text() == "\n\n".join([
        confirmed_rules.HEADER_BOTH,
        confirmed_rules.INTRO_BOTH,
        "Level 1:\nRuled out:\n- " + REFUTED_2,
        "Level 2:\nConfirmed:\n- " + RULE_1 + "\n- " + both + "\nRuled out:\n- " + REFUTED_1,
    ])


def test_refuted_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_REFUTED_RULES", "1")
    agent = _agent(monkeypatch, tmp_path, enabled=False)
    history = [*_turn(agent, "first", RULE_1 + " " + REFUTED_2), *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert trimmed[1]["content"].endswith("Level 1:\nRuled out:\n- " + REFUTED_2)
    assert RULE_1 not in trimmed[1]["content"]


def test_confirmed_only_layout_is_unchanged(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first", RULE_1 + " " + REFUTED_1, level=2), *_turn(agent, "second"), *_turn(agent, "third")]
    _cleared_up_to(agent, 2)
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert trimmed[1]["content"] == "\n\n".join(
        [HEADER, confirmed_rules.INTRO, "Level 2:\n- " + RULE_1]
    )


def test_content_filter_drops_empty_sentences_and_tidies_the_rest():
    text = (
        "So the target is confirmed.\n"
        "That's confirmed by the win.\n"
        "So max 2 confirmed ✗✗✗.\n"
        "Good enough ✓ CCW confirmed.\n"
        "So white = wall, confirmed.\n"
        "The next probe goes left.\n"
        "So the rule set is confirmed:\n- pushing moves the box one cell\n- walls stop the box"
    )
    assert confirmed_sentences(text, content_filter=False) == [
        "So the target is confirmed.",
        "That's confirmed by the win.",
        "So max 2 confirmed ✗✗✗.",
        "Good enough ✓ CCW confirmed.",
        "So white = wall, confirmed.",
        "So the rule set is confirmed:",
    ]
    assert confirmed_sentences(text) == [
        "White = wall, confirmed.",
        "The rule set is confirmed: pushing moves the box one cell walls stop the box",
    ]
    assert confirmed_rules.refuted_sentences("So the dynamic model is refuted ✓.") == []
    assert confirmed_rules.refuted_sentences("So U is ruled out for all three.") == [
        "U is ruled out for all three."
    ]


def test_confirmed_rules_wait_until_their_level_is_cleared(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_REFUTED_RULES", "1")
    agent = _agent(monkeypatch, tmp_path)
    history = [
        *_turn(agent, "first", RULE_1 + " " + REFUTED_1, level=2),
        *_turn(agent, "second"),
        *_turn(agent, "third"),
    ]
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    # still on level 2: the ruled-out line shows at once, the confirmed one waits
    assert REFUTED_1 in trimmed[1]["content"]
    assert RULE_1 not in trimmed[1]["content"]
    version = agent._rules_ledger().version

    # the level is cleared, but the section stays the same until the next trim
    _cleared_up_to(agent, 2)
    agent._context_budget_tokens = 10**9
    stored = agent._persistent_history_messages(trimmed)
    again = agent._trim_messages_for_context([SYSTEM, *stored])
    assert again[1] == trimmed[1]
    assert agent._rules_ledger().version == version

    # that trim drops a reply with no sentences of its own and still releases it
    _over_budget_by_one_turn(agent, stored)
    after = agent._trim_messages_for_context([SYSTEM, *stored])
    assert "Level 2:\nConfirmed:\n- " + RULE_1 + "\nRuled out:\n- " + REFUTED_1 in after[1]["content"]
    assert agent._rules_ledger().version == version + 1


def test_no_message_while_only_uncleared_rules_exist():
    ledger = ConfirmedRules()
    ledger.tag({"role": "assistant"}, 3, RULE_1)
    assert ledger.harvest([], 3) is False
    assert ledger.message() is None
    assert ledger.harvest([], 3) is False  # nothing new
    assert ledger.harvest([], 4) is True
    assert RULE_1 in ledger.text()
    assert ledger.harvest([], 5) is False  # moved once


def test_a_line_from_a_cleared_level_shows_at_once():
    ledger = ConfirmedRules()
    ledger.tag({"role": "assistant"}, 1, RULE_1)
    assert ledger.harvest([], 2) is True
    assert RULE_1 in ledger.text()


def test_held_rules_are_deduped_once():
    ledger = ConfirmedRules()
    ledger.tag({"role": "assistant"}, 1, RULE_1)
    ledger.tag({"role": "assistant"}, 1, RULE_1)
    ledger.harvest([], 1)
    ledger.harvest([], 2)
    assert ledger.text().count(RULE_1) == 1


def test_reset_clears_held_rules():
    ledger = ConfirmedRules()
    ledger.tag({"role": "assistant"}, 1, RULE_1)
    ledger.harvest([], 1)
    ledger.reset()
    assert ledger.harvest([], 2) is False
    assert ledger.message() is None


def test_hedged_next_sentence_drops_a_confirmed_line():
    assert confirmed_sentences(RULE_1 + " Hmm, but the second box did not move.") == []
    assert confirmed_sentences(RULE_1 + " Let me double-check that.") == []
    assert confirmed_sentences(RULE_1 + " Now the next probe goes left.") == [RULE_1]
    assert confirmed_sentences(RULE_1) == [RULE_1]  # nothing follows


def test_filters_drop_useless_lines():
    for name, line in (
        ("generic_artifact", "So the rotation model is confirmed."),
        ("generic_artifact", "My plan is now confirmed by the last frame."),
        ("fragment", "I've now confirmed all buttons' effects:"),
        ("fragment", "(17,2) unreachable ✓ confirmed."),
        ("snapshot", "Fresh state confirmed: P (2,3), O (9,6), bar 64."),
        ("snapshot", "Confirmed: the plus is at rows 15-23, cols 6-14."),
        ("snapshot", "Confirmed: at steps 2 and 3, cell (0,1) is black (no gray)."),
        ("dead_coords", "Confirmed: row 4 col 2 = M (wall), col 3 = B (open)."),
        ("dead_layout", "The maze map is confirmed with 3 rooms."),
        ("dead_literal", "B = {r+c<=9} level 2 confirmed it."),
    ):
        assert name in confirmed_rules.confirmed_line_filters(line, None), line


def test_filters_keep_useful_lines():
    for line in (
        RULE_1,
        RULE_2,
        "Confirmed: the arm PUSHES beads right (b went from slot2 to slot3).",
        "✓ Confirmed: the flowers are lethal.",
        "White = wall, confirmed.",
        "Clicking a red block REMOVES it, confirmed at step 180.",
        "Dots move rigidly with the shape (now at col 15), confirmed.",
        "Confirmed: the push rule applies to every bead on the row.",
    ):
        assert confirmed_rules.confirmed_line_filters(line, None) == [], line


def test_uppercase_and_impossibility_lines_stay_for_cleared_levels():
    # these only predicted errors on a live level, and the section no longer shows one
    assert confirmed_rules.confirmed_line_filters("The sparks are harmless ✓✓ CONFIRMED twice more.", None) == []
    assert confirmed_rules.confirmed_line_filters("The chamber is a dead end, CONFIRMED by the bar.", None) == []


def test_refuted_lines_ignore_the_new_filters():
    # same sentence shape the filters drop on the confirmed side; "as before" for ruled-out lines
    assert confirmed_rules.refuted_sentences(REFUTED_1 + " Hmm, but the second box did not move.") == [REFUTED_1]
    line = "Column 4 at row 2 is ruled out for the green bar."
    assert confirmed_rules.refuted_sentences(line) == [line]
