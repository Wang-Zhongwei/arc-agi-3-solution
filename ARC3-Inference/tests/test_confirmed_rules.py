"""ARC3_CONFIRMED_RULES: "confirmed" thinking sentences from trimmed replies, pinned by level."""

from inference.agent import confirmed_rules, tool_agent
from inference.agent.confirmed_rules import HEADER, RULES_MESSAGE_KEY, SEQ_KEY, ConfirmedRules, confirmed_sentences

SYSTEM = {"role": "system", "content": "sys"}
RULE_1 = "The recoloring mechanic is confirmed: a shape touching a slot takes its colour."
RULE_2 = "So the rule is confirmed: clicks within distance 12 move the square."


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


def test_sentence_filter():
    text = (
        "Let me look. " + RULE_1 + " Confirmed. Confirmed again.\n"
        "If the plus turns orange, the mechanic is confirmed. "
        "Is the goal confirmed by this frame? - " + RULE_2 + "\n"
        "The wall is not confirmed as solid yet."
    )
    assert confirmed_sentences(text) == [RULE_1, RULE_2]
    long = "So it is confirmed that " + "a" * 900 + "."
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
    assert ledger.harvest([])
    text = ledger.text()
    assert "Rule 0" not in text and "Rule 1" in text and "Rule 2" in text


def test_transcript_logs_each_change_once(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first", RULE_1), *_turn(agent, "second"), *_turn(agent, "third")]
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


REFUTED_1 = "So “g→base” is definitively refuted by level 3."
REFUTED_2 = "So U is ruled out for all three bars."


def test_refuted_filter_needs_strong_words_and_no_hedge():
    text = (
        REFUTED_1 + " " + REFUTED_2 + "\n"
        "My model of the slide was wrong: the white cell stops the player.\n"  # weak word
        "That contradicts my earlier reading of the board entirely.\n"  # weak word
        "So the wrap hypothesis is NOT ruled out by level 1!\n"
        "Let me click A again to rule out a coordinate issue.\n"
        "No, that wouldn’t disprove the all-eight rule at all.\n"
        "Hmm, but it may rule out valid plans for this level.\n"
        "That will reveal the structure and support/refute the hypothesis."
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
    agent._context_budget_tokens = agent._estimate_request_input_tokens([SYSTEM, *history[6:]]) + 50
    agent._persistent_history_messages([SYSTEM, *history])
    assert agent._rules_ledger().text() == "\n\n".join([
        confirmed_rules.HEADER_BOTH,
        confirmed_rules.INTRO_BOTH,
        "Level 1:\nRuled out:\n- " + REFUTED_2,
        "Level 2:\nConfirmed:\n- " + RULE_1 + "\nRuled out:\n- " + REFUTED_1 + "\n- " + both,
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
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert trimmed[1]["content"] == "\n\n".join(
        [HEADER, confirmed_rules.INTRO, "Level 2:\n- " + RULE_1]
    )
