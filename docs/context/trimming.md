# Trimming with transition reminders

When context is trimmed, keep a short reminder of the last level-up and last game-over so the model can still remember how it succeeded or failed. Follow these reminders with the recent original transcript.

Implemented by `inference/agent/transition_reminders.py` and `inference/agent/tool_agent.py`.

## Two config options

Options under `context`:

```json
{
  "context": {
    "retain_last_level_up": true,
    "retain_last_game_over": true
  }
}
```

Each option independently includes a reminder of the latest event of that kind if trimming removed its original exchange. Both default to `false` to preserve current behavior. Setting an option to `false` disables its reminder; events still present in the recent transcript remain there.

Keep at most one level-up reminder and one game-over reminder per game. A newer event replaces the previous reminder of the same kind, even if the newer event is still in the transcript. If no such event has occurred, omit that reminder.

## What a reminder contains

A reminder is a compact record of the transition, with enough context to recall the model's approach. It does not need to replay the original exchange word for word.

- The event's action number / `history` index and level.
- The action that triggered the event, with a short lead-in from the executed action sequence when needed.
- The original reasoning that led to those actions, written before the model observed their results, with the token limit below. Exclude reflections written after seeing the level-up or game-over. Omit this if unavailable; do not invent an explanation of why the action worked or failed.
- Historical board images, labeled with their original history indices and level: one before the successful action for level-up, and both before and after the fatal action for game-over.

Keep the original pre-action reasoning verbatim, up to 2,048 tokens per reminder, measured with the model's tokenizer. If it exceeds 2,048 tokens, retain the first 1,024 and last 1,024 tokens in their original order, separated by `[Middle of reasoning omitted]`. If it fits, retain it in full. Labels and the omission marker count separately toward the overall context budget.

For a level-up, say which level was completed and which level began. Keep the board immediately before the successful action. The resulting board may already be the next level's initial board, so omit that image from the reminder. A final win is labeled as game completion.

For a game-over, keep the fatal action and both board images: immediately before it and immediately after it. The after image is the fatal board. Describe any automatic RESET separately, so the reset board cannot be mistaken for the fatal board.

Keep a brief record of the reasoning before the fatal action so the model can reconsider what it expected. Label it as potentially mistaken: "Reasoning before the fatal action (may contain mistaken assumptions)." GAME OVER alone does not show which assumptions were wrong; the failure could also come from execution or a deliberate test.

Use the executed transition records for actions, outcomes, and frames, and the associated assistant exchange for earlier reasoning. Apply the reasoning limit above and omit unrelated tool output or replay of the entire level.

## Order in the trimmed context

After the usual system and opening messages, insert one user message containing the trimming notice and any enabled reminders, ordered by when the events happened. Then append the recent original transcript, keeping complete assistant/tool exchanges intact.

For example, if the recent transcript begins after `history[120]`:

```text
[user: trimming notice and reminders]
Earlier conversation has been trimmed. These reminders preserve your latest
level-up and game-over from that conversation.

Last level-up 2 → 3
Reasoning before the actions: <original pre-action reasoning, capped at 2,048 tokens>
Executed actions: <brief lead-in and triggering action>
Before the level-up (history[99].frame): <history[99].frame>

Last GAME OVER (level 3)
Reasoning before the fatal action (may contain mistaken assumptions):
<original pre-action reasoning, capped at 2,048 tokens>
Executed actions: <brief lead-in and fatal action>
Before GAME OVER (history[109].frame): <history[109].frame>
Fatal board (history[110].frame): <history[110].frame>

[recent history]
<original messages covering the retained recent history, roughly 53k tokens>
```

The example indices refer to game history, not conversation-message positions. The omitted intervals need only a trimming notice; they have not been semantically summarized. The recent transcript keeps its original messages, roles, tool calls, results, and order, subject to existing image and reasoning retention settings.

## Interaction with the context budget

Build reminders from the latest recorded events whenever the retained tail changes. If an event's original exchange is fully retained, omit its reminder to avoid duplication. Rebuild the reminder message rather than appending another copy on each trim.

Count reminders, including their required board images, inside the existing context budget. Reserve space for them when choosing the recent tail. If space is tight, shorten the action lead-in, then trim older complete exchanges from the recent tail to make room for the reminders. Keep the event identity, triggering action, and outcome concise; never exceed the context limit or split a tool exchange to preserve a reminder.

Trimming changes the conversation sent to the model. The full runtime `history` and `transitions` remain available for inspection.

Implementation uses the serving tokenizer APIs for the reasoning cap. If tokenization is unavailable, retain the action, outcome, and boards and omit reasoning with a logged warning. Partial Unicode characters at the two cut points are omitted to keep the retained text verbatim. Reminder images use the existing board upscale and survive the ordinary history-image retention window.
