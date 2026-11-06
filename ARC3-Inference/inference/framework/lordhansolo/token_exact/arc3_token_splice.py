"""Put a reply's sampled token ids back into the prompt instead of re-tokenizing its text.

Copied into the vLLM runtime as vllm/arc3_token_splice.py by the launcher. The harness sends an assistant
message whose reasoning is a marker carrying the ids, so the template renders

    <think>\n<marker>\n</think>\n\n<|im_end|>

and this module replaces the marker through <|im_end|> with the ids. The text around the markers is tokenized
as usual and the result goes to the renderer as a token prompt, whose image placeholders vLLM expands itself.
"""
import re

MARKER_OPEN = "ARC3TOKENS:"
MARKER_CLOSE = ""
_SPAN = re.compile(re.escape(MARKER_OPEN) + r"([0-9,]*)" + re.escape(MARKER_CLOSE) + r"\n</think>\n\n<\|im_end\|>")
THINK_END = "</think>"
TURN_END = "<|im_end|>"
# A sampled media token would claim an image that was never sent; detokenizing drops these tokens from text too.
MEDIA_TOKENS = ("<|image_pad|>", "<|video_pad|>", "<|audio_pad|>")


def splice_reply_tokens(prompt, tokenizer):
    """The token prompt for a rendered prompt holding markers; any other prompt comes back unchanged."""
    if not isinstance(prompt, str) or MARKER_OPEN not in prompt:
        return prompt

    def encode(text):
        return list(tokenizer.encode(text, add_special_tokens=False)) if text else []

    think_end, turn_end = (tokenizer.convert_tokens_to_ids(token) for token in (THINK_END, TURN_END))
    media = {tokenizer.convert_tokens_to_ids(token) for token in MEDIA_TOKENS}
    pieces = []
    tokens = []
    position = 0
    for match in _SPAN.finditer(prompt):
        pieces.append(prompt[position:match.start()])
        tokens.extend(encode(pieces[-1]))
        reply = [int(value) for value in match.group(1).split(",") if value]
        reply = [token for token in reply if token not in media]
        if think_end not in reply:
            # Cut off inside the thinking: close it the way the template closes every reply.
            if reply and reply[-1] == turn_end:
                reply.pop()
            reply.extend(encode(f"\n{THINK_END}\n\n"))
        if not reply or reply[-1] != turn_end:
            reply.append(turn_end)
        tokens.extend(reply)
        position = match.end()
    pieces.append(prompt[position:])
    if any(MARKER_OPEN in piece for piece in pieces):
        raise ValueError("A reply-token marker did not render as <think>, marker, </think>, <|im_end|>")
    tokens.extend(encode(pieces[-1]))
    return tokens
