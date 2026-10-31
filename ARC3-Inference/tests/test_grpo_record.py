"""GRPO mode's records and samples (inference/grpo/rl_record.py, episodes.py): what the harness writes per request
and per episode, which reply tokens a sample trains, and the episode's end."""

import base64
import io
import json
import random

from inference.grpo import episodes, rl_record


def png(width=64, height=64) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_common_prefix_matches_a_direct_scan():
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randrange(4) for _ in range(rng.randrange(40))]
        b = a[:rng.randrange(len(a) + 1)] + [rng.randrange(4) for _ in range(rng.randrange(10))]
        expected = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        assert rl_record.common_prefix(a, b) == expected


def test_budget_follows_the_human_actions_of_the_played_levels():
    assert rl_record.episode_budget([21, 48, 30]) == 1750 * 69  # bp35: 120,750
    assert rl_record.episode_budget([78, 175, 50]) == 130_000  # g50t: the cap
    assert rl_record.episode_budget(None) == 130_000


def test_payload_asks_for_ids_and_logprobs_and_only_lowers_max_tokens():
    payload = {"max_tokens": 14336}
    rl_record.EpisodeRecorder.augment(payload, 9000)
    assert payload == {"max_tokens": 9000, "logprobs": True, "top_logprobs": 0, "return_token_ids": True}
    payload = {"max_tokens": 2000}
    rl_record.EpisodeRecorder.augment(payload, 9000)
    assert payload["max_tokens"] == 2000


def recorder(tmp_path, levels, budget=20_000):
    return rl_record.EpisodeRecorder(tmp_path / "rl" / "ls20-abc_p0.jsonl", budget, lambda: levels[0], lambda: 7)


def response(prompt, ids, logprob=-0.1):
    return {"prompt_token_ids": prompt, "choices": [{"token_ids": ids, "logprobs": {
        "content": [{"token": f"token_id:{t}", "logprob": logprob} for t in ids]}}]}


def test_episode_ends_at_the_levels_to_play_or_when_the_budget_cannot_hold_a_request(tmp_path):
    levels = [0]
    rec = recorder(tmp_path, levels)
    assert not rec.should_stop()
    assert rec.request_cap(10_000) == 20_000 - 10_000 - rl_record.REQUEST_MARGIN
    levels[0] = 2
    assert rec.should_stop() and rec.stop_reason == "cleared 2 levels"
    rec2 = recorder(tmp_path, [0])
    assert rec2.request_cap(19_000) is None and rec2.should_stop() and rec2.stop_reason == "budget"


def test_records_chain_and_episode_line(tmp_path):
    levels = [0]
    rec = recorder(tmp_path, levels)
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url",
                                                                          "image_url": {"url": png()}}]}]
    p1, r1 = [1, 2, 3, 4], [5, 6, 7]
    rec.after_response(rl_record.EpisodeRecorder.extract(response(p1, r1)), messages=messages, finish_reason="stop",
                       max_tokens=100, usage={})
    p2, r2 = p1 + [5, 6, 9, 10], [11, 12]  # the reply came back with its last token changed
    rec.after_response(rl_record.EpisodeRecorder.extract(response(p2, r2)), messages=messages, finish_reason="stop",
                       max_tokens=100, usage={})
    levels[0] = 1

    class Run:
        game_id, levels_completed, number_of_levels = "ls20-abc", 1, 7
        actions_per_level, base_actions_per_level, state, solver_note = [5, 2, 0], [22, 123, 73], "gave_up", None

    rec.finish(Run())
    rows = [json.loads(line) for line in rec.path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["request", "request", "episode"]
    assert rows[1]["lcp"] == 6 and rows[1]["previous_len"] == 7
    assert rows[2]["final_prompt_ids"] == p2 and rows[2]["images"] == 1 and rows[2]["actions_per_level"][0] == 5
    assert (rec.image_dir / "0000.png").exists()


def write_episode(path, requests, final_prompt, images=0, thinking_budget=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(kind="request", i=i, **r) for i, r in enumerate(requests)]
    rows.append(dict(kind="episode", game_id="ls20-abc", levels_completed=1, actions_per_level=[5, 0],
                     base_actions_per_level=[22, 123], final_prompt_ids=final_prompt, stop_reason="budget"))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    image_dir = path.with_suffix("")
    image_dir.mkdir(exist_ok=True)
    for i in range(images):
        (image_dir / f"{i:04d}.png").write_bytes(b"png")


def req(prompt_len, ids, lcp, level=1, logprobs=None):
    return dict(prompt_len=prompt_len, ids=ids, logprobs=logprobs or [-0.5] * len(ids), lcp=lcp, level=level,
                previous_len=0, finish="stop")


def test_a_reply_is_trained_up_to_where_a_later_prompt_differs(tmp_path):
    t = episodes.THINK_END
    # request 0: prompt 0..9, reply 10..14; request 1 keeps only 3 of its tokens; request 2 keeps all of request 1's
    requests = [req(10, [100, 101, 102, t, 104], lcp=0),
                req(18, [200, t, 202], lcp=13),
                req(25, [300, 301], lcp=21, level=2)]
    final = list(range(10)) + [100, 101, 102] + [7] * 5 + [200, t, 202] + [8] * 4
    path = tmp_path / "rl" / "ls20-abc_p0.jsonl"
    write_episode(path, requests, final)
    ep = episodes.load(path, thinking_budget=10240)
    assert ep.error is None
    assert ep.targets == [10, 11, 12, 18, 19, 20, 25, 26]
    assert ep.ids == final + [300, 301]
    assert ep.stats["untrained_cut"] == 2 and ep.stats["replies_whole"] == 2
    assert ep.stats["thinking_per_level"] == {1: 3 + 1, 2: 2}


def test_a_forced_think_end_is_not_trained(tmp_path):
    t = episodes.THINK_END
    requests = [req(4, [50, 51, 52, t, 54], lcp=0, logprobs=[-0.1, -0.1, -0.1, 0.0, -0.1])]
    path = tmp_path / "rl" / "ls20-abc_p0.jsonl"
    write_episode(path, requests, [1, 2, 3, 4])
    ep = episodes.load(path, thinking_budget=3)  # the reply's thinking reached the budget: </think> was forced
    assert ep.targets == [4, 5, 6, 8] and ep.stats["untrained_forced"] == 1


def test_image_files_must_match_the_ids(tmp_path):
    path = tmp_path / "rl" / "ls20-abc_p0.jsonl"
    v = episodes.VISION_START
    write_episode(path, [req(6, [9], lcp=0)], [1, v, episodes.IMAGE_PAD, 2, v, episodes.IMAGE_PAD], images=1)
    assert "images" in episodes.load(path).error
