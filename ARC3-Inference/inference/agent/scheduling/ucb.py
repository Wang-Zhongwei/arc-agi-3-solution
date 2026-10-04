"""Yield-UCB: rank games by observed score yield plus a UCB exploration bonus.

An alternative to scheduling.priority's (A + B) * C estimate. Instead of
predicting what the current level is worth, it measures what each game has
actually returned per generated token and adds a bandit bonus that shrinks as
the game accumulates actions (or tokens) and grows slowly with shared wall
time. Ported from duck-harness's `ucb` allocation policy, minus its time
quanta: games still switch only when context trimming has already invalidated
the prefix, so scheduling never throws away a warm KV cache.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Generated tokens covered by the recent progress rate.
RECENT_WINDOW_TOKENS = 32_000

# Coefficient defaults per exploration denominator, in priority * sqrt(count).
UCB_DEFAULT_C = {"actions": 0.0002, "tokens": 0.005}

# Yield scores are tiny floats (weighted levels per token, ~1e-5..1e-3), but
# the gate and the server's `priority` field take integers below one priority
# band. x / (x + scale) is monotone, so squashing keeps the order.
RANK_SCALE = 1e-4
RANK_MAX = 999_999


def weighted_levels(levels_completed: int) -> float:
    """Unnormalized competition weight of the levels completed so far."""
    n = max(0, int(levels_completed))
    return n * (n + 1) / 2


def resolve_ucb_c(denominator: str, c: float | None) -> float:
    """Validate UCB settings and resolve an omitted coefficient in its units."""
    if denominator not in UCB_DEFAULT_C:
        raise ValueError("ARC3_UCB_DENOMINATOR must be 'actions' or 'tokens'")
    if c is None:
        return UCB_DEFAULT_C[denominator]
    if not math.isfinite(c) or c < 0:
        raise ValueError("ARC3_UCB_C must be finite and nonnegative")
    return c


@dataclass
class YieldProgress:
    """Weighted-level samples at lifetime generated-token positions."""
    samples: list[tuple[int, float]] = field(default_factory=list)

    def observe(self, tokens: int, weighted: float) -> None:
        # Only reward changes matter to either rate, and each must stay at the
        # token position where it happened, or the recent window would see an
        # old gain as new. One sample per level-up.
        sample = (max(0, int(tokens)), float(weighted))
        if self.samples and sample[1] == self.samples[-1][1]:
            return
        if self.samples and sample[0] == self.samples[-1][0]:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)

    def rate(self, tokens: int, recent_weight: float = 0.5) -> float:
        """(1 - w) * lifetime rate + w * rate over the last 32k tokens."""
        tokens = max(0, int(tokens))
        if not self.samples or tokens == 0:
            return 0.0
        weighted = self.samples[-1][1]
        cutoff = tokens - RECENT_WINDOW_TOKENS
        past = 0.0
        for sample_tokens, sample_weighted in self.samples:
            if sample_tokens > cutoff:
                break
            past = sample_weighted
        lifetime = weighted / tokens
        recent = max(0.0, weighted - past) / min(tokens, RECENT_WINDOW_TOKENS)
        w = min(1.0, max(0.0, recent_weight))
        return (1.0 - w) * lifetime + w * recent


@dataclass(frozen=True)
class YieldSnapshot:
    """What the gate needs to re-score a waiting game at any later time.

    The rate and count are fixed while the game waits; only the shared clock
    moves, which is what lets a long-waiting game's bonus catch up.
    """
    rate: float
    count: int
    c: float


def ucb_value(snapshot: YieldSnapshot, elapsed_minutes: float) -> float:
    if snapshot.count <= 0:
        return math.inf
    bonus = snapshot.c * math.sqrt(
        math.log1p(max(0.0, elapsed_minutes)) / snapshot.count
    )
    return snapshot.rate + bonus


def ucb_rank(value: float) -> int:
    """Map a Yield-UCB value onto the gate's integer scale, below one band."""
    if math.isinf(value):
        return RANK_MAX
    value = max(0.0, value)
    return max(1, min(RANK_MAX, 1 + int((RANK_MAX - 1) * value / (value + RANK_SCALE))))
