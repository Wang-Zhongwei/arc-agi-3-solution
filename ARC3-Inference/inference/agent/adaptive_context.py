"""Per-game context hysteresis, independent of the serving/scheduling backend."""

from dataclasses import dataclass
import os


@dataclass
class AdaptiveContext:
    normal_low: int = 32 * 1024
    normal_high: int = 64 * 1024
    expanded_low: int = 96 * 1024
    expanded_high: int = 128 * 1024
    appended_tokens: int = 0
    last_progress_tokens: int | None = None
    completed_levels: int = 0
    expanded: bool = False
    reset_pending: bool = False

    @classmethod
    def from_env(cls, input_budget: int) -> "AdaptiveContext | None":
        if os.environ.get("ARC3_ADAPTIVE_CONTEXT", "").lower().strip() not in {
            "1", "true", "yes", "on",
        }:
            return None
        thresholds = {
            name: int(os.environ.get(f"ARC3_CONTEXT_{name.upper()}") or default)
            for name, default in (
                ("normal_low", 32 * 1024), ("normal_high", 64 * 1024),
                ("expanded_low", 96 * 1024), ("expanded_high", 128 * 1024),
            )
        }
        low, high, expanded_low, expanded_high = thresholds.values()
        if not 0 < low < high <= expanded_low < expanded_high <= input_budget:
            raise ValueError(
                "Adaptive context requires 0 < normal_low < normal_high <= "
                "expanded_low < expanded_high <= the input budget "
                f"({input_budget}, after reserving output and safety tokens)"
            )
        return cls(**thresholds)

    def record_append(self, tokens: int) -> None:
        self.appended_tokens += max(0, tokens)

    def record_progress(self, completed_levels: int) -> None:
        if completed_levels <= self.completed_levels:
            return
        self.completed_levels = completed_levels
        self.last_progress_tokens = self.appended_tokens
        if self.expanded:
            self.reset_pending = True

    def limits(self, used: int, input_budget: int, *, force_drain: bool) -> tuple[int, int]:
        # Never answer a real server overflow by expanding the allowed history.
        stalled = (
            self.last_progress_tokens is None
            or self.appended_tokens - self.last_progress_tokens >= self.normal_high
        )
        if not self.expanded and used > self.normal_high and stalled and not force_drain:
            self.expanded = True
        high = self.expanded_high if self.expanded else self.normal_high
        low = self.expanded_low if self.expanded and not self.reset_pending else self.normal_low
        return min(high, input_budget), min(low, input_budget)

    def did_trim(self) -> None:
        # Latch success until an actual trim, even after another long stall.
        if self.reset_pending:
            self.expanded = False
            self.reset_pending = False
