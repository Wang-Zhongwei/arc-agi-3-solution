"""Shared data model for v9. Plain dataclasses; pydantic is used only at the model boundary
(schemas.py). Every structure here is harness-owned. The model never constructs these directly:
it proposes JSON, the validators build the typed patch, and the harness applies it.

Vocabulary
- Grid: 64x64 ints 0..15 (ARC palette letters COLOUR_CHARS).
- Blob: one 4-connected same-colour component of the settled frame (perception.py).
- Track: a persistent object identity across frames, id "t<k>" (tracker.py). Identity is not
  appearance: a track survives recolour, movement and short occlusion.
- Record: one immutable ledger row per executed action (ledger.py). Ground truth.
- Rule: a typed, compiled statement scored against the ledger (rules.py). Theory.
- Claim/Prediction/Comparison: the contract around every action (predict.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

Grid = list[list[int]]
Cell = tuple[int, int]

COLOUR_CHARS = "WwgGcBMPRbSYOrNp"
COLOUR_NAMES = {
    0: "white", 1: "lightgray", 2: "gray", 3: "darkgray", 4: "charcoal", 5: "black",
    6: "magenta", 7: "pink", 8: "red", 9: "blue", 10: "skyblue", 11: "yellow", 12: "orange",
    13: "darkred", 14: "green", 15: "purple",
}
ACTION_NAMES = ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7")


def colour_letter(v: int) -> str:
    return COLOUR_CHARS[max(0, min(15, int(v)))]


# --- perception -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Blob:
    """One connected component of the settled frame."""
    colour: int
    cells: frozenset[Cell]
    bbox: tuple[int, int, int, int]      # r0, c0, r1, c1 inclusive
    pixels: int
    shape_hash: str                      # colour + normalised shape (translation invariant)
    form_hash: str                       # normalised shape only (colour-blind)
    region: str = "board"                # board | hud | background | veiled
    holes: int = 0
    # A multi-colour sprite (perception.merge_sprites) is the union of several mono blobs;
    # `parts` holds their indices in the same blob list. Empty for a mono component, which is
    # how the tracker and the scalar registry avoid counting a sprite and its parts twice.
    parts: tuple[int, ...] = ()

    @property
    def centre(self) -> tuple[float, float]:
        r0, c0, r1, c1 = self.bbox
        return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

    @property
    def size(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0 + 1, c1 - c0 + 1)


@dataclass(frozen=True)
class Band:
    """An edge HUD band: a thin strip flush against one border that changes on its own."""
    edge: str                            # top | bottom | left | right
    rows: tuple[int, int]
    cols: tuple[int, int]
    filled: int                          # non-background cells in the band
    length: int


@dataclass
class Percept:
    """Deterministic reading of one frame. `raw` keeps every animation layer."""
    step: int
    level: int
    action: Optional[str]
    raw: list[Grid]
    grid: Grid
    background: int
    hud_bands: list[Band]
    veil: frozenset[Cell]
    blobs: list[Blob]
    colour_cells: dict[int, frozenset[Cell]]
    scalars: dict[str, int]
    fingerprint: str
    transform: Optional[dict[str, Any]] = None   # {"kind": "scroll"|"palette", ...}


# --- identity ---------------------------------------------------------------------------------


class TrackState(str, Enum):
    LIVE = "LIVE"
    OCCLUDED = "OCCLUDED"
    GONE = "GONE"
    CONSUMED = "CONSUMED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class Event:
    kind: str                            # SPAWN MOVE RECOLOUR RESHAPE OCCLUDE REAPPEAR VANISH CONSUME MERGE SPLIT
    tid: str
    step: int
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    tid: str
    level: int
    first_seen: int
    last_seen: int
    blob: Optional[Blob]
    last_bbox: tuple[int, int, int, int]
    colour: int
    form_hash: str
    velocity: tuple[int, int] = (0, 0)
    state: TrackState = TrackState.LIVE
    missing_for: int = 0
    ambiguity_group: Optional[int] = None
    events: list[Event] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)   # touched_at, entered_at, changed_at, consumed_at, contacts, no_ops, blinking
    klass: str = "unknown"               # hud | background | static | autonomous | controllable | unknown
    label: Optional[str] = None          # model-proposed, harness-validated
    group: Optional[int] = None          # rigid co-moving body id (tracker._regroup), an overlay on identity
    appearances: list[tuple[int, int, tuple[int, int]]] = field(default_factory=list)  # (step, colour, (r0,c0))

    @property
    def centre(self) -> tuple[float, float]:
        r0, c0, r1, c1 = self.last_bbox
        return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


# --- deltas and the prediction contract ------------------------------------------------------


@dataclass
class CellDelta:
    n: int
    by_pair: dict[str, int]              # "B>R": count of cells that went from B to R
    changed: frozenset[Cell]
    mask_hash: str


@dataclass
class ObjectDelta:
    moved: list[tuple[str, tuple[int, int]]]       # (tid, (dr, dc))
    recoloured: list[tuple[str, int, int]]         # (tid, from, to)
    reshaped: list[str]
    appeared: list[str]
    vanished: list[str]
    occluded: list[str]
    reappeared: list[str]
    consumed: list[str]
    ambiguous: list[str]


@dataclass
class Delta:
    cells: CellDelta
    objects: ObjectDelta
    scalars: dict[str, tuple[int, int]]  # name -> (before, after) for names whose value changed
    events: list[Event]
    tags: list[str]                       # noop | blocked | hud_only | veiled | level_up | game_over
    # Live tracks that produced no event this step. Rendered as "unchanged N" on the diff
    # line; kept here because render_diff sees only the Delta.
    unchanged: int = 0


@dataclass(frozen=True)
class Claim:
    var: str                              # a name from the scalar registry, or "cells" / "obj:<tid>.pos" etc.
    op: str                               # delta | equals | unchanged | changed
    value: Any
    p: float
    hypothesis: str = "none"


@dataclass
class Prediction:
    pid: str
    step: int
    action: str
    claims: list[Claim]
    abstain: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class Comparison:
    matched: list[Claim]
    violated: list[tuple[Claim, Any]]     # (claim, observed)
    abstained: list[Claim]
    unexpected: list[str]                 # variable names that changed with no claim
    surprise: float
    coverage: float
    coverage_status: str                  # complete | partial | vacuous


@dataclass
class Record:
    """One immutable ledger row. Appended by ledger.Ledger.append only."""
    idx: int
    step: int
    level: int
    action: str
    action_data: dict[str, int]
    before_fp: str
    after_fp: str
    delta: Delta
    prediction: Optional[Prediction]
    comparison: Optional[Comparison]
    outcome: dict[str, Any]               # {"levels_completed", "level_completed", "game_over", "win", "state"}
    turn: int
    mode: str
    # Added by the evidence layer (rules.py): object/scalar snapshots of the state the action was
    # taken in and the state it produced, so a compiled rule can be re-evaluated from the record
    # alone. Type is rules.Snapshot; kept as Any here to avoid a types -> rules import cycle.
    before_snap: Optional[Any] = None
    after_snap: Optional[Any] = None


# --- theory: rules, hypotheses, questions, goals ----------------------------------------------


RULE_TYPES = ("invariant", "conditional", "autonomous", "correlation", "causal", "temporal", "goal")


@dataclass
class RuleScore:
    support: int = 0
    contradict: int = 0
    abstain: int = 0
    graded: int = 0
    conf: float = 0.0
    coverage: float = 0.0
    status: str = "vacuous"               # vacuous | thin | partial | complete
    stale: bool = False
    supporting_idx: list[int] = field(default_factory=list)
    contradicting_idx: list[int] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status in ("partial", "complete") and self.conf >= 0.7 and not self.stale


@dataclass
class Rule:
    rid: str
    type: str
    when: dict[str, Any]                  # DSL guard
    then: list[dict[str, Any]]            # DSL effects
    scope: str = "level"                  # level | game
    because: list[int] = field(default_factory=list)
    source: str = "model"                 # model | harness
    status: str = "proposed"              # proposed | active | quarantined | retired
    score: RuleScore = field(default_factory=RuleScore)
    text: str = ""
    created_turn: int = 0


@dataclass
class Hypothesis:
    hid: str
    claim: str
    discriminator: str                    # what observation would separate it from its rivals
    rule_ids: list[str] = field(default_factory=list)
    status: str = "open"                  # open | supported | refuted
    created_turn: int = 0
    evidence: list[int] = field(default_factory=list)


@dataclass
class Question:
    qid: str
    text: str
    status: str = "open"                  # open | answered | dropped
    created_turn: int = 0
    answered_turn: Optional[int] = None
    answer: str = ""


@dataclass
class GoalHypothesis:
    gid: str
    text: str
    p: float
    tested: list[int] = field(default_factory=list)   # ledger idx of tests
    status: str = "open"


@dataclass
class MemorySlot:
    """A model-authored memory structure within one of the five kinds."""
    name: str
    kind: str                             # object | transition | rule | question | goal
    fields: dict[str, str]                # field -> type in {int, str, bool, tid, step, colour}
    scope: str                            # level | game | cross_game
    reason: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    created_turn: int = 0


# --- turn state -------------------------------------------------------------------------------


class Mode(str, Enum):
    ORIENT = "ORIENT"
    EXPLORE = "EXPLORE"
    VERIFY = "VERIFY"
    EXECUTE = "EXECUTE"
    RECOVER = "RECOVER"


@dataclass
class Confidences:
    perception: float = 0.0
    control: float = 0.0
    world_model: float = 0.0
    goal: float = 0.0
    plan: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"perception": self.perception, "control": self.control, "world_model": self.world_model,
                "goal": self.goal, "plan": self.plan}
