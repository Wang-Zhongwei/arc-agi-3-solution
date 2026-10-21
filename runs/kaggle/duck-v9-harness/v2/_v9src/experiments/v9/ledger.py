"""The immutable transition ledger: ground truth for everything else in v9.

One `Record` per executed action, appended from exactly one place (`Ledger.append`). Nothing
else may mutate the ledger or a record once it is in: the ledger overrides `__setattr__` after
construction, and every appended record has its class swapped to a frozen subclass so
attribute assignment on it raises too. Everything the model sees about the past is a
re-derivation from these rows; the fingerprint lets a later reader check the rows are the
ones that were written.

Stdlib only.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any, Iterable, Iterator, Optional

from v9.types import (
    CellDelta, Claim, Comparison, Delta, Event, ObjectDelta, Prediction, Record,
)

__all__ = ["Ledger", "FrozenRecord", "LedgerMutationError", "record_to_json", "record_from_json"]


class LedgerMutationError(AttributeError):
    """Raised on any attempt to change the ledger or a record other than through append."""


class FrozenRecord(Record):
    """A Record whose attributes can no longer be assigned. `Ledger.append` swaps an appended
    record's class to this one; the dataclass fields and their values are untouched."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise LedgerMutationError(f"ledger record {self.idx} is immutable (tried to set {name!r})")

    def __delattr__(self, name: str) -> None:
        raise LedgerMutationError(f"ledger record {self.idx} is immutable (tried to delete {name!r})")


# --- serialisation ------------------------------------------------------------------------


def _plain(obj: Any) -> Any:
    """dataclasses.asdict-like conversion that also turns frozensets into sorted lists and
    enums into their values, so the result is JSON-serialisable and deterministic."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (frozenset, set)):
        return sorted(_plain(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    return obj


def record_to_json(rec: Record) -> dict[str, Any]:
    """Serialise one record to a JSON-ready dict (snapshots included when they are dataclasses)."""
    return _plain(rec)


def _tup(x: Any) -> Any:
    return tuple(x) if isinstance(x, list) else x


def _cells(xs: Iterable[Any]) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in xs)


def _claim(d: dict[str, Any]) -> Claim:
    v = d.get("value")
    if isinstance(v, list):
        v = tuple(v)
    return Claim(var=d["var"], op=d["op"], value=v, p=float(d.get("p", 1.0)),
                 hypothesis=d.get("hypothesis", "none"))


def _prediction(d: Optional[dict[str, Any]]) -> Optional[Prediction]:
    if d is None:
        return None
    return Prediction(pid=d["pid"], step=int(d["step"]), action=d["action"],
                      claims=[_claim(c) for c in d.get("claims", [])],
                      abstain=list(d.get("abstain", [])), summary=d.get("summary", ""))


def _comparison(d: Optional[dict[str, Any]]) -> Optional[Comparison]:
    if d is None:
        return None
    violated = []
    for pair in d.get("violated", []):
        c, obs = pair
        violated.append((_claim(c), _tup(obs)))
    return Comparison(matched=[_claim(c) for c in d.get("matched", [])], violated=violated,
                      abstained=[_claim(c) for c in d.get("abstained", [])],
                      unexpected=list(d.get("unexpected", [])), surprise=float(d.get("surprise", 0.0)),
                      coverage=float(d.get("coverage", 0.0)),
                      coverage_status=d.get("coverage_status", "vacuous"))


def _event(d: dict[str, Any]) -> Event:
    return Event(kind=d["kind"], tid=d["tid"], step=int(d["step"]), detail=dict(d.get("detail", {})))


def _delta(d: dict[str, Any]) -> Delta:
    cd = d["cells"]
    od = d["objects"]
    cells = CellDelta(n=int(cd["n"]), by_pair=dict(cd.get("by_pair", {})),
                      changed=_cells(cd.get("changed", [])), mask_hash=cd.get("mask_hash", ""))
    objects = ObjectDelta(
        moved=[(t, (int(dr), int(dc))) for t, (dr, dc) in od.get("moved", [])],
        recoloured=[(t, int(a), int(b)) for t, a, b in od.get("recoloured", [])],
        reshaped=list(od.get("reshaped", [])), appeared=list(od.get("appeared", [])),
        vanished=list(od.get("vanished", [])), occluded=list(od.get("occluded", [])),
        reappeared=list(od.get("reappeared", [])), consumed=list(od.get("consumed", [])),
        ambiguous=list(od.get("ambiguous", [])),
    )
    scalars = {k: (int(a), int(b)) for k, (a, b) in d.get("scalars", {}).items()}
    return Delta(cells=cells, objects=objects, scalars=scalars,
                 events=[_event(e) for e in d.get("events", [])], tags=list(d.get("tags", [])))


def _snapshot(d: Any) -> Any:
    if d is None:
        return None
    try:  # lazy: rules imports ledger-free types only, but keep the modules decoupled
        from v9.rules import Snapshot
        return Snapshot.from_json(d)
    except Exception:
        return None


def record_from_json(d: dict[str, Any]) -> Record:
    """Rebuild a Record from `record_to_json` output. Unknown keys are ignored."""
    return Record(
        idx=int(d["idx"]), step=int(d["step"]), level=int(d["level"]), action=d["action"],
        action_data={k: int(v) for k, v in d.get("action_data", {}).items()},
        before_fp=d["before_fp"], after_fp=d["after_fp"], delta=_delta(d["delta"]),
        prediction=_prediction(d.get("prediction")), comparison=_comparison(d.get("comparison")),
        outcome=dict(d.get("outcome", {})), turn=int(d.get("turn", 0)), mode=d.get("mode", ""),
        before_snap=_snapshot(d.get("before_snap")), after_snap=_snapshot(d.get("after_snap")),
    )


# --- the ledger ---------------------------------------------------------------------------


def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


class Ledger:
    """Append-only sequence of Records.

    `append` is the only mutator. `records` returns a tuple (a copy), so holders of the view
    cannot reach the underlying list. Any attribute assignment on the ledger after
    construction raises LedgerMutationError.
    """

    def __init__(self, records: Iterable[Record] = ()) -> None:
        object.__setattr__(self, "_records", [])
        object.__setattr__(self, "_sealed", True)
        for r in records:
            self.append(r)

    # -- immutability ----------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise LedgerMutationError(f"Ledger is append-only (tried to set {name!r}); use append()")

    def __delattr__(self, name: str) -> None:
        raise LedgerMutationError(f"Ledger is append-only (tried to delete {name!r})")

    # -- the one mutator -------------------------------------------------------------------

    def append(self, record: Record) -> None:
        """Append one record. Its idx must equal the current length (no gaps, no rewrites).
        The record is frozen in place: further attribute assignment on it raises."""
        if not isinstance(record, Record):
            raise TypeError(f"Ledger.append expects a Record, got {type(record).__name__}")
        n = len(self._records)
        assert record.idx == n, f"record idx {record.idx} != ledger length {n}"
        if type(record) is not FrozenRecord:
            object.__setattr__(record, "__class__", FrozenRecord)
        self._records.append(record)

    # -- read-only views -------------------------------------------------------------------

    @property
    def records(self) -> tuple[Record, ...]:
        """All records, oldest first, as a tuple."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(tuple(self._records))

    def __getitem__(self, idx: int) -> Record:
        return self._records[idx]

    def last(self) -> Optional[Record]:
        """The most recent record, or None on an empty ledger."""
        return self._records[-1] if self._records else None

    def since(self, level: int) -> tuple[Record, ...]:
        """Records taken at `level` or later."""
        return tuple(r for r in self._records if r.level >= level)

    def at_level(self, level: int) -> tuple[Record, ...]:
        """Records taken at exactly `level`."""
        return tuple(r for r in self._records if r.level == level)

    def by_action(self, action: str) -> tuple[Record, ...]:
        """Records whose action is `action`."""
        return tuple(r for r in self._records if r.action == action)

    def level_boundaries(self) -> list[int]:
        """Indices of the records whose outcome reports `level_completed`."""
        return [r.idx for r in self._records if r.outcome.get("level_completed")]

    def fingerprint(self) -> str:
        """FNV-1a 64-bit over the (idx, action, before_fp, after_fp) rows, hex. Two ledgers with
        the same rows in the same order have the same fingerprint."""
        h = 0xCBF29CE484222325
        for r in self._records:
            row = f"{r.idx}|{r.action}|{r.before_fp}|{r.after_fp}\n".encode()
            for b in row:
                h ^= b
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"{h:016x}"

    # -- persistence -----------------------------------------------------------------------

    def to_jsonl(self, path: str) -> None:
        """Write one JSON object per record (dataclasses.asdict shape; frozensets as sorted
        lists, enums as values)."""
        with open(path, "w", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(record_to_json(r), sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "Ledger":
        """Read a ledger written by `to_jsonl`. Rows are appended in file order, so the idx
        invariant is re-checked on load."""
        out = cls()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(record_from_json(json.loads(line)))
        return out

    def __repr__(self) -> str:
        return f"Ledger(n={len(self._records)}, fp={self.fingerprint()})"
