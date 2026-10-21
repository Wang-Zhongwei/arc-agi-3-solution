"""Zero-dependency typed graph runner.

Nodes are pure-ish functions over one `TurnState` object; edges are functions from state to
the next node name. The shape is LangGraph's (nodes, conditional edges, a terminal), so a
LangGraph runner could be swapped in later; this one is ~50 lines and adds no latency or
dependency to the Kaggle bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

Node = Callable[[Any], Any]
Edge = Callable[[Any], str]

END = "end"


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    entry: str = ""

    def add(self, name: str, fn: Node, next_of: Edge | str) -> "Graph":
        self.nodes[name] = fn
        self.edges[name] = (lambda _s, _n=next_of: _n) if isinstance(next_of, str) else next_of
        if not self.entry:
            self.entry = name
        return self

    def check(self) -> None:
        for name, edge in self.edges.items():
            if name not in self.nodes:
                raise ValueError(f"edge from unknown node {name}")
        if self.entry not in self.nodes:
            raise ValueError("graph has no entry node")

    def run(self, state: Any, *, start: str | None = None, max_steps: int = 32) -> Any:
        """Run until END. Every visited node name is appended to state.trace."""
        node = start or self.entry
        steps = 0
        while node != END:
            if steps >= max_steps:
                raise RuntimeError(f"graph exceeded {max_steps} steps at node {node}")
            fn = self.nodes[node]
            state.trace.append(node)
            state = fn(state)
            node = self.edges[node](state)
            steps += 1
        return state
