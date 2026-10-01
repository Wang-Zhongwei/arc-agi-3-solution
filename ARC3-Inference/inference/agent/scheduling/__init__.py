"""Game scheduling strategies for the shared admission gate.

Both rank games at a context-trim handover; ARC3_SCHEDULER picks one.

- priority: (A + B) * C, an estimate of what a game can still bank.
- ucb: Yield-UCB, observed weighted levels per generated token plus an
  exploration bonus.
"""
