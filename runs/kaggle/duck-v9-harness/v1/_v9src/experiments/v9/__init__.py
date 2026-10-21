"""v9: a harness that owns the facts.

The harness owns perception, the transition ledger, memory, confidence numbers and the
decision to spend an action. The model owns interpretation: it proposes what things mean,
memory patches, hypotheses and actions, always as typed JSON the harness validates. Nothing
the model says becomes true without the harness checking it against the ledger.

Run it with `python -m v9.run` from the experiments/ directory (or with experiments/ on
PYTHONPATH). See README.md for the loop, the modes and the memory kinds.
"""

__version__ = "0.1.0"
