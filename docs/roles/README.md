# Roles for the next couple of days

We will evaluate ideas from dfranzen, lordhansolo, sirikilohit, and our own implementation, then integrate the strongest tested changes into a shared baseline.


## Proposed roles
1. **Infrastructure Engineer:** Review serving configs and patches; Find global optima under different concurrency/context tradeoffs and strategies
2. **Strategy Engineer:** Tune compute allocation, context management, prompts, and memory, including how the agent transfers knowledge between levels.
3. **Harness Engineer:** Improve APIs, action feedback, Python/workspace persistence, and observations such as animations and candidate rotation/symmetry signals.

Each role should deliver a small, tested change against the same baseline. Detailed responsibilities and experiment plans are in [Infrastructure](infrastructure/README.md), [Strategy](strategy/README.md), and [Harness](harness/README.md). We will then integrate the strongest changes and evaluate them together.