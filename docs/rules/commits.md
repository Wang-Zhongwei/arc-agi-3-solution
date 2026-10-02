# Rules

Each role—[Harness](../roles/harness/README.md), [Infrastructure](../roles/infrastructure/README.md), and [Strategy Solutionist](../roles/strategy/README.md)—works on its own branch. Coordinate when your changes affect another role.

1. Commit features on your separate branch.

2. Before merging tested features into `main`, write an A/B comparison report in `docs/<theme>/<report_name>.md` and create a pull request. Link the report in the PR so others can see what changed and how it performed.

3. Write the original doc yourself and let an agent help fix it. Do NOT fill it with AI slop. Include the necessary links and references.

4. A t-test result is recommended following the [guide](README.md), unless it is a no-brainer fix. For those fixes, explain what was wrong and how you checked it.

5. Run data is synced through Google Drive into `ARC3-Inference/runs/`. Keep downloaded `runs/kaggle/` artifacts out of Git; link the exact runs used in your report.

6. From `ARC3-Inference/`, run `make view PORT=8022` to inspect the data.
