# Make `import v9` work when pytest is run from the repo root (`pytest experiments/v9`), the way
# scripts/spark_test.sh runs it. The package is experiments/v9, so experiments/ goes on sys.path.
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))
