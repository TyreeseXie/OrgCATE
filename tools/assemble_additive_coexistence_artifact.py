"""Copy the public additive-coexistence module into a CI artifact directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "additive_coexistence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SOURCE,
        output,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    print(f"Assembled additive coexistence artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
