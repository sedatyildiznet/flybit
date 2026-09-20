from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybit.assays import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Flybit behavior assays")
    parser.add_argument("--output", type=Path, default=Path("build/behavior-assays.json"))
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()
    report = write_report(args.output, seed=args.seed, duration=args.duration)
    print(f"wrote {len(report['assays'])} assays to {args.output}")


if __name__ == "__main__": main()
