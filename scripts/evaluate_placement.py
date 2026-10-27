#!/usr/bin/env python3
"""Evaluate existing placement evidence with an explicit evaluation experiment.

This is offline evaluation, not a policy rollout runner. No replay, SDK, ROS,
USD loading or JEV requests are performed.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    from src.evaluation.placement import evaluate_bundle
    result = evaluate_bundle(args.bundle, task_config=args.experiment,
                             observations=args.observations, dry_run=args.dry_run)
    print(json.dumps(result['summary'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
