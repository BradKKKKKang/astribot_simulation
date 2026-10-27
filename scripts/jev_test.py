#!/usr/bin/env python3
"""Ask Decision 3.0 which block is closest to the robot arm.

Uses only the Python standard library; no ROS or simulation setup is needed.
Set DECISION_API_KEY in your environment, then run:
    python3 scripts/jev_test.py --image /path/to/photo.png
Preview the request without a key or network access:
    python3 scripts/jev_test.py --image /path/to/photo.png --dry-run

API reference: https://decision.vllm-sr.ai/docs
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.placement_analysis.jev import (ENDPOINT, QUESTION_ID, QUESTION, CHOICES,
                                       DEFAULT_MODEL, build_payload, ask_model, select_highest_probability)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="待识别的本地图片路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型 ID")
    parser.add_argument("--timeout", type=float, default=60.0, help="网络超时秒数，默认 60")
    parser.add_argument("--dry-run", action="store_true", help="仅检查并预览请求，不调用 API")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须是有限的正数")

    try:
        payload = build_payload(args.image.expanduser(), args.model)
        if args.dry_run:
            preview = dict(payload)
            preview["images"] = [payload["images"][0].split(",", 1)[0] + ",<omitted>"]
            print(json.dumps(preview, ensure_ascii=False, indent=2))
            print("本地检查通过，未发送 API 请求。")
            return 0

        api_key = os.environ.get("DECISION_API_KEY", "").strip()
        if not api_key:
            raise ValueError("请先设置环境变量 DECISION_API_KEY。")
        started = time.perf_counter()
        result = ask_model(payload, api_key, args.timeout)
        elapsed = time.perf_counter() - started
        print("原始响应:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        choice, probabilities = select_highest_probability(result, QUESTION_ID, CHOICES)
        print(f"\n问题: {QUESTION}")
        print(f"模型选择（最高概率）: {CHOICES[choice]}")
        for key, label in CHOICES.items():
            print(f"  {label}: {probabilities[key]:.2%}")
        print(f"请求耗时: {elapsed:.2f} 秒")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
