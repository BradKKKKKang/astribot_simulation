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
import base64
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://decision.vllm-sr.ai/v1/systemone"
QUESTION_ID = "nearest_object"
QUESTION = "机器人的手臂最接近桌面上的哪个物体?"
CHOICES = {"red": "红色方块", "blue": "蓝色方块", "yellow": "黄色方块"}


def build_payload(image_path: Path, model: str) -> dict:
    """Encode the original image without adding any answer hints."""
    data = image_path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("图片必须是非空的 PNG、JPEG 或 WebP 文件。")
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "model": model,
        "state": "请根据所附图片回答问题。",
        "images": [f"data:{mime};base64,{encoded}"],
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION,
                "criteria": CHOICES.copy(),
            }
        },
    }


def ask_model(payload: dict, api_key: str, timeout: float) -> dict:
    """Submit one request; leave retries to the caller."""
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").replace(api_key, "[REDACTED]")
        raise RuntimeError(f"API 返回 HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 API: {exc.reason}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="待识别的本地图片路径")
    parser.add_argument("--model", default="vllm-sr/d3-turbo", help="模型 ID")
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
        answers = result.get("answers", {}) if isinstance(result, dict) else {}
        answer = answers.get(QUESTION_ID, {}) if isinstance(answers, dict) else {}
        if not isinstance(answer, dict) or answer.get("choice") not in CHOICES:
            raise ValueError("响应中没有有效的选择答案，请查看上方原始响应。")
        print(f"\n问题: {QUESTION}")
        print(f"模型选择: {CHOICES[answer['choice']]}")
        probabilities = answer.get("probabilities", {})
        if isinstance(probabilities, dict):
            for key, label in CHOICES.items():
                probability = probabilities.get(key)
                if isinstance(probability, (int, float)):
                    print(f"  {label}: {probability:.2%}")
        print(f"请求耗时: {elapsed:.2f} 秒")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
