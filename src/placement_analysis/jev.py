"""Shared standard-library Decision choice client; credentials remain in memory."""
import base64
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://decision.vllm-sr.ai/v1/systemone"
QUESTION_ID = "nearest_object"
QUESTION = "机器人的手臂最接近桌面上的哪个物体?"
CHOICES = {"red": "红色方块", "blue": "蓝色方块", "yellow": "黄色方块"}
DEFAULT_MODEL = "vllm-sr/d3-turbo"
SELECTION_POLICY = "max_probability_v1"


def ordered_choices(config):
    """Preserve authored tie-breaking order across canonical JSON storage."""
    choices = config['choices']
    order = config.get('choice_order', list(choices))
    if len(order) != len(choices) or len(set(order)) != len(order) or set(order) != set(choices):
        raise ValueError('choice_order must list every candidate exactly once')
    return {key: choices[key] for key in order}


def select_highest_probability(raw, question_id, choices):
    """Select one configured candidate by probability, independent of model choice.

    Require a valid probability for every candidate so an omitted candidate
    cannot silently lose. Exact ties use configuration order, not response order.
    """
    answers = raw.get('answers', {}) if isinstance(raw, dict) else {}
    answer = answers.get(question_id, {}) if isinstance(answers, dict) else {}
    probabilities = answer.get('probabilities') if isinstance(answer, dict) else None
    if not choices or not isinstance(probabilities, dict):
        raise ValueError('JEV must return probabilities for every configured candidate')
    selected = {}
    for key in choices:
        value = probabilities.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'JEV has a missing or invalid probability for candidate {key}')
        selected[key] = float(value)
    return max(selected, key=selected.get), selected


def build_payload(image_path: Path, model: str = DEFAULT_MODEL, *,
                  question_id=QUESTION_ID, question=QUESTION, choices=None) -> dict:
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
            question_id: {
                "type": "choice",
                "instructions": question,
                "criteria": (CHOICES if choices is None else choices).copy(),
            }
        },
    }


def ask_model(payload: dict, api_key: str, timeout: float, *, endpoint=ENDPOINT) -> dict:
    """Submit one request; leave retries to the caller."""
    request = urllib.request.Request(
        endpoint,
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

