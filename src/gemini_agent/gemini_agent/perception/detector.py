"""2D boxes from a non-streaming Gemini call.

Only detect_boxes() is used elsewhere, so this can be swapped for another detector later.
"""

import json
import re

import cv2
from google.genai import types

_prompt_cache: dict = {}


def _prompt(cfg: dict) -> str:
    """The look prompt, read once from the path in config.yaml's perception block."""
    path = cfg['look_prompt']
    if path not in _prompt_cache:
        with open(path) as f:
            _prompt_cache[path] = f.read()
    return _prompt_cache[path]


def _strip_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r'^```(?:json)?\s*(.*?)\s*```$', text, re.DOTALL)
    return fence.group(1) if fence else text


def detect_boxes(client, cfg: dict, rgb, names: list) -> list:
    """-> [{"label": str, "box_2d": [...] normalized 0-1000, "depth_mm": float|None}].

    depth_mm is the model's estimate of the object's hidden extent away from the camera;
    it is a prior from object knowledge, not a measurement.
    """
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('JPEG encoding failed')

    response = client.models.generate_content(
        model=cfg['detector_model'],
        contents=[
            types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg'),
            _prompt(cfg).format(names=', '.join(names)),
        ],
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            thinking_config=types.ThinkingConfig(
                thinking_level=str(cfg['detector_thinking_level']).upper()),
        ),
    )

    try:
        raw = json.loads(_strip_fences(response.text or '[]'))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    boxes = []
    for entry in raw:  # drop anything malformed rather than failing the whole call
        if not isinstance(entry, dict):
            continue
        label = entry.get('label')
        box = entry.get('box_2d')
        if not isinstance(label, str) or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            box = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        try:  # a missing or unparseable depth estimate is not worth dropping the box over
            depth_mm = float(entry['depth_mm'])
        except (KeyError, TypeError, ValueError):
            depth_mm = None
        boxes.append({'label': label, 'box_2d': box, 'depth_mm': depth_mm})
    return boxes
