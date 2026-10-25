"""
vision.py — Crime scene image analysis: detection, annotation, structured output.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    DEMO_VISION_RESULT,
    DETECTION_CLASSES,
    GROQ_MODELS,
)


# ---------------------------------------------------------------------------
# Colour palette for bounding boxes (per detection class)
# ---------------------------------------------------------------------------

_CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "person":            (0, 200, 150),
    "firearm":           (255, 50,  50),
    "edged_weapon":      (255, 140,  0),
    "blood_spatter":     (200,   0,  0),
    "forced_entry":      (255, 200,  0),
    "vehicle":           (100, 180, 255),
    "suspicious_package":(200, 100, 255),
    "evidence_marker":   (255, 255,   0),
}
_DEFAULT_COLOUR = (100, 100, 100)


# ---------------------------------------------------------------------------
# Image → base64 helper
# ---------------------------------------------------------------------------

def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """
    Convert a PIL Image to a base64-encoded string.

    Args:
        img: PIL Image object.
        fmt: Output format ('JPEG' or 'PNG').

    Returns:
        Base64-encoded image string.
    """
    buf = io.BytesIO()
    rgb = img.convert("RGB") if fmt == "JPEG" else img
    rgb.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def pil_from_upload(uploaded_file: Any) -> Image.Image:
    """
    Open an uploaded Streamlit file as a PIL Image.

    Args:
        uploaded_file: st.file_uploader result.

    Returns:
        PIL Image.

    Raises:
        ValueError: If the file cannot be opened as an image.
    """
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.load()
        return img
    except Exception as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc


# ---------------------------------------------------------------------------
# Annotation — draw bounding boxes on a PIL image
# ---------------------------------------------------------------------------

def annotate_image(img: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    """
    Draw detection bounding boxes and labels on a copy of the image.

    Args:
        img:        Source PIL Image.
        detections: List of detection dicts with keys:
                    'class', 'confidence', 'bbox' ([x1,y1,x2,y2]).

    Returns:
        New annotated PIL Image.
    """
    annotated = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw_o = ImageDraw.Draw(overlay)
    draw_a = ImageDraw.Draw(annotated)

    w, h = img.size
    box_scale = max(1, min(w, h) // 400)   # dynamic line thickness

    for det in detections:
        cls = det.get("class", "unknown")
        conf = det.get("confidence", 0.0)
        bbox = det.get("bbox", [0, 0, 50, 50])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        colour = _CLASS_COLOURS.get(cls, _DEFAULT_COLOUR)
        colour_a = (*colour, 60)  # semi-transparent fill

        # Filled overlay rectangle
        draw_o.rectangle([x1, y1, x2, y2], fill=colour_a)

        # Solid border
        for t in range(box_scale):
            draw_a.rectangle(
                [x1 + t, y1 + t, x2 - t, y2 - t],
                outline=colour,
            )

        # Label background
        label = f"{cls.replace('_', ' ').upper()} {conf:.0%}"
        font_size = max(10, min(w, h) // 40)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        try:
            bbox_text = draw_a.textbbox((x1, y1 - font_size - 4), label, font=font)
            draw_a.rectangle(bbox_text, fill=colour)
            draw_a.text((x1, y1 - font_size - 4), label, fill=(10, 14, 26), font=font)
        except Exception:
            draw_a.text((x1, max(0, y1 - 14)), label, fill=colour)

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    return annotated


# ---------------------------------------------------------------------------
# LLM-powered detection
# ---------------------------------------------------------------------------

_VISION_SYSTEM_PROMPT = """You are a forensic image analysis AI.
Analyse the provided crime scene image and return ONLY valid JSON conforming to this schema:
{
  "threat_level": "<Low|Medium|High|Critical>",
  "scene_classification": "<Burglary|Homicide|Vandalism|Assault|Robbery|Unknown>",
  "scene_caption": "<one-sentence description>",
  "detected_items": [
    {
      "class": "<person|firearm|edged_weapon|blood_spatter|forced_entry|vehicle|suspicious_package|evidence_marker>",
      "confidence": <0.0-1.0>,
      "bbox": [x1, y1, x2, y2],
      "notes": "<brief forensic note>"
    }
  ],
  "investigative_leads": ["<lead 1>", "<lead 2>", ...]
}
Rules:
- bbox values must be pixel coordinates relative to the image dimensions.
- confidence must be a float 0.0-1.0.
- Only include classes from the defined list.
- Do not include any text outside the JSON object.
- Be conservative with confidence scores; forensic accuracy is paramount.
"""


def analyse_image_llm(
    provider: str,
    api_key: str,
    img: Image.Image,
) -> dict[str, Any]:
    """
    Analyse a crime scene image using the LLM vision API.

    Args:
        provider: 'Groq' or 'Gemini'.
        api_key:  Provider API key.
        img:      PIL Image to analyse.

    Returns:
        Validated detection result dict (with 'simulated': False).

    Raises:
        RuntimeError: On API or parse failure.
    """
    from modules.llm import vision_completion, safe_json_parse

    # Resize for API efficiency (max 1024px on longest side)
    img_resized = _resize_for_api(img)
    img_b64 = image_to_base64(img_resized)
    w, h = img_resized.size

    user_msg = (
        f"Image dimensions: {w}×{h} pixels.\n"
        "Identify all forensically relevant items. "
        "Return bbox coordinates that are accurate to the image dimensions provided."
    )

    raw = vision_completion(
        provider=provider,
        api_key=api_key,
        system_prompt=_VISION_SYSTEM_PROMPT,
        user_message=user_msg,
        image_base64=img_b64,
        mime_type="image/jpeg",
    )

    required_keys = ["threat_level", "scene_classification", "detected_items"]
    result = safe_json_parse(raw, fallback=None, schema_keys=required_keys)

    if result is None:
        raise RuntimeError("LLM returned invalid JSON for vision analysis.")

    # Scale bboxes back to original image size if resized
    orig_w, orig_h = img.size
    sx, sy = orig_w / w, orig_h / h
    for item in result.get("detected_items", []):
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            item["bbox"] = [
                int(bbox[0] * sx), int(bbox[1] * sy),
                int(bbox[2] * sx), int(bbox[3] * sy),
            ]

    result["simulated"] = False
    return result


def _resize_for_api(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Resize image so the longest side is ≤ max_side, maintaining aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Demo / simulated detection
# ---------------------------------------------------------------------------

def analyse_image_demo(img: Image.Image) -> dict[str, Any]:
    """
    Return simulated detection results for demo mode.

    The bounding boxes are scaled to the actual image dimensions so the
    annotations render correctly regardless of image size.

    Args:
        img: PIL Image (used only for dimension scaling).

    Returns:
        Demo detection result dict with 'simulated': True.
    """
    import copy
    result = copy.deepcopy(DEMO_VISION_RESULT)

    w, h = img.size
    # Scale demo bboxes (designed for 512×512) to actual image dimensions
    sx, sy = w / 512, h / 512
    for item in result["detected_items"]:
        b = item["bbox"]
        item["bbox"] = [
            int(b[0] * sx), int(b[1] * sy),
            int(b[2] * sx), int(b[3] * sy),
        ]

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_image(
    img: Image.Image,
    provider: str = "",
    api_key: str = "",
    demo_mode: bool = True,
) -> dict[str, Any]:
    """
    Analyse a crime scene image, using LLM or demo fallback.

    Args:
        img:       PIL Image.
        provider:  LLM provider ('Groq' or 'Gemini').
        api_key:   API key (empty string → demo mode).
        demo_mode: Force demo mode even if key is provided.

    Returns:
        Detection result dict.
    """
    if demo_mode or not api_key:
        return analyse_image_demo(img)

    try:
        return analyse_image_llm(provider, api_key, img)
    except Exception as exc:
        # Graceful fallback
        result = analyse_image_demo(img)
        result["_api_error"] = str(exc)
        return result
