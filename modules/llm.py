"""
llm.py — Unified Groq / Gemini LLM wrapper with JSON-safe parsing and retries.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Safe imports — neither SDK is required for demo mode
# ---------------------------------------------------------------------------
try:
    from groq import Groq  # type: ignore
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

try:
    import google.generativeai as genai  # type: ignore
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from config import GROQ_MODELS, GEMINI_MODELS


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_groq_client(api_key: str) -> Any:
    """Return an authenticated Groq client."""
    if not _GROQ_AVAILABLE:
        raise RuntimeError("groq package not installed.")
    return Groq(api_key=api_key)


def get_gemini_model(api_key: str, model_name: str) -> Any:
    """Return a configured Gemini GenerativeModel."""
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai package not installed.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


# ---------------------------------------------------------------------------
# Core chat completion
# ---------------------------------------------------------------------------

def chat_completion(
    provider: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    model_override: Optional[str] = None,
    max_retries: int = 2,
    temperature: float = 0.2,
) -> str:
    """
    Call the chosen LLM provider and return the assistant text.

    Args:
        provider:       'Groq' or 'Gemini'.
        api_key:        Provider API key.
        system_prompt:  System/context instructions.
        user_message:   The user turn content.
        model_override: Override default model name.
        max_retries:    Number of retry attempts on transient errors.
        temperature:    Sampling temperature.

    Returns:
        Raw text response from the model.

    Raises:
        RuntimeError: On unrecoverable API failure.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            if provider == "Groq":
                return _groq_chat(api_key, system_prompt, user_message, model_override, temperature)
            elif provider == "Gemini":
                return _gemini_chat(api_key, system_prompt, user_message, model_override, temperature)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")


def _groq_chat(
    api_key: str,
    system_prompt: str,
    user_message: str,
    model_override: Optional[str],
    temperature: float,
) -> str:
    """Internal Groq chat call."""
    client = get_groq_client(api_key)
    model = model_override or GROQ_MODELS["chat"]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=4096,
    )
    return resp.choices[0].message.content


def _gemini_chat(
    api_key: str,
    system_prompt: str,
    user_message: str,
    model_override: Optional[str],
    temperature: float,
) -> str:
    """Internal Gemini chat call."""
    model_name = model_override or GEMINI_MODELS["chat"]
    model = get_gemini_model(api_key, model_name)
    full_prompt = f"{system_prompt}\n\n{user_message}"
    resp = model.generate_content(
        full_prompt,
        generation_config={"temperature": temperature, "max_output_tokens": 4096},
    )
    return resp.text


# ---------------------------------------------------------------------------
# Vision (image) completion
# ---------------------------------------------------------------------------

def vision_completion(
    provider: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    image_base64: str,
    mime_type: str = "image/jpeg",
) -> str:
    """
    Call a vision-capable LLM with an image.

    Args:
        provider:      'Groq' or 'Gemini'.
        api_key:       Provider API key.
        system_prompt: System context.
        user_message:  Text instruction paired with the image.
        image_base64:  Base64-encoded image bytes.
        mime_type:     MIME type of the image.

    Returns:
        Model text response.
    """
    try:
        if provider == "Groq":
            return _groq_vision(api_key, system_prompt, user_message, image_base64, mime_type)
        elif provider == "Gemini":
            return _gemini_vision(api_key, system_prompt, user_message, image_base64, mime_type)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as exc:
        raise RuntimeError(f"Vision call failed: {exc}") from exc


def _groq_vision(
    api_key: str,
    system_prompt: str,
    user_message: str,
    image_base64: str,
    mime_type: str,
) -> str:
    """Groq vision call using Llama 4 Scout."""
    client = get_groq_client(api_key)
    model = GROQ_MODELS["vision"]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


def _gemini_vision(
    api_key: str,
    system_prompt: str,
    user_message: str,
    image_base64: str,
    mime_type: str,
) -> str:
    """Gemini vision call."""
    import base64
    model_name = GEMINI_MODELS["vision"]
    model = get_gemini_model(api_key, model_name)
    image_bytes = base64.b64decode(image_base64)
    resp = model.generate_content(
        [
            f"{system_prompt}\n\n{user_message}",
            {"mime_type": mime_type, "data": image_bytes},
        ],
        generation_config={"temperature": 0.1, "max_output_tokens": 2048},
    )
    return resp.text


# ---------------------------------------------------------------------------
# Whisper transcription (Groq only; Gemini has no Whisper endpoint)
# ---------------------------------------------------------------------------

def transcribe_audio(
    provider: str,
    api_key: str,
    audio_bytes: bytes,
    filename: str = "audio.wav",
) -> dict[str, Any]:
    """
    Transcribe audio using Whisper via the Groq API.

    Args:
        provider:    Must be 'Groq' for live transcription.
        api_key:     Groq API key.
        audio_bytes: Raw audio file bytes.
        filename:    Original filename (used to infer format).

    Returns:
        Dict with keys 'text' and 'segments'.

    Raises:
        RuntimeError: If provider is not Groq or call fails.
    """
    if provider != "Groq":
        raise RuntimeError("Live audio transcription requires the Groq provider.")

    if not _GROQ_AVAILABLE:
        raise RuntimeError("groq package not installed.")

    client = get_groq_client(api_key)
    import io
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    resp = client.audio.transcriptions.create(
        model=GROQ_MODELS["whisper"],
        file=audio_file,
        response_format="verbose_json",
    )

    segments = []
    if hasattr(resp, "segments") and resp.segments:
        for seg in resp.segments:
            segments.append({
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "").strip(),
            })

    return {"text": resp.text, "segments": segments}


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """
    Robustly extract the first JSON object or array from model output.

    Tries the following strategies in order:
    1. Direct json.loads on the full text.
    2. Extract content from a ```json ... ``` code block.
    3. Find the first { ... } or [ ... ] span in the text.

    Args:
        text: Raw model output string.

    Returns:
        Parsed Python object (dict or list).

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    # Strategy 1 — direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — code fence
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3 — first balanced brace/bracket
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError("No valid JSON found in model output.")


def safe_json_parse(
    text: str,
    fallback: Any,
    schema_keys: Optional[list[str]] = None,
) -> Any:
    """
    Parse JSON from model output with a safe fallback.

    Args:
        text:        Raw model output.
        fallback:    Return value on any parse failure.
        schema_keys: If provided, validates that all keys exist in the result.

    Returns:
        Parsed object, or fallback on failure.
    """
    try:
        result = extract_json(text)
        if schema_keys:
            if not isinstance(result, dict):
                return fallback
            for key in schema_keys:
                if key not in result:
                    return fallback
        return result
    except (ValueError, TypeError):
        return fallback
