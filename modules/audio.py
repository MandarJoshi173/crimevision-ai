"""
audio.py — Witness audio transcription and forensic event extraction.
"""
from __future__ import annotations

import io
import json
import re
from typing import Any, Optional

from config import DEMO_TRANSCRIPT


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a forensic intelligence analyst specialising in witness statements and emergency call transcripts.
Given a transcript, extract structured intelligence and return ONLY valid JSON:
{
  "segments": [
    {"start": <seconds>, "end": <seconds>, "speaker": "<Dispatcher|Caller|Witness|Suspect|Officer|Unknown>", "text": "<segment text>"}
  ],
  "entities": {
    "people":    ["<description>"],
    "vehicles":  ["<description>"],
    "locations": ["<address or landmark>"],
    "weapons":   ["<description>"]
  },
  "key_events": [
    {"timestamp": "<mm:ss>", "event": "<concise forensic event description>"}
  ]
}
Rules:
- speaker must be one of the allowed values.
- timestamp must be mm:ss format.
- Do not include any text outside the JSON.
- Be concise; focus on forensically relevant information.
"""


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    provider: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Transcribe audio via Groq Whisper API.

    Args:
        audio_bytes: Raw bytes of the audio file.
        filename:    Original filename (helps Groq detect format).
        provider:    Must be 'Groq' for live transcription.
        api_key:     Groq API key.

    Returns:
        Dict with 'text' (str) and 'segments' (list).

    Raises:
        RuntimeError: If transcription fails.
    """
    from modules.llm import transcribe_audio as llm_transcribe
    return llm_transcribe(provider, api_key, audio_bytes, filename)


# ---------------------------------------------------------------------------
# Event extraction from transcript
# ---------------------------------------------------------------------------

def extract_events_from_transcript(
    raw_text: str,
    segments: list[dict[str, Any]],
    provider: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Use the LLM to extract structured forensic events from a transcript.

    Args:
        raw_text:  Full transcript text.
        segments:  List of timed segment dicts.
        provider:  LLM provider.
        api_key:   API key.

    Returns:
        Extracted entity/event dict.

    Raises:
        RuntimeError: On LLM or parse failure.
    """
    from modules.llm import chat_completion, safe_json_parse

    segment_text = "\n".join(
        f"[{_fmt_time(s.get('start', 0))}] {s.get('speaker', '?')}: {s.get('text', '')}"
        for s in segments
    ) or raw_text

    user_msg = f"Transcript to analyse:\n\n{segment_text}"
    raw = chat_completion(
        provider=provider,
        api_key=api_key,
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
        user_message=user_msg,
    )

    required_keys = ["entities", "key_events"]
    result = safe_json_parse(raw, fallback=None, schema_keys=required_keys)
    if result is None:
        raise RuntimeError("LLM returned invalid JSON for transcript extraction.")

    return result


def _fmt_time(seconds: float) -> str:
    """Convert seconds to mm:ss string."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Speaker-tag enrichment
# ---------------------------------------------------------------------------

def enrich_segments_with_speakers(
    raw_transcript: dict[str, Any],
    extracted: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Merge Whisper segments with LLM-inferred speaker tags.

    Args:
        raw_transcript: Output of transcribe_audio (has 'segments').
        extracted:      Output of extract_events_from_transcript.

    Returns:
        Merged list of segment dicts with speaker fields.
    """
    base_segments = raw_transcript.get("segments", [])
    llm_segments = extracted.get("segments", [])

    # Build a lookup by start time (tolerance 2s)
    speaker_map: dict[int, str] = {}
    for seg in llm_segments:
        start = int(seg.get("start", 0))
        speaker_map[start] = seg.get("speaker", "Unknown")

    enriched = []
    for seg in base_segments:
        start = int(seg.get("start", 0))
        speaker = "Unknown"
        for t in range(start - 2, start + 3):
            if t in speaker_map:
                speaker = speaker_map[t]
                break
        enriched.append({**seg, "speaker": speaker})

    return enriched or base_segments


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def analyse_audio_demo() -> dict[str, Any]:
    """Return demo transcript and extracted events."""
    import copy
    return copy.deepcopy(DEMO_TRANSCRIPT)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_audio(
    audio_bytes: bytes,
    filename: str,
    provider: str = "",
    api_key: str = "",
    demo_mode: bool = True,
) -> dict[str, Any]:
    """
    Transcribe and extract events from an audio file.

    Args:
        audio_bytes: Raw bytes of the uploaded audio.
        filename:    Original filename.
        provider:    LLM provider.
        api_key:     API key.
        demo_mode:   Force demo mode.

    Returns:
        Dict with keys: raw_text, segments, extracted, simulated.
    """
    if demo_mode or not api_key:
        return analyse_audio_demo()

    try:
        # Step 1 — transcribe
        raw = transcribe_audio(audio_bytes, filename, provider, api_key)

        # Build raw text from segments if text is sparse
        raw_text = raw.get("text", "") or "\n".join(
            s.get("text", "") for s in raw.get("segments", [])
        )

        # Step 2 — extract events
        extracted = extract_events_from_transcript(
            raw_text, raw.get("segments", []), provider, api_key
        )

        # Step 3 — enrich segments with speaker tags
        enriched_segments = enrich_segments_with_speakers(raw, extracted)

        return {
            "raw_text": raw_text,
            "segments": enriched_segments,
            "extracted": extracted,
            "simulated": False,
        }

    except Exception as exc:
        result = analyse_audio_demo()
        result["_api_error"] = str(exc)
        return result
