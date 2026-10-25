"""
fusion.py — Master forensic event ledger: build, filter, and retrieve evidence rows.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Ledger schema columns
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = [
    "evidence_id",
    "source_type",
    "timestamp",
    "description",
    "confidence",
    "crime_category",
    "tags",
    "raw_ref",
]


# ---------------------------------------------------------------------------
# Ledger builder
# ---------------------------------------------------------------------------

def build_ledger(
    image_results: Optional[dict[str, Any]] = None,
    audio_results: Optional[dict[str, Any]] = None,
    video_results: Optional[dict[str, Any]] = None,
    image_ev_id: str = "",
    audio_ev_id: str = "",
    video_ev_id: str = "",
) -> pd.DataFrame:
    """
    Build the master forensic event ledger from all analysis results.

    Args:
        image_results:  Vision analysis result dict.
        audio_results:  Audio analysis result dict.
        video_results:  Video analysis result dict.
        image_ev_id:    Evidence ID for the image source.
        audio_ev_id:    Evidence ID for the audio source.
        video_ev_id:    Evidence ID for the video source.

    Returns:
        DataFrame conforming to LEDGER_COLUMNS.
    """
    rows: list[dict[str, Any]] = []

    # ---- Image evidence ----
    if image_results and image_ev_id:
        scene_cap = image_results.get("scene_caption", "")
        threat = image_results.get("threat_level", "Unknown")
        scene_cls = image_results.get("scene_classification", "Unknown")

        # Scene-level row
        rows.append({
            "evidence_id":    image_ev_id,
            "source_type":    "Image",
            "timestamp":      "00:00",
            "description":    f"Scene: {scene_cap}",
            "confidence":     1.0,
            "crime_category": scene_cls,
            "tags":           f"threat:{threat}",
            "raw_ref":        "scene_caption",
        })

        # Per-detection rows
        for item in image_results.get("detected_items", []):
            cls = item.get("class", "unknown").replace("_", " ")
            conf = item.get("confidence", 0.0)
            bbox = item.get("bbox", [])
            notes = item.get("notes", "")
            rows.append({
                "evidence_id":    image_ev_id,
                "source_type":    "Image",
                "timestamp":      "00:00",
                "description":    f"{cls.title()} detected — {notes}",
                "confidence":     conf,
                "crime_category": _map_class_to_crime(cls),
                "tags":           f"class:{cls};bbox:{bbox}",
                "raw_ref":        "detected_items",
            })

        # Investigative leads
        for lead in image_results.get("investigative_leads", []):
            rows.append({
                "evidence_id":    image_ev_id,
                "source_type":    "Image",
                "timestamp":      "00:00",
                "description":    f"[LEAD] {lead}",
                "confidence":     0.0,
                "crime_category": "Lead",
                "tags":           "type:lead",
                "raw_ref":        "investigative_leads",
            })

    # ---- Audio evidence ----
    if audio_results and audio_ev_id:
        extracted = audio_results.get("extracted", {})
        for evt in extracted.get("key_events", []):
            ts = evt.get("timestamp", "00:00")
            desc = evt.get("event", "")
            rows.append({
                "evidence_id":    audio_ev_id,
                "source_type":    "Audio",
                "timestamp":      ts,
                "description":    desc,
                "confidence":     0.85,
                "crime_category": _infer_crime_from_text(desc),
                "tags":           "type:witness_event",
                "raw_ref":        "key_events",
            })

        # Entity rows
        entities = extracted.get("entities", {})
        for category, items in entities.items():
            for item in items:
                rows.append({
                    "evidence_id":    audio_ev_id,
                    "source_type":    "Audio",
                    "timestamp":      "00:00",
                    "description":    f"[ENTITY:{category.upper()}] {item}",
                    "confidence":     0.80,
                    "crime_category": "Intelligence",
                    "tags":           f"entity:{category}",
                    "raw_ref":        "entities",
                })

    # ---- Video evidence ----
    if video_results and video_ev_id:
        for evt in video_results.get("events", []):
            rows.append({
                "evidence_id":    video_ev_id,
                "source_type":    "Video",
                "timestamp":      evt.get("timestamp", "00:00"),
                "description":    evt.get("event", ""),
                "confidence":     evt.get("confidence", 0.0),
                "crime_category": evt.get("crime_category", "Unknown"),
                "tags":           f"track:{evt.get('track_id', '?')}",
                "raw_ref":        "video_events",
            })

    if not rows:
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    df = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    df["confidence"] = df["confidence"].round(3)
    return df


def _map_class_to_crime(cls: str) -> str:
    """Map a detection class name to a crime category string."""
    mapping = {
        "person":             "Trespass",
        "firearm":            "Armed Threat",
        "edged weapon":       "Armed Threat",
        "blood spatter":      "Assault",
        "forced entry":       "Burglary",
        "vehicle":            "Unknown",
        "suspicious package": "Unknown",
        "evidence marker":    "Scene Processing",
    }
    return mapping.get(cls.lower(), "Unknown")


def _infer_crime_from_text(text: str) -> str:
    """Heuristic crime category inference from event description text."""
    text_l = text.lower()
    if any(w in text_l for w in ["weapon", "armed", "gun", "knife", "firearm"]):
        return "Armed Threat"
    if any(w in text_l for w in ["fled", "run", "escape", "vehicle", "car"]):
        return "Flee / Evasion"
    if any(w in text_l for w in ["assault", "attack", "scream", "fight"]):
        return "Assault"
    if any(w in text_l for w in ["break", "entry", "glass", "door"]):
        return "Burglary"
    if any(w in text_l for w in ["dispatch", "unit", "officer", "respond"]):
        return "Law Enforcement Response"
    return "Unknown"


# ---------------------------------------------------------------------------
# Retrieval for grounded RAG
# ---------------------------------------------------------------------------

def retrieve_relevant_rows(
    query: str,
    df: pd.DataFrame,
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Retrieve the most relevant ledger rows for a query using TF-IDF-style scoring.

    Args:
        query: Investigator's natural-language question.
        df:    Master ledger DataFrame.
        top_k: Maximum number of rows to return.

    Returns:
        Subset DataFrame of most relevant rows, sorted by relevance score.
    """
    if df.empty:
        return df

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return df.head(top_k)

    # Build corpus: concatenate description + tags + crime_category per row
    corpus = (
        df["description"].fillna("") + " "
        + df["tags"].fillna("") + " "
        + df["crime_category"].fillna("")
    ).str.lower()

    # Term frequency in each document
    scores: list[float] = []
    for doc in corpus:
        doc_tokens = _tokenize(doc)
        if not doc_tokens:
            scores.append(0.0)
            continue
        tf = Counter(doc_tokens)
        score = sum(tf.get(t, 0) for t in query_tokens)
        # Boost for exact phrase
        if query.lower() in doc:
            score += 5
        scores.append(score)

    df2 = df.copy()
    df2["_score"] = scores
    df2 = df2[df2["_score"] > 0].sort_values("_score", ascending=False)

    if df2.empty:
        # Fallback: return first top_k rows
        return df.head(top_k)

    return df2.drop(columns=["_score"]).head(top_k)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Ledger summary helpers
# ---------------------------------------------------------------------------

def ledger_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics from the master ledger.

    Args:
        df: Master ledger DataFrame.

    Returns:
        Dict with counts, top threats, modality breakdown, etc.
    """
    if df.empty:
        return {"total_events": 0, "modalities": {}, "top_crime_categories": {}}

    return {
        "total_events":       len(df),
        "modalities":         df["source_type"].value_counts().to_dict(),
        "top_crime_categories": df["crime_category"].value_counts().head(5).to_dict(),
        "avg_confidence":     round(df["confidence"].mean(), 3),
        "evidence_ids":       df["evidence_id"].unique().tolist(),
    }
