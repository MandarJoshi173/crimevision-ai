"""
chat.py — Grounded RAG chat: retrieve evidence rows, call LLM, enforce anti-hallucination.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from config import DEMO_CHAT_RESPONSES
from modules.fusion import retrieve_relevant_rows


# ---------------------------------------------------------------------------
# System prompt — strict anti-hallucination instructions
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are CrimeVision AI, a forensic intelligence assistant.

STRICT RULES — you MUST follow these without exception:
1. Answer ONLY based on the evidence rows provided in the context below.
2. Every factual claim must cite its Evidence ID inline like this: [EV-IMG-001] or [EV-VID-001 @ 00:12].
3. If the provided evidence does not contain sufficient information to answer, respond with:
   "Insufficient evidence in the case file."
4. Do NOT speculate, infer, or introduce information not present in the evidence rows.
5. Do NOT hallucinate Evidence IDs — only cite IDs that appear in the context.
6. Structure your answer clearly: start with a brief direct answer, then cite supporting evidence.
7. If multiple modalities support the same conclusion, cross-reference them explicitly.

This output may be used in legal proceedings. Accuracy is paramount.
"""


def _format_context(retrieved: pd.DataFrame) -> str:
    """
    Format retrieved evidence rows into a structured context string for the LLM.

    Args:
        retrieved: DataFrame of retrieved evidence rows.

    Returns:
        Formatted context string.
    """
    if retrieved.empty:
        return "No relevant evidence rows found in the case file."

    lines = ["=== RETRIEVED EVIDENCE ROWS ==="]
    for _, row in retrieved.iterrows():
        ev_id = row.get("evidence_id", "?")
        ts = row.get("timestamp", "?")
        src = row.get("source_type", "?")
        desc = row.get("description", "")
        conf = row.get("confidence", 0.0)
        cat = row.get("crime_category", "?")

        lines.append(
            f"[{ev_id} @ {ts}] SOURCE={src} CATEGORY={cat} CONF={conf:.0%}\n  {desc}"
        )

    lines.append("=== END OF EVIDENCE ROWS ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live LLM chat
# ---------------------------------------------------------------------------

def chat_grounded(
    query: str,
    ledger: pd.DataFrame,
    provider: str,
    api_key: str,
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Answer an investigator query grounded in the evidence ledger via LLM.

    Args:
        query:    Investigator's natural-language question.
        ledger:   Master forensic event ledger DataFrame.
        provider: LLM provider ('Groq' or 'Gemini').
        api_key:  Provider API key.
        top_k:    Number of evidence rows to retrieve.

    Returns:
        Dict with keys: 'answer' (str), 'retrieved_rows' (DataFrame), 'simulated' (bool).
    """
    from modules.llm import chat_completion

    retrieved = retrieve_relevant_rows(query, ledger, top_k=top_k)
    context = _format_context(retrieved)

    user_message = f"{context}\n\nINVESTIGATOR QUERY: {query}"

    answer = chat_completion(
        provider=provider,
        api_key=api_key,
        system_prompt=_SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.1,
    )

    return {
        "answer": answer,
        "retrieved_rows": retrieved,
        "simulated": False,
    }


# ---------------------------------------------------------------------------
# Demo mode — rule-based grounded answers
# ---------------------------------------------------------------------------

def chat_grounded_demo(
    query: str,
    ledger: pd.DataFrame,
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Answer an investigator query using rule-based demo logic grounded in the ledger.

    Args:
        query:  Investigator's natural-language question.
        ledger: Master forensic event ledger DataFrame.
        top_k:  Number of evidence rows to retrieve.

    Returns:
        Dict with keys: 'answer' (str), 'retrieved_rows' (DataFrame), 'simulated' (bool).
    """
    retrieved = retrieve_relevant_rows(query, ledger, top_k=top_k)
    query_lower = query.lower()

    # Rule-based keyword matching for canned responses
    answer = _build_demo_answer(query_lower, retrieved, ledger)

    return {
        "answer": answer,
        "retrieved_rows": retrieved,
        "simulated": True,
    }


def _build_demo_answer(
    query_lower: str,
    retrieved: pd.DataFrame,
    full_ledger: pd.DataFrame,
) -> str:
    """Build a rule-based grounded answer with evidence citations."""

    # Collect all evidence IDs present in the ledger
    all_ids = full_ledger["evidence_id"].unique().tolist() if not full_ledger.empty else []
    id_list = ", ".join(f"[{eid}]" for eid in all_ids) if all_ids else "[no evidence loaded]"

    # Keyword routing
    if any(w in query_lower for w in ["weapon", "gun", "knife", "firearm", "armed"]):
        base = DEMO_CHAT_RESPONSES["weapon"]
    elif any(w in query_lower for w in ["suspect", "person", "who", "perpetrator", "individual"]):
        base = DEMO_CHAT_RESPONSES["suspect"]
    elif any(w in query_lower for w in ["vehicle", "car", "sedan", "drove", "drove away"]):
        base = DEMO_CHAT_RESPONSES["vehicle"]
    else:
        # Try to build an answer from retrieved rows if any match
        if not retrieved.empty:
            lines = [
                "Based on the available evidence:\n",
            ]
            for _, row in retrieved.head(4).iterrows():
                ev_id = row.get("evidence_id", "?")
                ts = row.get("timestamp", "00:00")
                desc = row.get("description", "")
                lines.append(f"• [{ev_id} @ {ts}] — {desc}")
            lines.append(
                f"\n*Evidence IDs in case file: {id_list}*"
            )
            return "\n".join(lines)
        else:
            base = DEMO_CHAT_RESPONSES["default"]

    # Inject actual evidence IDs from the ledger into canned responses if present
    if all_ids:
        return base + f"\n\n*Evidence IDs in case file: {id_list}*"
    return base


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask_question(
    query: str,
    ledger: pd.DataFrame,
    provider: str = "",
    api_key: str = "",
    demo_mode: bool = True,
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Dispatch a forensic query to the grounded chat system.

    Args:
        query:     Investigator's natural-language question.
        ledger:    Master forensic event ledger DataFrame.
        provider:  LLM provider.
        api_key:   Provider API key.
        demo_mode: Force demo mode.
        top_k:     Evidence rows to retrieve.

    Returns:
        Dict with 'answer', 'retrieved_rows', 'simulated'.
    """
    if ledger.empty:
        return {
            "answer": "No evidence has been loaded into the case file yet. "
                      "Please upload and analyse evidence in Tabs 1–3 first.",
            "retrieved_rows": pd.DataFrame(),
            "simulated": True,
        }

    if demo_mode or not api_key:
        return chat_grounded_demo(query, ledger, top_k=top_k)

    try:
        return chat_grounded(query, ledger, provider, api_key, top_k=top_k)
    except Exception as exc:
        # Fall back to demo with error note
        result = chat_grounded_demo(query, ledger, top_k=top_k)
        result["answer"] += f"\n\n⚠️ *Live AI unavailable ({exc}). Showing rule-based response.*"
        return result
