"""
custody.py — Chain-of-custody IDs, SHA-256 hashing, and audit log management.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import IO, Any

import streamlit as st


# ---------------------------------------------------------------------------
# Evidence ID generation
# ---------------------------------------------------------------------------

_COUNTERS: dict[str, int] = {}


def _next_counter(prefix: str) -> int:
    """Return auto-incrementing counter stored in session state."""
    key = f"_ev_counter_{prefix}"
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] += 1
    return st.session_state[key]


def generate_case_id() -> str:
    """Generate a unique forensic case ID (e.g. CASE-2024-A3F2)."""
    suffix = uuid.uuid4().hex[:4].upper()
    year = datetime.now(timezone.utc).year
    return f"CASE-{year}-{suffix}"


def generate_evidence_id(source_type: str) -> str:
    """
    Generate a sequential Evidence ID based on source type.

    Args:
        source_type: 'image', 'audio', or 'video'.

    Returns:
        Formatted evidence ID such as 'EV-IMG-001'.
    """
    mapping = {"image": "IMG", "audio": "AUD", "video": "VID"}
    tag = mapping.get(source_type.lower(), "OTH")
    n = _next_counter(tag)
    return f"EV-{tag}-{n:03d}"


# ---------------------------------------------------------------------------
# SHA-256 hashing
# ---------------------------------------------------------------------------

def sha256_file(file_obj: IO[bytes]) -> str:
    """
    Compute SHA-256 hash of an uploaded file-like object.

    Args:
        file_obj: File-like object with a read() method.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    file_obj.seek(0)
    while chunk := file_obj.read(65536):
        h.update(chunk)
    file_obj.seek(0)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# UTC timestamp
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Audit / chain-of-custody log
# ---------------------------------------------------------------------------

def init_custody_log() -> None:
    """Initialise the chain-of-custody log in session state if absent."""
    if "custody_log" not in st.session_state:
        st.session_state.custody_log = []


def log_action(
    action: str,
    evidence_id: str = "",
    details: str = "",
    investigator: str = "Unknown",
) -> None:
    """
    Append an action record to the chain-of-custody log.

    Args:
        action:       Short action label (e.g. 'UPLOAD', 'ANALYSIS', 'EXPORT').
        evidence_id:  Associated evidence ID (optional).
        details:      Free-text details.
        investigator: Name of the acting investigator.
    """
    init_custody_log()
    record: dict[str, Any] = {
        "timestamp": utc_now(),
        "action": action.upper(),
        "evidence_id": evidence_id,
        "investigator": investigator,
        "details": details,
    }
    st.session_state.custody_log.append(record)


def get_custody_log() -> list[dict[str, Any]]:
    """Return the full chain-of-custody log."""
    init_custody_log()
    return st.session_state.custody_log


# ---------------------------------------------------------------------------
# Evidence ledger helpers
# ---------------------------------------------------------------------------

def init_evidence_registry() -> None:
    """Initialise the per-file evidence registry in session state."""
    if "evidence_registry" not in st.session_state:
        st.session_state.evidence_registry = {}


def register_evidence(
    evidence_id: str,
    filename: str,
    source_type: str,
    sha256: str,
    investigator: str = "Unknown",
) -> dict[str, Any]:
    """
    Register an uploaded file in the evidence registry.

    Args:
        evidence_id:  The generated Evidence ID.
        filename:     Original filename.
        source_type:  'image', 'audio', or 'video'.
        sha256:       SHA-256 hash of the file.
        investigator: Investigator name.

    Returns:
        The registry record dict.
    """
    init_evidence_registry()
    record: dict[str, Any] = {
        "evidence_id": evidence_id,
        "filename": filename,
        "source_type": source_type,
        "sha256": sha256,
        "uploaded_at": utc_now(),
        "investigator": investigator,
    }
    st.session_state.evidence_registry[evidence_id] = record
    log_action("UPLOAD", evidence_id, f"File: {filename} | SHA256: {sha256[:16]}…", investigator)
    return record


def get_evidence_registry() -> dict[str, Any]:
    """Return the full evidence registry."""
    init_evidence_registry()
    return st.session_state.evidence_registry
