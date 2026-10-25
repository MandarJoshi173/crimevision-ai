"""
CrimeVision AI - Configuration, Constants, Theme, and Demo Data
"""
from __future__ import annotations
import os
from typing import Any

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME = "CrimeVision AI"
APP_SUBTITLE = "Intelligent Multimodal Forensic Analysis Platform"
APP_VERSION = "1.0.0"
CAPSTONE_TAG = "Skill Wallet Capstone Project 2 — Generative AI / Criminal Justice & Public Safety"

# ---------------------------------------------------------------------------
# File-size limits (MB)
# ---------------------------------------------------------------------------
MAX_IMAGE_MB = 10
MAX_AUDIO_MB = 25
MAX_VIDEO_MB = 50

# ---------------------------------------------------------------------------
# Video sampling
# ---------------------------------------------------------------------------
VIDEO_FPS_SAMPLE = 1          # frames per second to sample
VIDEO_MAX_FRAMES = 60         # hard cap for performance

# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
PROVIDERS = ["Groq", "Gemini"]
DEFAULT_PROVIDER = "Groq"

GROQ_MODELS = {
    "chat": "openai/gpt-oss-120b",
    "vision": "qwen/qwen3.8-27b",
    "whisper": "whisper-large-v3",
}
GEMINI_MODELS = {
    "chat": "gemini-2.0-flash",
    "vision": "gemini-2.0-flash",
    "whisper": None,          # Gemini doesn't expose Whisper; falls back to demo
}

# ---------------------------------------------------------------------------
# Detection labels / taxonomy
# ---------------------------------------------------------------------------
DETECTION_CLASSES = [
    "person", "firearm", "edged_weapon", "blood_spatter",
    "forced_entry", "vehicle", "suspicious_package", "evidence_marker",
]
THREAT_LEVELS = ["Low", "Medium", "High", "Critical"]
SCENE_TYPES = ["Burglary", "Homicide", "Vandalism", "Assault", "Robbery", "Unknown"]
CRIME_CATEGORIES = ["Trespass", "Theft", "Assault", "Armed Threat", "Vandalism", "Unknown"]

# ---------------------------------------------------------------------------
# Custom CSS — dark forensic theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'Courier New', Courier, monospace;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #070B14;
    border-right: 1px solid #1E2D40;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00FF9C;
    letter-spacing: 2px;
    font-size: 0.75rem;
    text-transform: uppercase;
}

/* ---- Tabs ---- */
div[data-baseweb="tab-list"] {
    background: #111827;
    border-bottom: 2px solid #00FF9C22;
    gap: 4px;
}
button[data-baseweb="tab"] {
    background: transparent;
    color: #94A3B8;
    font-size: 0.78rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 4px 4px 0 0;
    transition: all 0.2s;
}
button[data-baseweb="tab"]:hover {
    color: #00FF9C;
    background: #0A0E1A;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00FF9C;
    border-bottom: 2px solid #00FF9C;
    background: #0A0E1A;
}

/* ---- Cards / containers ---- */
div.forensic-card {
    background: #111827;
    border: 1px solid #1E2D40;
    border-left: 3px solid #00FF9C;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
div.threat-critical { border-left-color: #FF3B3B; }
div.threat-high     { border-left-color: #FF8C00; }
div.threat-medium   { border-left-color: #FFD700; }
div.threat-low      { border-left-color: #00FF9C; }

/* ---- Demo badge ---- */
div.demo-badge {
    display: inline-block;
    background: #FF8C0022;
    border: 1px solid #FF8C00;
    color: #FF8C00;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.70rem;
    letter-spacing: 2px;
    font-weight: bold;
    text-transform: uppercase;
}
div.live-badge {
    display: inline-block;
    background: #00FF9C22;
    border: 1px solid #00FF9C;
    color: #00FF9C;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.70rem;
    letter-spacing: 2px;
    font-weight: bold;
    text-transform: uppercase;
}

/* ---- Evidence ID pill ---- */
code.ev-id {
    background: #1E2D40;
    color: #7DD3FC;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.75rem;
}

/* ---- Tables ---- */
thead tr th {
    background: #1E2D40 !important;
    color: #00FF9C !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ---- Buttons ---- */
div.stButton > button {
    background: linear-gradient(135deg, #00FF9C22, #0A0E1A);
    border: 1px solid #00FF9C;
    color: #00FF9C;
    font-family: 'Courier New', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.75rem;
    border-radius: 4px;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: #00FF9C;
    color: #0A0E1A;
    font-weight: bold;
}

/* ---- Metric cards ---- */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1E2D40;
    border-radius: 6px;
    padding: 0.5rem;
}

/* ---- Header ---- */
.app-header {
    background: linear-gradient(90deg, #070B14 0%, #0A1628 50%, #070B14 100%);
    border-bottom: 1px solid #00FF9C33;
    padding: 0.5rem 1rem;
    margin-bottom: 1rem;
}
.app-title {
    color: #00FF9C;
    font-size: 1.5rem;
    font-weight: bold;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.app-subtitle {
    color: #64748B;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
"""

# ---------------------------------------------------------------------------
# Demo / fallback data
# ---------------------------------------------------------------------------

DEMO_VISION_RESULT: dict[str, Any] = {
    "threat_level": "High",
    "scene_classification": "Assault",
    "scene_caption": (
        "Crime scene shows signs of a violent altercation. A person is visible "
        "near the central area with what appears to be an edged weapon. Forced "
        "entry indicators are present on the left perimeter. Blood spatter "
        "patterns are detected in the lower-right quadrant."
    ),
    "detected_items": [
        {
            "class": "person",
            "confidence": 0.93,
            "bbox": [142, 88, 310, 420],
            "notes": "Standing figure, facing east",
        },
        {
            "class": "edged_weapon",
            "confidence": 0.81,
            "bbox": [290, 220, 340, 310],
            "notes": "Blade-like object in suspect's right hand",
        },
        {
            "class": "blood_spatter",
            "confidence": 0.74,
            "bbox": [380, 310, 510, 410],
            "notes": "Impact spatter pattern — medium velocity",
        },
        {
            "class": "forced_entry",
            "confidence": 0.87,
            "bbox": [20, 50, 120, 300],
            "notes": "Door frame damage, latch mechanism broken",
        },
    ],
    "investigative_leads": [
        "Identify and locate individual detected at [142,88,310,420].",
        "Recover edged weapon for fingerprint and DNA analysis.",
        "Document blood spatter geometry for trajectory reconstruction.",
        "Photograph forced entry point for tool-mark analysis.",
    ],
    "simulated": True,
}

DEMO_TRANSCRIPT: dict[str, Any] = {
    "raw_text": (
        "[00:00] Dispatcher: 911, what's your emergency?\n"
        "[00:04] Caller: Yes, I'm at 45 Maple Street — I can hear screaming and breaking glass "
        "from the apartment upstairs, unit 4B.\n"
        "[00:12] Dispatcher: Are you safe right now?\n"
        "[00:14] Caller: Yes, I'm in the hallway, third floor.\n"
        "[00:18] Dispatcher: Did you see anyone entering or leaving the building?\n"
        "[00:22] Caller: A man in a dark hoodie — maybe six feet tall — ran out the back "
        "about two minutes ago. He was carrying something, like a bag.\n"
        "[00:34] Dispatcher: Any vehicles?\n"
        "[00:36] Caller: A dark sedan, maybe a Honda, no plates visible, headed north on Maple.\n"
        "[00:44] Dispatcher: Units are on the way. Stay on the line.\n"
    ),
    "segments": [
        {"start": 0, "end": 4, "speaker": "Dispatcher", "text": "911, what's your emergency?"},
        {"start": 4, "end": 12, "speaker": "Caller",
         "text": "Yes, I'm at 45 Maple Street — I can hear screaming and breaking glass from the apartment upstairs, unit 4B."},
        {"start": 12, "end": 14, "speaker": "Dispatcher", "text": "Are you safe right now?"},
        {"start": 14, "end": 18, "speaker": "Caller", "text": "Yes, I'm in the hallway, third floor."},
        {"start": 18, "end": 22, "speaker": "Dispatcher", "text": "Did you see anyone entering or leaving the building?"},
        {"start": 22, "end": 34, "speaker": "Caller",
         "text": "A man in a dark hoodie — maybe six feet tall — ran out the back about two minutes ago. He was carrying something, like a bag."},
        {"start": 34, "end": 36, "speaker": "Dispatcher", "text": "Any vehicles?"},
        {"start": 36, "end": 44, "speaker": "Caller",
         "text": "A dark sedan, maybe a Honda, no plates visible, headed north on Maple."},
        {"start": 44, "end": 48, "speaker": "Dispatcher", "text": "Units are on the way. Stay on the line."},
    ],
    "extracted": {
        "entities": {
            "people": ["Male suspect, dark hoodie, ~6ft"],
            "vehicles": ["Dark sedan (possible Honda), no plates, heading north on Maple St"],
            "locations": ["45 Maple Street, Unit 4B", "Back exit of building", "North on Maple St"],
            "weapons": ["Unknown carried bag/item"],
        },
        "key_events": [
            {"timestamp": "00:04", "event": "Call received — report of screaming and breaking glass at 45 Maple St Unit 4B"},
            {"timestamp": "00:22", "event": "Suspect fled via back exit — male in dark hoodie carrying bag"},
            {"timestamp": "00:36", "event": "Suspect vehicle identified — dark sedan, northbound Maple St, no plates"},
            {"timestamp": "00:44", "event": "Law enforcement units dispatched"},
        ],
    },
    "simulated": True,
}

DEMO_VIDEO_EVENTS: list[dict[str, Any]] = [
    {"frame": 0, "timestamp": "00:00", "track_id": "T-01", "event": "Person enters frame from south entrance", "crime_category": "Trespass", "confidence": 0.88},
    {"frame": 5, "timestamp": "00:05", "track_id": "T-01", "event": "Person moves toward display counter", "crime_category": "Trespass", "confidence": 0.82},
    {"frame": 12, "timestamp": "00:12", "track_id": "T-01", "event": "Person reaches into display case", "crime_category": "Theft", "confidence": 0.91},
    {"frame": 18, "timestamp": "00:18", "track_id": "T-02", "event": "Second person enters frame from north", "crime_category": "Trespass", "confidence": 0.85},
    {"frame": 24, "timestamp": "00:24", "track_id": "T-01", "event": "Possible weapon brandished by T-01", "crime_category": "Armed Threat", "confidence": 0.76},
    {"frame": 30, "timestamp": "00:30", "track_id": "T-02", "event": "T-02 moves to block exit", "crime_category": "Assault", "confidence": 0.79},
    {"frame": 38, "timestamp": "00:38", "track_id": "T-01", "event": "T-01 removes item from case and conceals it", "crime_category": "Theft", "confidence": 0.94},
    {"frame": 45, "timestamp": "00:45", "track_id": "T-01", "event": "T-01 exits north — suspect fled", "crime_category": "Theft", "confidence": 0.90},
    {"frame": 48, "timestamp": "00:48", "track_id": "T-02", "event": "T-02 exits south", "crime_category": "Trespass", "confidence": 0.83},
    {"frame": 55, "timestamp": "00:55", "track_id": "VEH-01", "event": "Dark vehicle departs from parking area", "crime_category": "Unknown", "confidence": 0.70},
]

DEMO_CHAT_RESPONSES: dict[str, str] = {
    "weapon": (
        "Based on the case evidence:\n\n"
        "• [EV-IMG-001] — An edged weapon (confidence 81%) was detected at timestamp of image upload, "
        "located at bbox [290, 220, 340, 310].\n"
        "• [EV-VID-001 @ 00:24] — A possible weapon was brandished by Track T-01 (confidence 76%).\n\n"
        "**Assessment:** A weapon was present prior to the suspect's flight at [EV-VID-001 @ 00:45]. "
        "The timeline shows weapon detection preceded the flee event by approximately 21 seconds."
    ),
    "suspect": (
        "Based on the case evidence:\n\n"
        "• [EV-AUD-001 @ 00:22] — Caller described male suspect in dark hoodie, ~6ft tall, "
        "fled via back exit carrying a bag.\n"
        "• [EV-VID-001 @ 00:00] — Track T-01 entered via south entrance.\n"
        "• [EV-VID-001 @ 00:45] — Track T-01 exited north.\n\n"
        "**Assessment:** One primary suspect (T-01) corroborated across audio and video modalities."
    ),
    "vehicle": (
        "Based on the case evidence:\n\n"
        "• [EV-AUD-001 @ 00:36] — Dark sedan, possible Honda, no plates, northbound Maple St.\n"
        "• [EV-VID-001 @ 00:55] — Dark vehicle departed parking area.\n\n"
        "**Assessment:** Vehicle evidence is corroborated across audio and video sources. "
        "BOLO recommended for dark sedan, no plates, last seen northbound."
    ),
    "default": (
        "Based on the available evidence in the case file, I cannot locate a specific answer "
        "to your query. **Insufficient evidence in the case file** for this specific question.\n\n"
        "Relevant evidence IDs loaded: [EV-IMG-001], [EV-AUD-001], [EV-VID-001]. "
        "Please refine your query or upload additional evidence."
    ),
}
