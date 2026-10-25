"""
app.py — CrimeVision AI: Streamlit entry point, sidebar, and tab routing.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image

# ── Local imports ─────────────────────────────────────────────────────────────
from config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    CAPSTONE_TAG,
    CUSTOM_CSS,
    MAX_AUDIO_MB,
    MAX_IMAGE_MB,
    MAX_VIDEO_MB,
    PROVIDERS,
    DEFAULT_PROVIDER,
)
from modules.custody import (
    generate_case_id,
    generate_evidence_id,
    get_custody_log,
    get_evidence_registry,
    init_custody_log,
    init_evidence_registry,
    log_action,
    register_evidence,
    sha256_file,
    utc_now,
)
from modules.vision import analyse_image, annotate_image, pil_from_upload
from modules.audio import analyse_audio
from modules.video import analyse_video
from modules.fusion import build_ledger, ledger_summary
from modules.chat import ask_question
from modules.report import generate_pdf

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrimeVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_session() -> None:
    """Initialise all session-state keys on first run."""
    defaults: dict[str, Any] = {
        "case_id":          generate_case_id(),
        "investigator":     "",
        "signed_off":       False,
        "api_key":          "",
        "provider":         DEFAULT_PROVIDER,
        "demo_mode":        True,
        # Per-tab analysis results
        "image_result":     None,
        "image_ev_id":      "",
        "annotated_img":    None,   # PIL Image
        "audio_result":     None,
        "audio_ev_id":      "",
        "video_result":     None,
        "video_ev_id":      "",
        # Master ledger
        "ledger":           pd.DataFrame(),
        # Chat history: list of {query, answer, retrieved_rows, simulated}
        "chat_history":     [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    init_custody_log()
    init_evidence_registry()


_init_session()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    """Render the configuration + chain-of-custody sidebar."""
    with st.sidebar:
        # App identity
        st.markdown(
            f"""
            <div style="text-align:center;padding:0.5rem 0;">
              <div style="color:#00FF9C;font-size:1.3rem;font-weight:bold;letter-spacing:3px;">
                🔬 CRIMEVISION AI
              </div>
              <div style="color:#64748B;font-size:0.65rem;letter-spacing:2px;">
                {APP_VERSION} &nbsp;|&nbsp; {CAPSTONE_TAG[:40]}…
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── API Configuration ──────────────────────────────────────────────
        st.markdown("### ⚙️ AI Configuration")

        try:
            server_api_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            server_api_key = ""

        if server_api_key:
            provider = "Groq"
            api_key = server_api_key
            st.session_state.provider = provider
            st.session_state.api_key = api_key
            st.session_state.demo_mode = False

            st.caption("Live AI is configured by the app owner.")
            st.markdown(
                '<div class="live-badge">● LIVE AI — Groq</div>',
                unsafe_allow_html=True,
            )
        else:
            provider = st.selectbox(
                "Provider",
                PROVIDERS,
                index=PROVIDERS.index(st.session_state.provider),
                key="_provider_select",
            )
            st.session_state.provider = provider

            api_key = st.text_input(
                "API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="Enter API key (leave blank for demo)",
                key="_api_key_input",
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key

            demo_mode = not bool(api_key.strip())
            st.session_state.demo_mode = demo_mode

            if demo_mode:
                st.markdown(
                    '<div class="demo-badge">⚠ DEMO MODE — SIMULATED OUTPUT</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="live-badge">● LIVE AI — {provider}</div>',
                    unsafe_allow_html=True,
                )
        st.divider()

        # ── Case Panel ─────────────────────────────────────────────────────
        st.markdown("### 📁 Case Panel")

        st.code(f"CASE ID: {st.session_state.case_id}", language=None)

        investigator = st.text_input(
            "Investigator Name",
            value=st.session_state.investigator,
            placeholder="Detective / Analyst Name",
            key="_investigator_input",
        )
        st.session_state.investigator = investigator

        signed = st.checkbox(
            "✅ I have reviewed and sign off on this analysis",
            value=st.session_state.signed_off,
            key="_signed_off_check",
        )
        st.session_state.signed_off = signed

        if st.button("🔄 New Case", help="Reset the case ID (analysis results are preserved)"):
            st.session_state.case_id = generate_case_id()
            log_action("NEW_CASE", details=f"New case created: {st.session_state.case_id}",
                       investigator=investigator)
            st.rerun()

        st.divider()

        # ── Evidence Registry ──────────────────────────────────────────────
        st.markdown("### 🗃️ Evidence Registry")
        registry = get_evidence_registry()
        if registry:
            for ev_id, rec in registry.items():
                st.markdown(
                    f"`{ev_id}` — {rec['source_type'].upper()} — `{rec['filename'][:22]}`",
                )
        else:
            st.caption("No evidence registered yet.")

        st.divider()

        # ── Chain-of-Custody Log ───────────────────────────────────────────
        st.markdown("### 🔗 Chain of Custody")
        log = get_custody_log()
        if log:
            cdf = pd.DataFrame(log)[["timestamp", "action", "evidence_id", "investigator"]]
            cdf.columns = ["UTC", "Action", "EV ID", "By"]
            st.dataframe(
                cdf.tail(15),
                use_container_width=True,
                hide_index=True,
                height=220,
            )
        else:
            st.caption("No actions recorded yet.")


_render_sidebar()


# ══════════════════════════════════════════════════════════════════════════════
# App header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-title">🔬 CRIMEVISION AI</div>
      <div class="app-subtitle">{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Tab 1 — Visual Evidence",
    "🎙️ Tab 2 — Audio / Witness",
    "🎥 Tab 3 — CCTV Video",
    "🔗 Tab 4 — Fusion & Chat",
    "📄 Tab 5 — PDF Export",
])


# ══════════════════════════════════════════════════════════════════════════════
# Ledger rebuild helper — defined BEFORE tabs that call it
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_ledger() -> None:
    """Merge all analysis results into the master forensic event ledger."""
    st.session_state.ledger = build_ledger(
        image_results=st.session_state.image_result,
        audio_results=st.session_state.audio_result,
        video_results=st.session_state.video_result,
        image_ev_id=st.session_state.image_ev_id,
        audio_ev_id=st.session_state.audio_ev_id,
        video_ev_id=st.session_state.video_ev_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: mode badge
# ──────────────────────────────────────────────────────────────────────────────

def _mode_badge(simulated: bool) -> str:
    if simulated:
        return '<div class="demo-badge">SIMULATED OUTPUT</div>'
    return '<div class="live-badge">LIVE AI OUTPUT</div>'


def _threat_colour(level: str) -> str:
    return {
        "Critical": "threat-critical",
        "High":     "threat-high",
        "Medium":   "threat-medium",
        "Low":      "threat-low",
    }.get(level, "")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Crime Scene Visual Evidence Tagger
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Crime Scene Visual Evidence Tagger")
    st.caption("Upload a crime scene photograph. AI detects persons, weapons, blood spatter, forced entry, and more.")

    col_upload, col_info = st.columns([1, 1])

    with col_upload:
        img_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            key="img_uploader",
            help=f"Max {MAX_IMAGE_MB} MB · JPG / PNG",
        )

    if img_file is not None:
        # Size check
        img_file.seek(0, 2)
        size_mb = img_file.tell() / 1_048_576
        img_file.seek(0)
        if size_mb > MAX_IMAGE_MB:
            st.error(f"File too large ({size_mb:.1f} MB). Limit is {MAX_IMAGE_MB} MB.")
        else:
            # Register evidence (idempotent per filename+case)
            ev_key = f"ev_img_{img_file.name}_{st.session_state.case_id}"
            if ev_key not in st.session_state:
                ev_id = generate_evidence_id("image")
                sha = sha256_file(img_file)
                register_evidence(ev_id, img_file.name, "image", sha, st.session_state.investigator)
                st.session_state[ev_key] = ev_id
                st.session_state.image_ev_id = ev_id
            else:
                st.session_state.image_ev_id = st.session_state[ev_key]

            ev_id = st.session_state.image_ev_id

            try:
                pil_img = pil_from_upload(img_file)
            except ValueError as e:
                st.error(f"Cannot open image: {e}")
                pil_img = None

            if pil_img:
                with col_info:
                    st.markdown(f"**Evidence ID:** `{ev_id}`")
                    w, h = pil_img.size
                    st.markdown(f"**Dimensions:** {w} × {h} px &nbsp;|&nbsp; **Size:** {size_mb:.2f} MB")
                    registry = get_evidence_registry()
                    sha_display = registry.get(ev_id, {}).get("sha256", "?")[:32]
                    st.markdown(f"**SHA-256:** `{sha_display}…`")
                    st.markdown(f"**Uploaded:** `{utc_now()}`")

                # Analyse button
                if st.button("🔍 Analyse Crime Scene Image", key="btn_analyse_img"):
                    with st.spinner("Analysing image…"):
                        try:
                            result = analyse_image(
                                img=pil_img,
                                provider=st.session_state.provider,
                                api_key=st.session_state.api_key,
                                demo_mode=st.session_state.demo_mode,
                            )
                            st.session_state.image_result = result
                            # Annotate
                            ann = annotate_image(pil_img, result.get("detected_items", []))
                            st.session_state.annotated_img = ann
                            log_action("ANALYSIS", ev_id, "Image analysis complete",
                                       st.session_state.investigator)
                            _rebuild_ledger()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")

                # Show results
                result = st.session_state.image_result
                if result:
                    st.markdown(_mode_badge(result.get("simulated", True)), unsafe_allow_html=True)

                    # Threat + scene
                    threat = result.get("threat_level", "Unknown")
                    scene_cls = result.get("scene_classification", "Unknown")
                    tc = _threat_colour(threat)

                    st.markdown(
                        f'<div class="forensic-card {tc}">'
                        f'<b>Threat Level:</b> {threat} &nbsp;|&nbsp; '
                        f'<b>Scene:</b> {scene_cls}<br/>'
                        f'<i>{result.get("scene_caption", "")}</i>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Side by side: original + annotated
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(pil_img, caption="Original", use_container_width=True)
                    with c2:
                        ann_img = st.session_state.annotated_img
                        if ann_img:
                            st.image(ann_img, caption="Annotated", use_container_width=True)

                    # Detections table
                    items = result.get("detected_items", [])
                    if items:
                        st.subheader("Detected Items")
                        det_df = pd.DataFrame([
                            {
                                "Class":       d.get("class", "?").replace("_", " ").title(),
                                "Confidence":  f"{d.get('confidence', 0):.1%}",
                                "BBox [x1,y1,x2,y2]": str(d.get("bbox", [])),
                                "Notes":       d.get("notes", ""),
                            }
                            for d in items
                        ])
                        st.dataframe(det_df, use_container_width=True, hide_index=True)

                    # Investigative leads
                    leads = result.get("investigative_leads", [])
                    if leads:
                        st.subheader("🎯 Investigative Leads")
                        for i, lead in enumerate(leads, 1):
                            st.markdown(f"{i}. {lead}")

                    # Raw JSON
                    with st.expander("🔧 Raw JSON Output"):
                        safe_result = {k: v for k, v in result.items() if k != "simulated"}
                        st.json(safe_result)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Witness Audio Transcription & Timeline
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Witness Audio Transcription & Timeline")
    st.caption("Upload a witness statement or 911 call. AI transcribes and extracts forensic events.")

    aud_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "m4a"],
        key="aud_uploader",
        help=f"Max {MAX_AUDIO_MB} MB · WAV / MP3 / M4A",
    )

    if aud_file is not None:
        aud_file.seek(0, 2)
        size_mb = aud_file.tell() / 1_048_576
        aud_file.seek(0)
        if size_mb > MAX_AUDIO_MB:
            st.error(f"File too large ({size_mb:.1f} MB). Limit is {MAX_AUDIO_MB} MB.")
        else:
            ev_key = f"ev_aud_{aud_file.name}_{st.session_state.case_id}"
            if ev_key not in st.session_state:
                ev_id = generate_evidence_id("audio")
                sha = sha256_file(aud_file)
                register_evidence(ev_id, aud_file.name, "audio", sha, st.session_state.investigator)
                st.session_state[ev_key] = ev_id
                st.session_state.audio_ev_id = ev_id
            else:
                st.session_state.audio_ev_id = st.session_state[ev_key]

            ev_id = st.session_state.audio_ev_id
            st.markdown(f"**Evidence ID:** `{ev_id}` &nbsp;|&nbsp; **Size:** {size_mb:.2f} MB")
            st.audio(aud_file)

            if st.button("🎙️ Transcribe & Extract Events", key="btn_analyse_audio"):
                with st.spinner("Transcribing audio…"):
                    try:
                        aud_file.seek(0)
                        aud_bytes = aud_file.read()
                        result = analyse_audio(
                            audio_bytes=aud_bytes,
                            filename=aud_file.name,
                            provider=st.session_state.provider,
                            api_key=st.session_state.api_key,
                            demo_mode=st.session_state.demo_mode,
                        )
                        st.session_state.audio_result = result
                        log_action("ANALYSIS", ev_id, "Audio transcription complete",
                                   st.session_state.investigator)
                        _rebuild_ledger()
                    except Exception as e:
                        st.error(f"Audio analysis failed: {e}")

            result = st.session_state.audio_result
            if result:
                st.markdown(_mode_badge(result.get("simulated", True)), unsafe_allow_html=True)
                st.divider()

                # Transcript
                st.subheader("📝 Transcript")
                raw = result.get("raw_text", "")
                st.text_area("Full Transcript", value=raw, height=200, disabled=True, key="transcript_display")

                # Segments table
                segments = result.get("segments", [])
                if segments:
                    st.subheader("🕐 Timestamped Segments")
                    seg_df = pd.DataFrame([
                        {
                            "Start":   f"{int(s.get('start',0))//60:02d}:{int(s.get('start',0))%60:02d}",
                            "End":     f"{int(s.get('end',0))//60:02d}:{int(s.get('end',0))%60:02d}",
                            "Speaker": s.get("speaker", "Unknown"),
                            "Text":    s.get("text", ""),
                        }
                        for s in segments
                    ])
                    st.dataframe(seg_df, use_container_width=True, hide_index=True)

                # Key events
                extracted = result.get("extracted", {})
                events = extracted.get("key_events", [])
                if events:
                    st.subheader("🔑 Key Forensic Events")
                    evt_df = pd.DataFrame(events)
                    st.dataframe(evt_df, use_container_width=True, hide_index=True)

                # Entities
                entities = extracted.get("entities", {})
                if entities:
                    st.subheader("🗂️ Extracted Entities")
                    cols = st.columns(len(entities))
                    for col, (cat, items) in zip(cols, entities.items()):
                        with col:
                            st.markdown(f"**{cat.upper()}**")
                            for item in items:
                                st.markdown(f"- {item}")

                with st.expander("🔧 Raw JSON Output"):
                    st.json({k: v for k, v in result.items() if k not in ("simulated", "raw_text")})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CCTV Video Analysis & Event Progression
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("CCTV Video Analysis & Event Progression")
    st.caption("Upload a CCTV video. AI samples frames, tracks persons/vehicles, and builds a crime timeline.")

    vid_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        key="vid_uploader",
        help=f"Max {MAX_VIDEO_MB} MB · MP4 / AVI / MOV",
    )

    if vid_file is not None:
        vid_file.seek(0, 2)
        size_mb = vid_file.tell() / 1_048_576
        vid_file.seek(0)
        if size_mb > MAX_VIDEO_MB:
            st.error(f"File too large ({size_mb:.1f} MB). Limit is {MAX_VIDEO_MB} MB.")
        else:
            ev_key = f"ev_vid_{vid_file.name}_{st.session_state.case_id}"
            if ev_key not in st.session_state:
                ev_id = generate_evidence_id("video")
                sha = sha256_file(vid_file)
                register_evidence(ev_id, vid_file.name, "video", sha, st.session_state.investigator)
                st.session_state[ev_key] = ev_id
                st.session_state.video_ev_id = ev_id
            else:
                st.session_state.video_ev_id = st.session_state[ev_key]

            ev_id = st.session_state.video_ev_id
            st.markdown(f"**Evidence ID:** `{ev_id}` &nbsp;|&nbsp; **Size:** {size_mb:.2f} MB")
            st.video(vid_file)

            if st.button("🎥 Analyse Video", key="btn_analyse_video"):
                progress_bar = st.progress(0, text="Sampling frames…")
                try:
                    vid_file.seek(0)
                    vid_bytes = vid_file.read()

                    def _prog(frac: float) -> None:
                        progress_bar.progress(min(frac, 1.0), text=f"Processing… {frac:.0%}")

                    result = analyse_video(
                        video_bytes=vid_bytes,
                        provider=st.session_state.provider,
                        api_key=st.session_state.api_key,
                        demo_mode=st.session_state.demo_mode,
                        progress_callback=_prog,
                    )
                    st.session_state.video_result = result
                    log_action("ANALYSIS", ev_id, "Video analysis complete",
                               st.session_state.investigator)
                    _rebuild_ledger()
                    progress_bar.progress(1.0, text="Analysis complete!")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Video analysis failed: {e}")

            result = st.session_state.video_result
            if result:
                st.markdown(_mode_badge(result.get("simulated", True)), unsafe_allow_html=True)

                # Metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Frames Sampled", result.get("frame_count", "?"))
                mc2.metric("Duration", f"{result.get('duration_sec', 0):.1f}s")
                mc3.metric("Tracks", len(result.get("track_ids", [])))
                mc4.metric("Events Logged", len(result.get("events", [])))

                # Event log
                events = result.get("events", [])
                if events:
                    st.subheader("📋 Chronological Event Log")
                    evt_df = pd.DataFrame(events)[
                        ["timestamp", "track_id", "event", "crime_category", "confidence"]
                    ]
                    evt_df.columns = ["Timestamp", "Track ID", "Event", "Crime Category", "Confidence"]
                    evt_df["Confidence"] = evt_df["Confidence"].apply(lambda x: f"{x:.0%}")
                    st.dataframe(evt_df, use_container_width=True, hide_index=True, height=350)

                    # Category breakdown
                    st.subheader("🏷️ Crime Category Breakdown")
                    cat_counts = pd.DataFrame(events)["crime_category"].value_counts().reset_index()
                    cat_counts.columns = ["Category", "Count"]
                    st.bar_chart(cat_counts.set_index("Category"))

                # Trajectory image
                traj = result.get("trajectory_image")
                if traj is not None:
                    st.subheader("🛣️ Movement Trajectories (Last Frame)")
                    st.image(traj, caption="Track trajectories overlaid on last sampled frame",
                             use_container_width=True)

                with st.expander("🔧 Raw JSON Events"):
                    st.json(events[:20])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Multimodal Fusion & Grounded Evidence Chat
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Multimodal Fusion & Grounded Evidence Chat")
    st.caption(
        "Master forensic ledger merging all evidence. Chat is strictly grounded in the evidence — "
        "the AI will only cite what exists in the case file."
    )

    ledger: pd.DataFrame = st.session_state.ledger

    if ledger.empty:
        st.info("ℹ️ No evidence analysed yet. Upload and analyse evidence in Tabs 1–3 to populate the ledger.")
    else:
        # ── Ledger summary metrics ─────────────────────────────────────────
        summary = ledger_summary(ledger)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Events", summary["total_events"])
        mc2.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.0%}")
        mc3.metric("Modalities", len(summary.get("modalities", {})))
        mc4.metric("Evidence IDs", len(summary.get("evidence_ids", [])))

        st.divider()

        # ── Filter controls ────────────────────────────────────────────────
        with st.expander("🔍 Filter & Sort Ledger", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                src_filter = st.multiselect(
                    "Source Type",
                    options=ledger["source_type"].unique().tolist(),
                    default=ledger["source_type"].unique().tolist(),
                    key="ledger_src_filter",
                )
            with fc2:
                cat_filter = st.multiselect(
                    "Crime Category",
                    options=ledger["crime_category"].unique().tolist(),
                    default=ledger["crime_category"].unique().tolist(),
                    key="ledger_cat_filter",
                )
            with fc3:
                min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, key="ledger_conf_filter")

        # Apply filters
        display_df = ledger.copy()
        if src_filter:
            display_df = display_df[display_df["source_type"].isin(src_filter)]
        if cat_filter:
            display_df = display_df[display_df["crime_category"].isin(cat_filter)]
        display_df = display_df[display_df["confidence"] >= min_conf]

        # ── Ledger table ───────────────────────────────────────────────────
        st.subheader(f"📊 Master Forensic Event Ledger ({len(display_df)} events)")
        st.dataframe(
            display_df.drop(columns=["raw_ref"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            height=320,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.0%"
                ),
            },
        )

        # ── Export ledger ──────────────────────────────────────────────────
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                "⬇️ Export Ledger as JSON",
                data=display_df.to_json(orient="records", indent=2),
                file_name=f"{st.session_state.case_id}_ledger.json",
                mime="application/json",
                key="dl_ledger_json",
            )
        with ec2:
            st.download_button(
                "⬇️ Export Ledger as CSV",
                data=display_df.to_csv(index=False),
                file_name=f"{st.session_state.case_id}_ledger.csv",
                mime="text/csv",
                key="dl_ledger_csv",
            )

        st.divider()

        # ── Grounded RAG Chat ──────────────────────────────────────────────
        st.subheader("🤖 Grounded Evidence Chat")

        if st.session_state.demo_mode:
            st.markdown(
                '<div class="demo-badge">DEMO MODE — Rule-based grounded responses with evidence citations</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="live-badge">LIVE AI — {st.session_state.provider} | Strict evidence grounding</div>',
                unsafe_allow_html=True,
            )

        st.caption(
            "Ask any question about the evidence. The AI will **only** answer from the case file "
            "and cite Evidence IDs. If the answer isn't in the file, it will say so."
        )

        # Suggested queries
        suggested = [
            "Was a weapon visible before the suspect fled?",
            "Describe the suspect based on all available evidence.",
            "What vehicles were identified in the case?",
            "What is the chronological sequence of events?",
            "What are the top investigative leads?",
        ]
        st.markdown("**💡 Suggested queries:**")
        query_cols = st.columns(len(suggested))
        for col, q in zip(query_cols, suggested):
            with col:
                if st.button(q[:35] + "…" if len(q) > 35 else q, key=f"sq_{hash(q)}"):
                    st.session_state["_pending_query"] = q

        # Chat input
        user_query = st.chat_input("Ask an investigative question about the case evidence…")
        if "_pending_query" in st.session_state:
            user_query = st.session_state.pop("_pending_query")

        if user_query:
            with st.spinner("Searching evidence and generating grounded answer…"):
                try:
                    chat_result = ask_question(
                        query=user_query,
                        ledger=ledger,
                        provider=st.session_state.provider,
                        api_key=st.session_state.api_key,
                        demo_mode=st.session_state.demo_mode,
                    )
                    st.session_state.chat_history.append({
                        "query":          user_query,
                        "answer":         chat_result["answer"],
                        "retrieved_rows": chat_result["retrieved_rows"],
                        "simulated":      chat_result["simulated"],
                    })
                    log_action("CHAT_QUERY", details=f"Q: {user_query[:80]}",
                               investigator=st.session_state.investigator)
                except Exception as e:
                    st.error(f"Chat error: {e}")

        # Display chat history
        for entry in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(entry["query"])
            with st.chat_message("assistant"):
                st.markdown(_mode_badge(entry.get("simulated", True)), unsafe_allow_html=True)
                st.markdown(entry["answer"])

                # Show retrieved evidence rows
                ret_rows = entry.get("retrieved_rows")
                if ret_rows is not None and not ret_rows.empty:
                    with st.expander(f"📎 Retrieved Evidence Rows ({len(ret_rows)})"):
                        st.dataframe(
                            ret_rows.drop(columns=["raw_ref"], errors="ignore"),
                            use_container_width=True,
                            hide_index=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Automated Forensic PDF Export
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("Automated Forensic PDF Export")
    st.caption("Generate a court-style forensic report with chain of custody, annotated images, full timeline, and chat excerpts.")

    # Pre-flight checks
    has_image = st.session_state.image_result is not None
    has_audio = st.session_state.audio_result is not None
    has_video = st.session_state.video_result is not None

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Image Evidence", "✅ Ready" if has_image else "⬜ None")
    cc2.metric("Audio Evidence", "✅ Ready" if has_audio else "⬜ None")
    cc3.metric("Video Evidence", "✅ Ready" if has_video else "⬜ None")

    if not (has_image or has_audio or has_video):
        st.warning("No analysis results yet. Analyse evidence in Tabs 1–3 before exporting.")
    else:
        st.info(
            f"Report will include: "
            f"{'Image ✅ ' if has_image else ''}"
            f"{'Audio ✅ ' if has_audio else ''}"
            f"{'Video ✅ ' if has_video else ''}"
            f"| Chat queries: {len(st.session_state.chat_history)} | "
            f"Ledger events: {len(st.session_state.ledger)}"
        )

        if not st.session_state.signed_off:
            st.warning("⚠️ Consider signing off in the sidebar before exporting.")

        if st.button("📄 Generate Forensic PDF Report", key="btn_gen_pdf"):
            with st.spinner("Building PDF report…"):
                try:
                    # Prepare annotated image bytes
                    ann_img: Optional[Image.Image] = st.session_state.annotated_img
                    ann_bytes: Optional[bytes] = None
                    if ann_img:
                        buf = io.BytesIO()
                        ann_img.convert("RGB").save(buf, format="JPEG", quality=85)
                        ann_bytes = buf.getvalue()

                    pdf_bytes = generate_pdf(
                        case_id=st.session_state.case_id,
                        investigator=st.session_state.investigator or "Not Specified",
                        provider=st.session_state.provider,
                        demo_mode=st.session_state.demo_mode,
                        custody_log=get_custody_log(),
                        evidence_registry=get_evidence_registry(),
                        image_result=st.session_state.image_result,
                        image_ev_id=st.session_state.image_ev_id,
                        annotated_img_bytes=ann_bytes,
                        audio_result=st.session_state.audio_result,
                        audio_ev_id=st.session_state.audio_ev_id,
                        video_result=st.session_state.video_result,
                        video_ev_id=st.session_state.video_ev_id,
                        ledger=st.session_state.ledger if not st.session_state.ledger.empty else None,
                        chat_history=st.session_state.chat_history,
                    )

                    log_action("EXPORT", details="PDF report generated",
                               investigator=st.session_state.investigator)

                    st.session_state["_pdf_bytes"] = pdf_bytes
                    st.success("✅ PDF report generated successfully!")

                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        if "_pdf_bytes" in st.session_state and st.session_state["_pdf_bytes"]:
            fname = f"{st.session_state.case_id}_forensic_report.pdf"
            st.download_button(
                label="⬇️ Download Forensic PDF Report",
                data=st.session_state["_pdf_bytes"],
                file_name=fname,
                mime="application/pdf",
                key="dl_pdf_btn",
            )
            st.markdown(
                "**⚠️ DISCLAIMER:** This PDF is AI-assisted. "
                "All findings require independent expert verification before legal use."
            )
