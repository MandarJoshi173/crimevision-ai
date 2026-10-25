"""
report.py — Automated forensic PDF export using ReportLab.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
    PageBreak,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# Colour palette (forensic dark theme approximated in PDF)
# ---------------------------------------------------------------------------

_DARK_BG      = colors.HexColor("#0A0E1A")
_ACCENT_GREEN = colors.HexColor("#00FF9C")
_ACCENT_AMBER = colors.HexColor("#FF8C00")
_ACCENT_RED   = colors.HexColor("#FF3B3B")
_LIGHT_TEXT   = colors.HexColor("#E2E8F0")
_MUTED        = colors.HexColor("#64748B")
_PANEL_BG     = colors.HexColor("#111827")
_BORDER       = colors.HexColor("#1E2D40")

PAGE_W, PAGE_H = A4


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CVTitle",
            parent=base["Title"],
            fontSize=22,
            textColor=_ACCENT_GREEN,
            fontName="Courier-Bold",
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "CVSubtitle",
            fontSize=9,
            textColor=_MUTED,
            fontName="Courier",
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "CVH1",
            fontSize=13,
            textColor=_ACCENT_GREEN,
            fontName="Courier-Bold",
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "CVH2",
            fontSize=10,
            textColor=_ACCENT_AMBER,
            fontName="Courier-Bold",
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "CVBody",
            fontSize=8,
            textColor=colors.HexColor("#CBD5E1"),
            fontName="Courier",
            leading=12,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "CVCaption",
            fontSize=7,
            textColor=_MUTED,
            fontName="Courier",
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "disclaimer": ParagraphStyle(
            "CVDisclaimer",
            fontSize=7,
            textColor=_ACCENT_AMBER,
            fontName="Courier-Bold",
            alignment=TA_CENTER,
            borderPad=6,
            spaceAfter=8,
        ),
        "mono_small": ParagraphStyle(
            "CVMono",
            fontSize=7,
            textColor=colors.HexColor("#94A3B8"),
            fontName="Courier",
            leading=10,
        ),
    }


def _header_footer(canvas_obj: canvas.Canvas, doc: Any) -> None:
    """Draw page header and footer on every page."""
    canvas_obj.saveState()
    w, h = PAGE_W, PAGE_H

    # Header bar
    canvas_obj.setFillColor(_DARK_BG)
    canvas_obj.rect(0, h - 1.2 * cm, w, 1.2 * cm, fill=1, stroke=0)
    canvas_obj.setFillColor(_ACCENT_GREEN)
    canvas_obj.rect(0, h - 1.2 * cm, w, 1.5 * mm, fill=1, stroke=0)
    canvas_obj.setFont("Courier-Bold", 8)
    canvas_obj.setFillColor(_ACCENT_GREEN)
    canvas_obj.drawString(1 * cm, h - 0.85 * cm, "CRIMEVISION AI — FORENSIC INTELLIGENCE REPORT")
    canvas_obj.setFont("Courier", 7)
    canvas_obj.setFillColor(_MUTED)
    canvas_obj.drawRightString(w - 1 * cm, h - 0.85 * cm, f"CONFIDENTIAL — LAW ENFORCEMENT USE ONLY")

    # Footer bar
    canvas_obj.setFillColor(_DARK_BG)
    canvas_obj.rect(0, 0, w, 0.9 * cm, fill=1, stroke=0)
    canvas_obj.setFillColor(_BORDER)
    canvas_obj.rect(0, 0.9 * cm, w, 0.5 * mm, fill=1, stroke=0)
    canvas_obj.setFont("Courier", 7)
    canvas_obj.setFillColor(_MUTED)
    canvas_obj.drawString(1 * cm, 0.4 * cm, f"Page {doc.page}")
    canvas_obj.drawCentredString(w / 2, 0.4 * cm,
        "AI-ASSISTED ANALYSIS — REQUIRES HUMAN EXPERT VERIFICATION BEFORE LEGAL USE")
    canvas_obj.drawRightString(w - 1 * cm, 0.4 * cm,
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    canvas_obj.restoreState()


def _table_style(header_bg: Any = _PANEL_BG) -> TableStyle:
    """Return a consistent forensic-themed table style."""
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _ACCENT_GREEN),
        ("FONTNAME",     (0, 0), (-1, 0),  "Courier-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_DARK_BG, _PANEL_BG]),
        ("TEXTCOLOR",    (0, 1), (-1, -1), colors.HexColor("#CBD5E1")),
        ("FONTNAME",     (0, 1), (-1, -1), "Courier"),
        ("FONTSIZE",     (0, 1), (-1, -1), 7),
        ("GRID",         (0, 0), (-1, -1), 0.3, _BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("ROWHEIGHT",    (0, 0), (-1, -1), 14),
    ])


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _cover_page(
    case_id: str,
    investigator: str,
    provider: str,
    demo_mode: bool,
    s: dict,
) -> list:
    """Build the cover page flowables."""
    story = []

    story.append(Spacer(1, 2.5 * cm))
    story.append(Paragraph("🔬 CRIMEVISION AI", s["title"]))
    story.append(Paragraph("Intelligent Multimodal Forensic Analysis Platform", s["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_ACCENT_GREEN, spaceAfter=16))

    meta = [
        ["CASE ID",       case_id],
        ["INVESTIGATOR",  investigator or "Not Specified"],
        ["GENERATED",     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["ANALYSIS MODE", "⚠ SIMULATED (DEMO)" if demo_mode else "LIVE AI"],
        ["CLASSIFICATION","CONFIDENTIAL — LAW ENFORCEMENT USE ONLY"],
    ]
    t = Table(meta, colWidths=[5 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), _PANEL_BG),
        ("TEXTCOLOR",   (0, 0), (0, -1), _ACCENT_GREEN),
        ("FONTNAME",    (0, 0), (0, -1), "Courier-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (1, 0), (1, -1), _LIGHT_TEXT),
        ("FONTNAME",    (1, 0), (1, -1), "Courier"),
        ("GRID",        (0, 0), (-1, -1), 0.5, _BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("BACKGROUND",  (0, 4), (1, 4), colors.HexColor("#FF3B3B22")),
        ("TEXTCOLOR",   (1, 4), (1, 4), _ACCENT_RED),
    ]))
    story.append(t)
    story.append(Spacer(1, 1 * cm))

    story.append(Paragraph(
        "⚠  DISCLAIMER: This report was generated with AI assistance. "
        "All findings are investigative support only and must be independently verified "
        "by qualified forensic experts before use in legal proceedings. "
        "AI outputs may contain errors and should not be treated as conclusive evidence.",
        s["disclaimer"],
    ))

    story.append(PageBreak())
    return story


def _custody_section(
    custody_log: list[dict],
    evidence_registry: dict,
    s: dict,
) -> list:
    """Build the chain-of-custody section."""
    story = [Paragraph("CHAIN OF CUSTODY", s["h1"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT_GREEN, spaceAfter=8))

    # Evidence registry table
    if evidence_registry:
        story.append(Paragraph("Evidence Registry", s["h2"]))
        headers = ["Evidence ID", "Filename", "Type", "SHA-256 (first 24 chars)", "Uploaded (UTC)"]
        rows = [headers]
        for ev_id, rec in evidence_registry.items():
            rows.append([
                ev_id,
                rec.get("filename", "?")[:30],
                rec.get("source_type", "?").upper(),
                rec.get("sha256", "?")[:24] + "…",
                rec.get("uploaded_at", "?"),
            ])
        t = Table(rows, colWidths=[2.8 * cm, 4 * cm, 2 * cm, 5.5 * cm, 3.5 * cm])
        t.setStyle(_table_style())
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

    # Audit log
    if custody_log:
        story.append(Paragraph("Audit Log", s["h2"]))
        headers = ["Timestamp (UTC)", "Action", "Evidence ID", "Investigator", "Details"]
        rows = [headers]
        for rec in custody_log:
            rows.append([
                rec.get("timestamp", "?"),
                rec.get("action", "?"),
                rec.get("evidence_id", "—"),
                rec.get("investigator", "?")[:20],
                rec.get("details", "")[:50],
            ])
        t = Table(rows, colWidths=[4 * cm, 2.5 * cm, 2.8 * cm, 3 * cm, 5.5 * cm])
        t.setStyle(_table_style())
        story.append(t)

    story.append(PageBreak())
    return story


def _vision_section(
    image_result: Optional[dict],
    image_ev_id: str,
    annotated_img_bytes: Optional[bytes],
    s: dict,
) -> list:
    """Build the visual evidence section."""
    story = [Paragraph("TAB 1 — CRIME SCENE VISUAL EVIDENCE", s["h1"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT_GREEN, spaceAfter=8))

    if not image_result:
        story.append(Paragraph("No image evidence analysed.", s["body"]))
        return story

    # Scene summary
    meta = [
        ["Evidence ID",          image_ev_id],
        ["Threat Level",         image_result.get("threat_level", "?")],
        ["Scene Classification", image_result.get("scene_classification", "?")],
        ["Analysis Mode",        "SIMULATED" if image_result.get("simulated") else "LIVE AI"],
    ]
    t = Table(meta, colWidths=[4.5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), _PANEL_BG),
        ("TEXTCOLOR",   (0, 0), (0, -1), _ACCENT_GREEN),
        ("FONTNAME",    (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",   (1, 0), (1, -1), _LIGHT_TEXT),
        ("GRID",        (0, 0), (-1, -1), 0.3, _BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1),  4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    # Caption
    caption = image_result.get("scene_caption", "")
    if caption:
        story.append(Paragraph(f"Scene Caption: {caption}", s["body"]))
        story.append(Spacer(1, 0.2 * cm))

    # Annotated image
    if annotated_img_bytes:
        try:
            img_buf = io.BytesIO(annotated_img_bytes)
            rl_img = RLImage(img_buf, width=14 * cm, height=9 * cm, kind="proportional")
            story.append(rl_img)
            story.append(Paragraph(f"Figure 1 — Annotated crime scene ({image_ev_id})", s["caption"]))
        except Exception:
            story.append(Paragraph("[Image rendering failed]", s["body"]))

    # Detections table
    items = image_result.get("detected_items", [])
    if items:
        story.append(Paragraph("Detected Items", s["h2"]))
        headers = ["Class", "Confidence", "Bounding Box [x1,y1,x2,y2]", "Notes"]
        rows = [headers]
        for item in items:
            rows.append([
                item.get("class", "?").replace("_", " ").title(),
                f"{item.get('confidence', 0):.1%}",
                str(item.get("bbox", [])),
                item.get("notes", "")[:60],
            ])
        t = Table(rows, colWidths=[3.5 * cm, 2.5 * cm, 5.5 * cm, 6.3 * cm])
        t.setStyle(_table_style())
        story.append(t)

    # Leads
    leads = image_result.get("investigative_leads", [])
    if leads:
        story.append(Paragraph("Investigative Leads", s["h2"]))
        for i, lead in enumerate(leads, 1):
            story.append(Paragraph(f"{i}. {lead}", s["body"]))

    story.append(PageBreak())
    return story


def _audio_section(audio_result: Optional[dict], audio_ev_id: str, s: dict) -> list:
    """Build the audio transcription section."""
    story = [Paragraph("TAB 2 — WITNESS AUDIO TRANSCRIPTION", s["h1"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT_GREEN, spaceAfter=8))

    if not audio_result:
        story.append(Paragraph("No audio evidence analysed.", s["body"]))
        return story

    story.append(Paragraph(
        f"Evidence ID: {audio_ev_id} | Mode: {'SIMULATED' if audio_result.get('simulated') else 'LIVE AI'}",
        s["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))

    # Key events
    extracted = audio_result.get("extracted", {})
    events = extracted.get("key_events", [])
    if events:
        story.append(Paragraph("Key Events", s["h2"]))
        headers = ["Timestamp", "Event Description"]
        rows = [headers] + [[e.get("timestamp", "?"), e.get("event", "")] for e in events]
        t = Table(rows, colWidths=[2.5 * cm, 15.3 * cm])
        t.setStyle(_table_style())
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))

    # Entities
    entities = extracted.get("entities", {})
    if entities:
        story.append(Paragraph("Extracted Entities", s["h2"]))
        for cat, items in entities.items():
            story.append(Paragraph(f"<b>{cat.upper()}:</b>", s["body"]))
            for item in items:
                story.append(Paragraph(f"  • {item}", s["body"]))

    story.append(Spacer(1, 0.3 * cm))

    # Transcript excerpt
    raw = audio_result.get("raw_text", "")
    if raw:
        story.append(Paragraph("Transcript Excerpt", s["h2"]))
        excerpt = raw[:1200] + ("…" if len(raw) > 1200 else "")
        story.append(Paragraph(excerpt.replace("\n", "<br/>"), s["mono_small"]))

    story.append(PageBreak())
    return story


def _video_section(video_result: Optional[dict], video_ev_id: str, s: dict) -> list:
    """Build the video analysis section."""
    story = [Paragraph("TAB 3 — CCTV VIDEO ANALYSIS", s["h1"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT_GREEN, spaceAfter=8))

    if not video_result:
        story.append(Paragraph("No video evidence analysed.", s["body"]))
        return story

    story.append(Paragraph(
        f"Evidence ID: {video_ev_id} | Frames Sampled: {video_result.get('frame_count', '?')} | "
        f"Duration: {video_result.get('duration_sec', '?')}s | "
        f"Mode: {'SIMULATED' if video_result.get('simulated') else 'LIVE AI'}",
        s["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))

    events = video_result.get("events", [])
    if events:
        story.append(Paragraph("Chronological Event Log", s["h2"]))
        headers = ["Timestamp", "Track ID", "Event", "Crime Category", "Confidence"]
        rows = [headers]
        for evt in events:
            rows.append([
                evt.get("timestamp", "?"),
                evt.get("track_id", "?"),
                evt.get("event", "")[:55],
                evt.get("crime_category", "?"),
                f"{evt.get('confidence', 0):.0%}",
            ])
        t = Table(rows, colWidths=[2.5 * cm, 2.5 * cm, 8 * cm, 3.5 * cm, 2.3 * cm])
        t.setStyle(_table_style())
        story.append(t)

    story.append(PageBreak())
    return story


def _ledger_section(ledger: Optional[pd.DataFrame], s: dict) -> list:
    """Build the master event ledger section."""
    story = [Paragraph("TAB 4 — MASTER FORENSIC EVENT LEDGER", s["h1"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT_GREEN, spaceAfter=8))

    if ledger is None or ledger.empty:
        story.append(Paragraph("No ledger data available.", s["body"]))
        return story

    story.append(Paragraph(f"Total Events: {len(ledger)}", s["body"]))
    story.append(Spacer(1, 0.2 * cm))

    headers = ["Evidence ID", "Source", "Timestamp", "Description", "Confidence", "Category"]
    rows = [headers]
    for _, row in ledger.iterrows():
        rows.append([
            row.get("evidence_id", "?"),
            row.get("source_type", "?"),
            row.get("timestamp", "?"),
            str(row.get("description", ""))[:55],
            f"{float(row.get('confidence', 0)):.0%}",
            row.get("crime_category", "?"),
        ])

    t = Table(rows, colWidths=[2.8 * cm, 2 * cm, 2.3 * cm, 8 * cm, 2.3 * cm, 3 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(PageBreak())
    return story


def _chat_section(chat_history: list[dict], s: dict) -> list:
    """Build the chat Q&A section."""
    story = [Paragraph("GROUNDED EVIDENCE CHAT — Q&A LOG", s["h1"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT_GREEN, spaceAfter=8))

    if not chat_history:
        story.append(Paragraph("No chat queries recorded.", s["body"]))
        return story

    for i, entry in enumerate(chat_history, 1):
        story.append(Paragraph(f"Q{i}: {entry.get('query', '')}", s["h2"]))
        answer = entry.get("answer", "").replace("\n", "<br/>")
        story.append(Paragraph(answer, s["body"]))
        mode = "SIMULATED" if entry.get("simulated") else "LIVE AI"
        story.append(Paragraph(f"[Analysis mode: {mode}]", s["mono_small"]))
        story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())
    return story


def _signature_section(investigator: str, case_id: str, s: dict) -> list:
    """Build the signature block section."""
    story = [Paragraph("SIGN-OFF & CERTIFICATION", s["h1"])]
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT_GREEN, spaceAfter=12))

    story.append(Paragraph(
        "I certify that the information contained in this report accurately reflects the "
        "AI-assisted analysis performed on the listed evidence items. This report is intended "
        "as an investigative aid and has been reviewed for consistency with known facts.",
        s["body"],
    ))
    story.append(Spacer(1, 1 * cm))

    sig_data = [
        ["Case ID:",        case_id],
        ["Investigator:",   investigator or "________________"],
        ["Date (UTC):",     datetime.now(timezone.utc).strftime("%Y-%m-%d")],
        ["Signature:",      "_" * 30],
        ["Badge / ID:",     "________________"],
    ]
    t = Table(sig_data, colWidths=[4 * cm, 13 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1), _ACCENT_GREEN),
        ("TEXTCOLOR",   (1, 0), (1, -1), _LIGHT_TEXT),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LINEBELOW",   (0, -1), (-1, -1), 0.5, _BORDER),
    ]))
    story.append(t)

    story.append(Spacer(1, 1.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=_BORDER, spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This document was generated by CrimeVision AI, an AI-assisted forensic "
        "analysis tool. AI outputs are probabilistic and may contain errors. This report does NOT "
        "constitute legal evidence on its own. All conclusions must be independently verified by "
        "qualified forensic professionals before use in any legal proceeding.",
        s["disclaimer"],
    ))
    story.append(Paragraph(
        "Skill Wallet Capstone Project 2 — Generative AI / Criminal Justice & Public Safety",
        s["caption"],
    ))
    return story


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def generate_pdf(
    case_id: str,
    investigator: str,
    provider: str,
    demo_mode: bool,
    custody_log: list[dict],
    evidence_registry: dict,
    image_result: Optional[dict],
    image_ev_id: str,
    annotated_img_bytes: Optional[bytes],
    audio_result: Optional[dict],
    audio_ev_id: str,
    video_result: Optional[dict],
    video_ev_id: str,
    ledger: Optional[pd.DataFrame],
    chat_history: list[dict],
) -> bytes:
    """
    Generate the full forensic PDF report.

    Args:
        case_id:            Case identifier string.
        investigator:       Investigator name.
        provider:           LLM provider name.
        demo_mode:          Whether running in demo mode.
        custody_log:        Chain-of-custody audit records.
        evidence_registry:  Uploaded file registry.
        image_result:       Vision analysis result dict.
        image_ev_id:        Image evidence ID.
        annotated_img_bytes: JPEG bytes of the annotated image.
        audio_result:       Audio analysis result dict.
        audio_ev_id:        Audio evidence ID.
        video_result:       Video analysis result dict.
        video_ev_id:        Video evidence ID.
        ledger:             Master event ledger DataFrame.
        chat_history:       List of chat Q&A dicts.

    Returns:
        PDF file as bytes.
    """
    buf = io.BytesIO()
    s = _styles()

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.5 * cm,
        title=f"CrimeVision AI Report — {case_id}",
        author=investigator or "CrimeVision AI",
        subject="Forensic Intelligence Report",
    )

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height - 0.5 * cm,
        id="main_frame",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=_header_footer)
    doc.addPageTemplates([template])

    story: list = []
    story += _cover_page(case_id, investigator, provider, demo_mode, s)
    story += _custody_section(custody_log, evidence_registry, s)
    story += _vision_section(image_result, image_ev_id, annotated_img_bytes, s)
    story += _audio_section(audio_result, audio_ev_id, s)
    story += _video_section(video_result, video_ev_id, s)
    story += _ledger_section(ledger, s)
    story += _chat_section(chat_history, s)
    story += _signature_section(investigator, case_id, s)

    doc.build(story)
    buf.seek(0)
    return buf.read()
