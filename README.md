# 🔬 CrimeVision AI
### Intelligent Multimodal Forensic Analysis Platform
> **Skill Wallet Capstone Project 2 — Generative AI / Criminal Justice & Public Safety**

---

## Architecture Overview (Data Flow)

```
Upload (img/audio/video)
  → custody.py        # SHA-256 hash, Evidence ID, chain-of-custody log
  → vision/audio/video.py  # analysis (LLM API or demo fallback)
  → fusion.py         # merge into master event ledger (pandas DataFrame)
  → chat.py           # TF-IDF retrieval → LLM grounded answer (cite Evidence IDs)
  → report.py         # ReportLab PDF: cover, custody, all tabs, chat, sign-off
```

All state lives in `st.session_state`. No external database. Zero hardcoded keys.

---

## Project Structure

```
crimevision-ai/
├── app.py                  # Streamlit entry, sidebar, all 5 tabs
├── config.py               # Constants, CSS, demo data
├── modules/
│   ├── custody.py          # IDs, hashing, audit log
│   ├── vision.py           # Image detection + annotation (PIL/OpenCV)
│   ├── audio.py            # Whisper transcription + event extraction
│   ├── video.py            # Frame sampling, centroid tracking, event log
│   ├── fusion.py           # Master ledger + TF-IDF retrieval
│   ├── chat.py             # Grounded RAG chat (anti-hallucination)
│   ├── report.py           # ReportLab PDF export
│   └── llm.py              # Groq/Gemini wrapper, JSON-safe parsing, retries
├── assets/                 # (optional sample demo files)
├── .streamlit/
│   └── config.toml         # Dark forensic theme
├── requirements.txt
├── packages.txt
└── README.md
```

---

## Quick Start — Local

```bash
# 1. Clone / navigate to project
cd crimevision-ai

# 2. Create virtual environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501` in **Demo Mode** (no API key required).

---

## Demo Mode (No API Key)

All five tabs work fully in demo mode:

| Feature | Demo Behaviour |
|---|---|
| Image detection | Pre-built bounding boxes + threat assessment |
| Audio transcription | Sample 911 call transcript with entities |
| Video analysis | Simulated tracks + chronological event log |
| Grounded chat | Rule-based answers citing real Evidence IDs |
| PDF export | Full report with all demo data included |

Every simulated output is clearly labelled **`SIMULATED`** in the UI and PDF.

---

## Live AI Mode

### Groq (Recommended)
1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Models used:
   - **Chat:** `llama-3.3-70b-versatile`
   - **Vision:** `meta-llama/llama-4-scout-17b-16e-instruct`
   - **Audio:** `whisper-large-v3`

### Gemini
1. Get an API key at [aistudio.google.com](https://aistudio.google.com)
2. Models used:
   - **Chat + Vision:** `gemini-2.0-flash`
   - **Audio:** ⚠️ Not supported — falls back to demo transcript

Enter the key in the sidebar. The badge switches to **`● LIVE AI`** automatically.

---

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo and set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, add your API key (optional):
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   *(The app also accepts keys typed in the sidebar at runtime.)*
5. Click **Deploy**. The `packages.txt` installs system dependencies automatically.

---

## Test Checklist (Demo Mode)

Run through these checks with no API key to verify every tab:

### Tab 1 — Visual Evidence
- [ ] Upload any JPG/PNG image
- [ ] Click **Analyse Crime Scene Image**
- [ ] Verify `SIMULATED` badge appears
- [ ] Verify bounding boxes rendered on annotated image
- [ ] Verify detections table shows class, confidence, bbox, notes
- [ ] Verify investigative leads list populated
- [ ] Expand Raw JSON — valid JSON with correct schema

### Tab 2 — Audio / Witness
- [ ] Upload any WAV/MP3/M4A file (or use a dummy 1-second file)
- [ ] Click **Transcribe & Extract Events**
- [ ] Verify `SIMULATED` badge appears
- [ ] Verify transcript text area populated
- [ ] Verify timestamped segments table
- [ ] Verify key events and entities extracted

### Tab 3 — CCTV Video
- [ ] Upload any MP4 file
- [ ] Click **Analyse Video**
- [ ] Progress bar advances to 100%
- [ ] Verify `SIMULATED` badge
- [ ] Verify 4 metric cards (frames, duration, tracks, events)
- [ ] Verify chronological event log table
- [ ] Verify crime category bar chart

### Tab 4 — Fusion & Chat
- [ ] Switch to Tab 4 after completing at least one of Tabs 1–3
- [ ] Verify master ledger is populated with events from all completed tabs
- [ ] Filter by source type — table updates
- [ ] Export ledger as JSON and CSV
- [ ] Ask: *"Was a weapon visible before the suspect fled?"*
- [ ] Verify answer cites Evidence IDs like `[EV-IMG-001]`
- [ ] Verify retrieved evidence rows shown beside answer
- [ ] Try the suggested query buttons

### Tab 5 — PDF Export
- [ ] Click **Generate Forensic PDF Report**
- [ ] Verify success message
- [ ] Click **Download Forensic PDF Report**
- [ ] Open PDF — verify cover page, chain of custody, annotated image, timeline, chat Q&A, disclaimer, signature block

---

## Limitations & Disclaimer

> **AI-assisted analysis is investigative support only.**
> All outputs — detections, transcriptions, event classifications, and chat answers —
> are probabilistic and may contain errors. They must be independently verified by
> qualified forensic professionals before use in any legal proceeding.
> This tool does not constitute legal evidence.

Additional limitations:
- Video tracking uses centroid matching (no deep learning by default); accuracy is lower than production tracking systems.
- Whisper transcription accuracy depends on audio quality and accents.
- Vision detection via LLM may miss small or partially occluded objects.
- Chain-of-custody log is session-scoped; restart the app and the log is cleared (no persistence by design — zero external database constraint).

---

## Security Notes

- API keys are accepted via `st.text_input(type="password")` — never logged or displayed.
- SHA-256 hashes provide file integrity verification for each piece of evidence.
- Never commit API keys to version control; use Streamlit Secrets for deployment.
