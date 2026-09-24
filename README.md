# AI-Based Smart Meeting Summarization
   🔗 **Live Demo:** [https://meeting-summarizer-ibm.streamlit.app](https://meeting-summarizer-ibm.streamlit.app/)

Convert meeting audio or video into concise summaries, key points, decisions and action items using AI — running **100% locally** on a Windows laptop (CPU only, no paid APIs).

> B.Tech Project • Python • Streamlit • faster-whisper • DistilBART • spaCy

---

## 1. Final project architecture

```
 ┌──────────────────────┐     ┌─────────────────────────────────────────────────────────┐
 │  Streamlit Web UI    │     │                 services/pipeline.py                     │
 │  (app.py, localhost) │ ──► │                                                          │
 │  • upload MP3/WAV/   │     │  1 video_processor.py   → MP4/MP3/M4A → 16 kHz mono WAV  │
 │    M4A/MP4           │     │  2 speech_to_text.py    → faster-whisper (CPU, int8)     │
 │  • or paste text     │     │  3 preprocessing.py     → remove fillers/noise/repeats   │
 │  • settings sidebar  │     │  4 information_extractor.py → spaCy NER + rules + TextRank│
 └──────────────────────┘     │  5 summarizer.py        → DistilBART-SAMSum (or TextRank)│
            ▲                 │  6 build structured result (dict)                       │
            │                 └─────────────────────────────────────────────────────────┘
            │                                         │
            └──── cards / tables / downloads ◄── utils/output_utils.py (TXT, JSON, PDF)
```

The **transcript-only mode** skips steps 1–2 and starts from step 3.

## 2. Technology / model selection (and why)

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Easiest ecosystem for AI/NLP |
| UI | **Streamlit** | Professional web UI in pure Python, runs on `localhost`, no HTML/JS needed |
| Audio/video decoding | **PyAV** (`av`) | Contains FFmpeg inside the pip package → **no separate FFmpeg install on Windows** |
| Speech-to-Text | **faster-whisper** (Whisper `base`, int8, CPU) | Same accuracy as OpenAI Whisper but ~4× faster and uses less RAM on CPU; works offline after first download |
| Summarization | **`philschmid/distilbart-cnn-12-6-samsum`** | DistilBART fine-tuned on **SAMSum (dialogue summaries)** → suits meetings; ~40% smaller/faster than BART-large; runs on CPU. A large LLM (7B+) would need 8–16 GB RAM/GPU and be too slow for a demo |
| Fallback summarizer | **TextRank** (own implementation with NumPy) | No download, instant, always works (offline/low-RAM) |
| Entities & topics | **spaCy `en_core_web_sm`** | Small (12 MB), fast Named Entity Recognition and noun phrases |
| Decisions, action items, deadlines | **Rule-based NLP (regex patterns)** | Copies names and dates *exactly*, so nothing is invented. Missing info → "Not specified" |
| PDF | **fpdf2** | Pure Python, no extra software |

## 3. Folder structure

```
meeting-summarizer/
├── app.py                      # Streamlit web application (UI)
├── config.py                   # All settings (model sizes, limits, folders)
├── download_models.py          # Pre-downloads all models (run once with internet)
├── requirements.txt
├── setup_windows.bat           # One-click setup
├── run_app.bat                 # One-click start
├── README.md
├── .streamlit/config.toml      # Upload limit (500 MB) + light theme
├── services/
│   ├── pipeline.py             # Connects all steps (can also run from command line)
│   ├── video_processor.py      # Step 1: audio extraction / conversion
│   ├── speech_to_text.py       # Step 2: faster-whisper ASR
│   ├── preprocessing.py        # Step 3: text cleaning + speaker detection
│   ├── information_extractor.py# Step 4: topics, key points, decisions, tasks, deadlines, entities
│   └── summarizer.py           # Step 5: DistilBART (abstractive) + TextRank (extractive)
├── utils/
│   ├── file_utils.py           # Upload validation, temp files
│   └── output_utils.py         # TXT / JSON / PDF generation
├── sample_data/
│   ├── sample_transcript.txt   # 4-person college project meeting
│   ├── sample_meeting.mp3      # Same meeting, synthetic voices (3.4 min)
│   └── sample_meeting.mp4      # Same audio as a video file
├── models/                     # Downloaded AI models are stored here
├── outputs/                    # Every result is auto-saved here (TXT, JSON, PDF)
└── docs/
    ├── PROJECT_REPORT.md       # Academic documentation
    └── VIVA_PREPARATION.md     # 2-min & 5-min scripts + viva Q&A
```

## 4. System requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / 11 (64-bit) | Windows 11 |
| CPU | Any 4-core (Intel i3 / Ryzen 3) | Intel i5 / Ryzen 5 or better |
| RAM | 4 GB (use `tiny` + Fast/Extractive mode) | **8 GB or more** |
| GPU | **Not required** | — |
| Free disk | ~4 GB | 6 GB |
| Python | 3.10 – 3.12 | **3.11** |
| Internet | Only for installation + first model download | — |

Disk usage: Python packages ≈ 2–2.5 GB (mostly PyTorch), Whisper base ≈ 145 MB, DistilBART ≈ 1.2 GB, spaCy ≈ 12 MB.

---

## 5. Windows installation (step by step)

### Option A — one click
1. Install **Python 3.11** from https://www.python.org/downloads/ → on the first installer screen tick **"Add python.exe to PATH"**.
2. Extract this project folder (e.g. to `C:\Projects\meeting-summarizer`).
3. Double-click **`setup_windows.bat`** (takes 5–15 min, downloads ~3 GB once).
4. Double-click **`run_app.bat`**.

### Option B — manual commands (Command Prompt)

```bat
:: 1. Check Python (should print 3.10, 3.11 or 3.12)
python --version

:: 2. Go to the project folder
cd C:\Projects\meeting-summarizer

:: 3. Create a virtual environment
python -m venv venv

:: 4. Activate it (Command Prompt)
venv\Scripts\activate

:: 5. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 6. Install the spaCy English model
python -m spacy download en_core_web_sm

:: 7. (Recommended) Download AI models now so the demo works offline
python download_models.py
```

**PowerShell users:** activate with
```powershell
.\venv\Scripts\Activate.ps1
```
If you get *"running scripts is disabled on this system"*, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Do I need FFmpeg?
**No.** `faster-whisper` installs **PyAV**, which already contains the FFmpeg libraries, so MP3/WAV/M4A/MP4 all work out of the box. (FFmpeg is only used as a fallback if PyAV is missing. If you still want it: `winget install Gyan.FFmpeg`, then open a new terminal and check `ffmpeg -version`.)

### Internet
Needed **only** for `pip install` and the **first** time each model is used (downloaded into the `models/` folder). After running `download_models.py`, you can switch Wi-Fi off — the app loads the models locally.

---

## 6. How to run the project

```bat
cd C:\Projects\meeting-summarizer
venv\Scripts\activate
streamlit run app.py
```

* The browser opens automatically at **http://localhost:8501** (if not, open that address manually).
* **Stop the app:** click the terminal window and press **Ctrl + C** (then close the terminal).
* Next time just run `run_app.bat`.

### Demo in 60 seconds
1. Sidebar → **Load sample transcript** → **Generate Summary** (works even without audio).
2. Switch to **Upload Audio/Video** → upload `sample_data/sample_meeting.mp3` or `.mp4` → **Generate Summary**.
3. Show the tabs: Summary, Key Points, Decisions, Action Items, Topics & Entities, Deadlines, Transcript (original vs cleaned).
4. Download **TXT / JSON / PDF** (also auto-saved in `outputs/`).

> The sample audio uses synthetic (robotic) voices. For a more impressive demo, record your team reading `sample_transcript.txt` with the Windows **Voice Recorder / Sound Recorder** app (saves `.m4a`) and upload it.

### Command-line test (no UI)
```bat
python -m services.pipeline sample_data\sample_transcript.txt extractive
python -m services.pipeline sample_data\sample_transcript.txt abstractive
```

---

## 7. How the complete workflow works

1. **Processing file** – upload is validated (format, size, empty file) and saved to `outputs/temp`. `video_processor.py` uses PyAV to decode it; for MP4 only the audio track is used. Audio is resampled to **16 kHz mono WAV** (Whisper's input format). Audio shorter than 2 s or longer than 120 min is rejected with a clear message.
2. **Transcribing meeting** – faster-whisper converts speech → text with timestamps. Voice Activity Detection (VAD) skips silence. Progress bar follows the audio position.
3. **Cleaning transcript** – `preprocessing.py` removes noise tags (`[Music]`, `(inaudible)`), filler words (um, uh, hmm), stutters ("the the"), back-to-back repeated sentences, fixes spacing/capitals, and detects speakers written as `Name: text`. Content words are never removed.
4. **Extracting important information** – spaCy finds people, organisations, dates; noun phrases become **topics**; **TextRank** ranks **key points**; regex rules find **decisions** ("we decided", "let's go with", "agreed"), **action items** ("I will…", "Rahul, can you…", "Sneha will…", "we need to…"), the **responsible person** (named person, the speaker for "I will", or whoever replies "Sure" to "can you…?") and **deadlines** ("by Friday", "25th March", "within five days", "one week before…").
5. **Generating summary** – DistilBART-SAMSum reads the dialogue (split into ~550-word chunks if long) and writes an abstractive summary. If the model can't load (no internet on first run, low RAM), the app automatically falls back to TextRank and shows a warning.
6. **Preparing final result** – everything is combined into one dictionary → shown as cards/tables, downloadable as TXT/JSON/PDF, and saved in `outputs/`.

---

## 8. Performance and model sizes

Change the Whisper size in the **sidebar** (or `DEFAULT_WHISPER_MODEL` in `config.py`):

| Whisper size | Download | RAM | Speed on a typical i5 laptop CPU* | Use when |
|---|---|---|---|---|
| `tiny` | ~75 MB | ~0.5 GB | ~10–20 s per 3-min meeting | Old/slow laptop, 4 GB RAM |
| **`base`** | ~145 MB | ~0.7 GB | ~20–60 s per 3-min meeting | **Recommended for the demo** |
| `small` | ~480 MB | ~1.2 GB | ~1–3 min per 3-min meeting | Accents / noisy audio, more accuracy |

\*Rough estimates — actual speed depends on your CPU. Summarization with DistilBART takes roughly 5–30 s per chunk on CPU.

**Fast / lightweight configuration** (4 GB RAM or slow laptop): Whisper **`tiny`** + **"Fast (Extractive – TextRank)"** in the sidebar. No 1.2 GB summarizer download is needed.

**Change the summarization model** in `config.py` → `SUMMARIZATION_MODEL` (e.g. `philschmid/bart-large-cnn-samsum` for higher quality but slower).

**Have an NVIDIA GPU?** In `config.py` set `WHISPER_DEVICE = "cuda"` and `WHISPER_COMPUTE_TYPE = "float16"` (needs CUDA). Not required.

---

## 9. Error handling built in

| Situation | What the user sees |
|---|---|
| Unsupported format (e.g. `.avi`, `.docx`) | "Unsupported file format…" (uploader also restricts types) |
| Corrupted / fake media file | "The file could not be opened. It may be corrupted…" |
| Video without audio | "No audio track was found in this file…" |
| Empty transcript / too short (< 15 words) | Clear message asking for more text |
| Very short audio (< 2 s) | "The audio is too short…" |
| Very long meeting (> 30 min / > 120 min) | Warning that it will take time / limit message |
| Silent audio | "No speech could be detected…" |
| Model download failure (no internet) | Message explaining internet is needed once; summarizer falls back to TextRank |
| Out of memory | Advice to use `tiny` model / extractive mode |
| spaCy model missing | App still runs in basic mode; sidebar shows a warning |
| Any other error | Friendly message + "Technical details" expander (no crash) |

---

## 10. Possible improvements

* Speaker diarization (who spoke when) with `pyannote.audio` for audio files.
* Live microphone recording and real-time summarization.
* A small local LLM (e.g. via Ollama) for richer action-item wording when better hardware is available.
* Multilingual summaries (Hindi/Hinglish) with mBART / mT5.
* Converting relative deadlines ("next Friday") to calendar dates and exporting to Google Calendar/Outlook (.ics).
* Email the minutes to all participants automatically.
* Search across past meetings stored in `outputs/`.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick **Add python.exe to PATH**, or use `py -3.11` instead of `python`. |
| `venv\Scripts\activate` does nothing in PowerShell / "scripts disabled" | Use `.\venv\Scripts\Activate.ps1` and run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt. |
| `pip install` fails building a package | You are probably on a very new Python (e.g. 3.14). Install **Python 3.11** and recreate the venv. |
| `OSError: [WinError 126] ... c10.dll` / `ctranslate2` DLL error | Install **Microsoft Visual C++ Redistributable (x64)**: https://aka.ms/vs/17/release/vc_redist.x64.exe, then restart. |
| `'streamlit' is not recognized` | The venv is not active → run `venv\Scripts\activate` first (you should see `(venv)` in the prompt). Or use `python -m streamlit run app.py`. |
| Model download stuck / fails | Check internet; college Wi-Fi may block huggingface.co → use mobile hotspot once, run `python download_models.py`. |
| "AI summarization model unavailable, used extractive mode" | Model not downloaded yet or not enough RAM. Run `python download_models.py` with internet, or keep using Fast mode. |
| Sidebar says spaCy model missing | `python -m spacy download en_core_web_sm`, then restart the app. |
| Transcription is slow | Use `tiny`, close other apps, keep the laptop plugged in (battery saver slows the CPU). |
| Transcript has wrong words | Use `small` model, record closer to the mic, choose the right language in the sidebar. |
| `MemoryError` / app freezes | Use `tiny` + Fast mode; shorten the audio; close Chrome tabs. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Upload bigger than 500 MB | Change `maxUploadSize` in `.streamlit/config.toml` and `MAX_UPLOAD_MB` in `config.py`. |
| Wrong owner for an action item | Rules depend on phrasing. In pasted transcripts use `Name: text` lines; audio transcripts have no speaker labels so "I will…" shows "Not specified". |
