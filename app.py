"""
app.py
------
AI-Based Smart Meeting Summarization - Streamlit web interface.

Run with:
    streamlit run app.py
Then open http://localhost:8501 in your browser.
"""

import html
import os

import pandas as pd
import streamlit as st

import config
from services.information_extractor import get_nlp
from services.pipeline import STEPS, PipelineError, summarize_transcript, transcribe_media
from services.speech_to_text import TranscriptionError, load_whisper_model
from services.summarizer import AbstractiveSummarizer
from services.video_processor import is_ffmpeg_available
from utils.file_utils import (InvalidFileError, cleanup_temp_folder, delete_file, is_video,
                              read_text_file, save_uploaded_file)
from utils.output_utils import result_to_json, result_to_pdf, result_to_text, save_outputs

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Smart Meeting Summarizer", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1F4E79 0%, #2E75B6 100%);
        padding: 1.4rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.2rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #E3F0FF; margin: 0.4rem 0 0 0; font-size: 1.05rem; }
    .card {
        background: #F7FAFD; border: 1px solid #D6E4F0; border-left: 5px solid #2E75B6;
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; color: #1b1b1b;
    }
    .card h4 { margin: 0 0 0.4rem 0; color: #1F4E79; }
    .pill {
        display: inline-block; background: #E3F0FF; color: #1F4E79; border-radius: 20px;
        padding: 0.25rem 0.8rem; margin: 0.2rem; font-size: 0.9rem; border: 1px solid #BFD7F0;
    }
    .kp-group { font-weight: 600; color: #1F4E79; margin: 0.6rem 0 0.2rem 0; }
    .card ul { margin: 0.2rem 0 0.2rem 1.1rem; padding: 0; }
    .card li { margin-bottom: 0.25rem; }
    .footer { text-align: center; color: #888; font-size: 0.85rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached model loaders (models are loaded only ONCE, then reused)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_whisper(model_size: str):
    return load_whisper_model(model_size)


@st.cache_resource(show_spinner=False)
def get_summarizer(model_name: str):
    return AbstractiveSummarizer(model_name)


@st.cache_resource(show_spinner=False)
def startup_tasks():
    cleanup_temp_folder()
    return get_nlp() is not None  # loads spaCy once


# ---------------------------------------------------------------------------
# Session state (remembers values between button clicks)
# ---------------------------------------------------------------------------
defaults = {"result": None, "transcript_text": "", "uploader_key": 0, "saved_paths": None}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def load_sample():
    path = os.path.join(config.SAMPLE_DATA_DIR, "sample_transcript.txt")
    with open(path, encoding="utf-8") as f:
        st.session_state.transcript_text = f.read()
    st.session_state.input_mode = "📋 Paste Transcript"


def clear_all():
    st.session_state.result = None
    st.session_state.saved_paths = None
    st.session_state.transcript_text = ""
    st.session_state.uploader_key += 1  # new key = empty file uploader


def main_points_html(points: list, title: str = "2. Main Key Points") -> str:
    """Build the 'Main Key Points' card: discussion points, decisions and tasks, grouped."""
    groups = [("Discussion", "💬 Key discussion points"), ("Decision", "✅ Decisions"),
              ("Action", "📌 Action items")]
    body = ""
    for kind, label in groups:
        items = [p["text"] for p in points if p["type"] == kind]
        if items:
            body += f"<div class='kp-group'>{label}</div><ul>"
            body += "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"
    if not body:
        body = "No key points identified."
    heading = f"<h4>{title}</h4>" if title else ""
    return f"<div class='card'>{heading}{body}</div>"


spacy_ok = startup_tasks()

# ---------------------------------------------------------------------------
# Sidebar - settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    whisper_size = st.selectbox(
        "Speech-to-Text model (Whisper)",
        config.WHISPER_MODEL_SIZES,
        index=config.WHISPER_MODEL_SIZES.index(config.DEFAULT_WHISPER_MODEL),
        help="tiny = fastest, base = recommended, small = most accurate but slower on CPU.",
    )
    language_label = st.selectbox("Meeting language", ["English", "Auto-detect", "Hindi"], index=0)
    language = {"English": "en", "Auto-detect": None, "Hindi": "hi"}[language_label]

    summary_label = st.radio(
        "Summarization method",
        ["AI (Abstractive - Transformer)", "Fast (Extractive - TextRank)"],
        help="Abstractive writes a new summary using DistilBART. Extractive picks the most "
             "important sentences and needs no model download.",
    )
    summary_mode = "abstractive" if summary_label.startswith("AI") else "extractive"

    st.divider()
    st.subheader("🖥️ System status")
    st.write("✅ Running on **CPU**" if config.WHISPER_DEVICE == "cpu" else "⚡ Running on GPU")
    st.write("✅ Audio/video decoder ready" if is_ffmpeg_available()
             else "❌ Audio decoder missing (pip install av)")
    st.write("✅ spaCy NLP model loaded" if spacy_ok
             else "⚠️ spaCy model missing - basic mode (see README)")
    st.caption(f"Summarizer: `{config.SUMMARIZATION_MODEL}`")

    st.divider()
    st.button("📄 Load sample transcript", on_click=load_sample, width="stretch")
    st.caption("Tip: the first run downloads the AI models (internet needed once). "
               "After that the app works offline.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>📝 AI-Based Smart Meeting Summarization</h1>
        <p>Convert meeting audio or video into concise summaries, key points, decisions and action items using AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.subheader("1️⃣ Input")
input_mode = st.radio("Choose input type", ["🎧 Upload Audio/Video", "📋 Paste Transcript"],
                      horizontal=True, key="input_mode", label_visibility="collapsed")

uploaded_file = None
if input_mode == "🎧 Upload Audio/Video":
    uploaded_file = st.file_uploader(
        "Upload a meeting recording (MP3, WAV, M4A or MP4)",
        type=config.SUPPORTED_FORMATS,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"📁 **{uploaded_file.name}** — {size_mb:.2f} MB")
        if is_video(uploaded_file.name):
            st.video(uploaded_file)
        else:
            st.audio(uploaded_file)
else:
    txt_file = st.file_uploader("Optional: upload a .txt transcript", type=["txt"],
                                key=f"txt_uploader_{st.session_state.uploader_key}")
    if txt_file is not None and not st.session_state.transcript_text:
        try:
            st.session_state.transcript_text = read_text_file(txt_file)
        except InvalidFileError as err:
            st.error(str(err))
    st.text_area(
        "Paste or type the meeting transcript (use 'Name: text' lines to identify speakers)",
        key="transcript_text",
        height=260,
        placeholder="Priya: Let's start the meeting...\nRahul: I will finish the report by Friday...",
    )
    st.caption(f"Words: {len(st.session_state.transcript_text.split())}")

col_generate, col_clear, _ = st.columns([1.3, 1, 4])
generate_clicked = col_generate.button("🚀 Generate Summary", type="primary", width="stretch")
col_clear.button("🧹 Clear", on_click=clear_all, width="stretch")

# ---------------------------------------------------------------------------
# Processing section
# ---------------------------------------------------------------------------
if generate_clicked:
    st.session_state.result = None
    st.subheader("2️⃣ Processing")
    progress_bar = st.progress(0, text="Starting...")
    saved_path = None

    with st.status("Processing meeting...", expanded=True) as status:

        def on_step(number, label):
            progress_bar.progress((number - 1) / len(STEPS), text=f"Step {number}/6: {label}")
            status.update(label=f"Step {number}/6: {label}...")
            st.write(f"⏳ **Step {number}:** {label}")

        def on_progress(fraction):
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=f"Working... {fraction * 100:.0f}%")

        try:
            if input_mode == "🎧 Upload Audio/Video":
                if uploaded_file is None:
                    raise PipelineError("Please upload an audio/video file first (or switch to 'Paste Transcript').")
                saved_path = save_uploaded_file(uploaded_file)

                st.write(f"📥 Loading Whisper **{whisper_size}** model "
                         "(first time: downloading, please wait)...")
                try:
                    whisper_model = get_whisper(whisper_size)
                except TranscriptionError as err:
                    raise PipelineError(str(err)) from err

                media = transcribe_media(saved_path, whisper_model, language=language,
                                         on_step=on_step, on_progress=on_progress, on_info=st.warning)
                st.write(f"✅ Transcribed {media['duration']:.0f} seconds of audio "
                         f"(language: {media['language']}).")
                raw_text = media["text"]
                source = f"{'Video' if is_video(uploaded_file.name) else 'Audio'} file: {uploaded_file.name}"
            else:
                raw_text = st.session_state.transcript_text
                if not raw_text.strip():
                    raise PipelineError("The transcript box is empty. Paste a transcript or load the sample.")
                media, source = None, "Pasted transcript"
                on_step(1, STEPS[0])
                st.write("ℹ️ Transcript mode: audio steps skipped.")
                on_step(2, "Transcription skipped (text input)")

            if summary_mode == "abstractive":
                st.write("📥 Loading summarization model (first time: ~1.2 GB download)...")

            result = summarize_transcript(
                raw_text,
                summary_mode=summary_mode,
                summarizer_loader=lambda: get_summarizer(config.SUMMARIZATION_MODEL),
                source=source,
                media_info=media,
                on_step=on_step,
                on_progress=on_progress,
            )
            st.session_state.result = result
            st.session_state.saved_paths = save_outputs(result)

            progress_bar.progress(1.0, text="Done!")
            status.update(label="✅ Summary generated successfully!", state="complete", expanded=False)

        except (PipelineError, InvalidFileError) as err:
            status.update(label="❌ Processing failed", state="error", expanded=True)
            st.error(str(err))
        except MemoryError:
            status.update(label="❌ Out of memory", state="error", expanded=True)
            st.error("The computer ran out of memory. Close other programs, choose the 'tiny' Whisper "
                     "model, or use the 'Fast (Extractive)' summarization method.")
        except Exception as err:  # last safety net - never show a raw crash to the user
            status.update(label="❌ Unexpected error", state="error", expanded=True)
            st.error(f"Something went wrong: {err}")
            with st.expander("Technical details"):
                st.exception(err)
        finally:
            delete_file(saved_path)

# ---------------------------------------------------------------------------
# Output section
# ---------------------------------------------------------------------------
result = st.session_state.result
if result:
    st.subheader("3️⃣ Results")
    for warning in result["warnings"]:
        st.warning(warning)

    meta, stats = result["meta"], result["stats"]
    m1, m2, m3, m4, m5 = st.columns(5)
    duration = meta["audio_duration_seconds"]
    m1.metric("Audio length", f"{int(duration // 60)}m {int(duration % 60)}s" if duration else "Text input")
    m2.metric("Words (cleaned)", stats["cleaned_words"])
    m3.metric("Speakers found", stats["speakers_detected"] or "-")
    m4.metric("Decisions", len(result["decisions"]))
    m5.metric("Action items", len(result["action_items"]))
    st.caption(f"Method: {meta['summary_method']} • Generated: {meta['generated_at']} • "
               f"Fillers removed: {stats['fillers_removed']}")

    tabs = st.tabs(["📄 Summary", "📌 Key Points", "✅ Decisions", "🗂️ Action Items",
                    "🏷️ Topics & Entities", "📅 Deadlines", "🗒️ Transcript"])

    with tabs[0]:
        st.markdown(f"<div class='card'><h4>1. Overview</h4>{html.escape(result['summary']['overview'])}</div>",
                    unsafe_allow_html=True)
        st.markdown(main_points_html(result.get("main_points", [])), unsafe_allow_html=True)
        st.markdown(f"<div class='card'><h4>6. Short Summary</h4>{result['summary']['short_summary']}</div>",
                    unsafe_allow_html=True)
        if result["participants"]:
            st.markdown("**Participants:** " + "".join(f"<span class='pill'>👤 {p}</span>"
                                                       for p in result["participants"]),
                        unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("#### 2. Main Key Points")
        st.markdown(main_points_html(result.get("main_points", []), title=""), unsafe_allow_html=True)
        st.markdown("#### Important Discussion Points (with speaker)")
        for point in result["key_points"] or ["No key points identified."]:
            st.markdown(f"- {point}")

    with tabs[2]:
        st.markdown("#### 3. Decisions Made")
        if result["decisions"]:
            for d in result["decisions"]:
                st.success(f"**{d['decision']}**  \n_Said by: {d['said_by']}_")
        else:
            st.info("No explicit decisions were detected in this meeting.")

    with tabs[3]:
        st.markdown("#### 4. Action Items")
        if result["action_items"]:
            df = pd.DataFrame(result["action_items"])[["task", "responsible", "deadline"]]
            df.columns = ["Task", "Responsible Person", "Deadline"]
            df.index = range(1, len(df) + 1)
            st.table(df)
            st.caption("'Not specified' means the person or deadline was not mentioned in the meeting.")
        else:
            st.info("No action items were detected.")

    with tabs[4]:
        st.markdown("#### 5. Important Topics")
        st.markdown("".join(f"<span class='pill'>{t}</span>" for t in result["topics"]) or "None identified.",
                    unsafe_allow_html=True)
        st.markdown("#### Important Names / Entities")
        cols = st.columns(len(result["entities"]))
        for col, (group, values) in zip(cols, result["entities"].items()):
            col.markdown(f"**{group}**")
            col.write(", ".join(values) if values else "—")

    with tabs[5]:
        st.markdown("#### Deadlines & Dates Mentioned")
        if result["deadlines"]:
            df = pd.DataFrame(result["deadlines"])[["deadline", "said_by", "context"]]
            df.columns = ["Deadline / Date", "Said by", "Context"]
            df.index = range(1, len(df) + 1)
            st.table(df)
        else:
            st.info("No deadlines or dates were mentioned.")

    with tabs[6]:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Original Transcript")
            st.text_area("original", result["transcript"]["timed"] or result["transcript"]["raw"],
                         height=350, label_visibility="collapsed", disabled=True)
        with right:
            st.markdown("#### Cleaned Transcript")
            st.text_area("cleaned", result["transcript"]["cleaned"], height=350,
                         label_visibility="collapsed", disabled=True)

    # ---- Downloads -------------------------------------------------------
    st.subheader("4️⃣ Download")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Download TXT", result_to_text(result), file_name="meeting_summary.txt",
                       mime="text/plain", width="stretch")
    d2.download_button("⬇️ Download JSON", result_to_json(result), file_name="meeting_summary.json",
                       mime="application/json", width="stretch")
    try:
        d3.download_button("⬇️ Download PDF", result_to_pdf(result), file_name="meeting_summary.pdf",
                           mime="application/pdf", width="stretch")
    except Exception as err:
        d3.warning(f"PDF not available: {err}")

    if st.session_state.saved_paths:
        st.caption("Also saved automatically in the **outputs** folder: "
                   + ", ".join(os.path.basename(p) for p in st.session_state.saved_paths.values()))

st.markdown("<div class='footer'>AI-Based Smart Meeting Summarization • B.Tech Mini Project • "
            "Runs 100% locally with open-source models</div>", unsafe_allow_html=True)
