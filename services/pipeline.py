"""
pipeline.py
-----------
Connects all the steps together (this is the "brain" that app.py calls).

    Meeting Audio/Video
        -> video_processor.extract_audio()          (Step 1: processing file)
        -> speech_to_text.transcribe_audio()        (Step 2: transcribing)
        -> preprocessing.preprocess_transcript()    (Step 3: cleaning)
        -> information_extractor.extract_information()  (Step 4: extraction)
        -> summarizer (abstractive / extractive)     (Step 5: summary)
        -> build_result()                            (Step 6: final structured output)

Keeping the logic here (and not inside app.py) means the same pipeline can be
tested from the command line:   python -m services.pipeline sample_data/sample_transcript.txt
"""

import os
import time
from datetime import datetime

import config
from services.information_extractor import extract_information
from services.preprocessing import preprocess_transcript, split_sentences
from services.speech_to_text import TranscriptionError, segments_to_timed_text, transcribe_audio
from services.summarizer import (AbstractiveSummarizer, SummarizationError, abstractive_summary,
                                 extractive_summary)
from services.video_processor import AudioProcessingError, extract_audio

STEPS = [
    "Processing file",
    "Transcribing meeting",
    "Cleaning transcript",
    "Extracting important information",
    "Generating summary",
    "Preparing final result",
]


class PipelineError(Exception):
    """A friendly error message that can be shown directly to the user."""


def _noop(*_args, **_kwargs):
    return None


def _join_names(names: list) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# ---------------------------------------------------------------------------
# Audio / video  ->  transcript
# ---------------------------------------------------------------------------
def transcribe_media(file_path: str, whisper_model, language=config.DEFAULT_LANGUAGE,
                     on_step=_noop, on_progress=_noop, on_info=_noop) -> dict:
    """Steps 1 and 2: convert the file to WAV and run speech-to-text."""
    on_step(1, STEPS[0])
    try:
        wav_path, duration = extract_audio(file_path, config.TEMP_DIR)
    except AudioProcessingError as exc:
        raise PipelineError(str(exc)) from exc

    if duration < config.MIN_AUDIO_SECONDS:
        raise PipelineError(
            f"The audio is too short ({duration:.1f} s). Please upload at least "
            f"{config.MIN_AUDIO_SECONDS} seconds of speech."
        )
    if duration > config.MAX_AUDIO_MINUTES * 60:
        raise PipelineError(
            f"The audio is {duration / 60:.0f} minutes long. For a laptop demo the limit is "
            f"{config.MAX_AUDIO_MINUTES} minutes (change MAX_AUDIO_MINUTES in config.py)."
        )
    if duration > config.LONG_AUDIO_WARNING_MINUTES * 60:
        on_info(f"Long meeting detected ({duration / 60:.0f} min). Transcription on CPU may take "
                "several minutes - please keep this window open.")

    on_step(2, STEPS[1])
    try:
        result = transcribe_audio(whisper_model, wav_path, language=language,
                                  duration=duration, progress_callback=on_progress)
    except TranscriptionError as exc:
        raise PipelineError(str(exc)) from exc
    finally:
        try:
            os.remove(wav_path)  # delete temporary WAV file
        except OSError:
            pass

    result["duration"] = duration
    result["timed_text"] = segments_to_timed_text(result["segments"])
    return result


# ---------------------------------------------------------------------------
# Transcript  ->  structured summary
# ---------------------------------------------------------------------------
def summarize_transcript(raw_text: str, summary_mode: str = config.DEFAULT_SUMMARY_MODE,
                         summarizer_loader=None, source: str = "Pasted transcript",
                         media_info: dict = None, on_step=_noop, on_progress=_noop) -> dict:
    """
    Steps 3 to 6. Works on its own for the "paste transcript" mode.

    Args:
        raw_text: the meeting transcript
        summary_mode: "abstractive" or "extractive"
        summarizer_loader: function that returns an AbstractiveSummarizer (cached by the app)
        source: text describing where the transcript came from
        media_info: transcription details (duration, language, segments) if audio was used
    """
    start_time = time.time()
    warnings = []

    # ---- Step 3: preprocessing -------------------------------------------
    on_step(3, STEPS[2])
    try:
        pre = preprocess_transcript(raw_text)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc

    if pre["stats"]["cleaned_words"] < config.MIN_TRANSCRIPT_WORDS:
        raise PipelineError(
            f"The transcript is too short to summarize ({pre['stats']['cleaned_words']} words). "
            f"Please provide at least {config.MIN_TRANSCRIPT_WORDS} words."
        )

    # ---- Step 4: information extraction -----------------------------------
    on_step(4, STEPS[3])
    info = extract_information(pre)

    # ---- Step 5: summary generation ---------------------------------------
    on_step(5, STEPS[4])
    method_used = "Extractive (TextRank)"
    core_summary = ""
    if summary_mode == "abstractive":
        try:
            loader = summarizer_loader or (lambda: AbstractiveSummarizer(config.SUMMARIZATION_MODEL))
            summarizer = loader()
            core_summary = abstractive_summary(summarizer, pre["cleaned_text"], progress_callback=on_progress)
            method_used = f"Abstractive ({config.SUMMARIZATION_MODEL})"
        except SummarizationError as exc:
            warnings.append(f"AI summarization model unavailable, used extractive mode instead. {exc}")
        except Exception as exc:  # any unexpected model problem -> still give a result
            warnings.append(f"AI summarization failed, used extractive mode instead. ({exc})")

    if not core_summary:
        core_summary = " ".join(extractive_summary([s["text"] for s in pre["sentences"]], 3))

    # ---- Step 6: build final structured result ----------------------------
    on_step(6, STEPS[5])
    speakers = pre["speakers"]
    topics = info["topics"]

    # A short, fact-based opening sentence built only from detected data (nothing invented)
    topic_text = _join_names([t.lower() for t in topics[:3]])
    if speakers:
        intro = (f"This meeting involved {len(speakers)} participant{'s' if len(speakers) != 1 else ''} "
                 f"({_join_names(speakers)})")
        intro += f" and mainly discussed {topic_text}." if topics else "."
    elif topics:
        intro = f"This meeting mainly discussed {topic_text}."
    else:
        intro = ""
    overview = (intro + " " + core_summary).strip()

    summary_sentences = split_sentences(core_summary)[:3]
    counts = []
    if info["decisions"]:
        n = len(info["decisions"])
        counts.append(f"{n} decision{'s were' if n != 1 else ' was'} made")
    if info["action_items"]:
        n = len(info["action_items"])
        counts.append(f"{n} action item{'s were' if n != 1 else ' was'} assigned")
    if counts:
        summary_sentences.append(" and ".join(counts).capitalize() + ".")
    if len(summary_sentences) < 3 and info["key_points"]:
        extra = extractive_summary([s["text"] for s in pre["sentences"]], 2)
        summary_sentences.extend(e for e in extra if e not in summary_sentences)
    short_summary = " ".join(summary_sentences[:5])

    media_info = media_info or {}
    return {
        "meta": {
            "title": "Meeting Summary",
            "generated_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "source": source,
            "audio_duration_seconds": round(media_info.get("duration", 0), 1),
            "language": media_info.get("language", "en"),
            "summary_method": method_used,
            "processing_seconds": round(time.time() - start_time, 1),
        },
        "summary": {"overview": overview, "short_summary": short_summary},
        "main_points": info["main_points"],
        "key_points": info["key_points"],
        "decisions": info["decisions"],
        "action_items": info["action_items"],
        "topics": topics,
        "deadlines": info["deadlines"],
        "entities": info["entities"],
        "participants": speakers,
        "transcript": {
            "raw": raw_text.strip(),
            "timed": media_info.get("timed_text", ""),
            "cleaned": pre["cleaned_text"],
        },
        "stats": pre["stats"],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Command-line test:  python -m services.pipeline sample_data/sample_transcript.txt [extractive]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.SAMPLE_DATA_DIR, "sample_transcript.txt")
    mode = sys.argv[2] if len(sys.argv) > 2 else config.DEFAULT_SUMMARY_MODE
    with open(path, encoding="utf-8") as f:
        text = f.read()
    output = summarize_transcript(text, summary_mode=mode,
                                  on_step=lambda n, label: print(f"[{n}/6] {label}..."))
    output["transcript"] = "(omitted)"
    print(json.dumps(output, indent=2, ensure_ascii=False))
