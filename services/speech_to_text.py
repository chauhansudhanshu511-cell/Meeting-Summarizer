"""
speech_to_text.py
-----------------
Step 2 of the pipeline: Automatic Speech Recognition (ASR).

We use "faster-whisper", a fast re-implementation of OpenAI's Whisper model
built on the CTranslate2 engine. On a CPU it is about 4x faster than the
original Whisper and uses less memory, which makes it ideal for a laptop.

The model is downloaded automatically the FIRST time (internet needed once).
After that it is loaded from the local "models/whisper" folder and works offline.
"""

import config


class TranscriptionError(Exception):
    """Raised when speech-to-text fails."""


def load_whisper_model(model_size: str = config.DEFAULT_WHISPER_MODEL):
    """
    Load a faster-whisper model on the CPU.

    First we try to load it from the local "models" folder (offline).
    If it is not there yet, we download it from the internet (one time only).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "The 'faster-whisper' package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    common_args = dict(
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        download_root=config.WHISPER_MODELS_DIR,
    )

    # 1) Try offline (already downloaded)
    try:
        return WhisperModel(model_size, local_files_only=True, **common_args)
    except Exception:
        pass  # not downloaded yet -> try downloading below

    # 2) Download (needs internet the first time)
    try:
        return WhisperModel(model_size, **common_args)
    except MemoryError as exc:
        raise TranscriptionError(
            "Not enough RAM to load the Whisper model. Close other programs or choose the 'tiny' model."
        ) from exc
    except Exception as exc:
        raise TranscriptionError(
            f"Could not load/download the Whisper '{model_size}' model. "
            "An internet connection is needed the first time the model is used. "
            f"Details: {exc}"
        ) from exc


def transcribe_audio(model, wav_path: str, language=config.DEFAULT_LANGUAGE,
                     duration: float = 0.0, progress_callback=None) -> dict:
    """
    Convert speech in a WAV file into text.

    Args:
        model: a loaded WhisperModel
        wav_path: path to the 16 kHz WAV file
        language: "en", another language code, or None for auto-detect
        duration: audio length in seconds (used only for the progress bar)
        progress_callback: optional function(fraction_between_0_and_1)

    Returns:
        {
          "text": full transcript,
          "segments": [{"start": 0.0, "end": 3.2, "text": "..."}, ...],
          "language": detected language code
        }
    """
    try:
        segments_generator, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=1,            # greedy decoding = fastest on CPU
            vad_filter=True,        # skip silence using Voice Activity Detection
            condition_on_previous_text=False,  # reduces repeated-text loops
        )

        segments = []
        # faster-whisper returns a generator: transcription happens while we loop
        for segment in segments_generator:
            text = segment.text.strip()
            if text:
                segments.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                })
            if progress_callback and duration > 0:
                progress_callback(min(segment.end / duration, 1.0))

    except MemoryError as exc:
        raise TranscriptionError(
            "Ran out of memory during transcription. Try the 'tiny' model or a shorter file."
        ) from exc
    except RuntimeError as exc:
        if "memory" in str(exc).lower():
            raise TranscriptionError(
                "Ran out of memory during transcription. Try the 'tiny' model or a shorter file."
            ) from exc
        raise TranscriptionError(f"Transcription failed: {exc}") from exc
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    full_text = " ".join(seg["text"] for seg in segments).strip()
    if not full_text:
        raise TranscriptionError(
            "No speech could be detected in the audio. The file may be silent, too short, "
            "or the speech may be unclear."
        )

    return {"text": full_text, "segments": segments, "language": info.language}


def format_timestamp(seconds: float) -> str:
    """Convert 75.4 -> '01:15' for display."""
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def segments_to_timed_text(segments: list) -> str:
    """Build a readable transcript with [mm:ss] timestamps on every line."""
    return "\n".join(f"[{format_timestamp(s['start'])}] {s['text']}" for s in segments)
