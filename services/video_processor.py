"""
video_processor.py
------------------
Step 1 of the pipeline: take ANY supported upload (MP3, WAV, M4A, MP4) and
convert it into a clean 16 kHz mono WAV file, which is the format Whisper
works best with.

How it works:
    * We use PyAV (the "av" Python package). PyAV already contains the FFmpeg
      libraries inside it, so on Windows you normally do NOT need to install
      FFmpeg separately.
    * If PyAV is missing for some reason, we fall back to the FFmpeg program
      (ffmpeg.exe) if it is installed on the system PATH.
    * For MP4 videos, only the audio track is extracted - the video frames are ignored.
"""

import os
import shutil
import subprocess
import wave

import numpy as np

TARGET_SAMPLE_RATE = 16000  # Whisper expects 16,000 samples per second


class AudioProcessingError(Exception):
    """Raised when the uploaded file cannot be read or converted."""


def _convert_with_pyav(input_path: str, output_path: str) -> float:
    """Decode the file with PyAV, resample to 16 kHz mono, save as WAV.
    Returns the audio duration in seconds."""
    import av  # imported here so the app still starts if PyAV is missing

    try:
        container = av.open(input_path)
    except Exception as exc:  # corrupt / not a media file
        raise AudioProcessingError(
            "The file could not be opened. It may be corrupted or not a real audio/video file."
        ) from exc

    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise AudioProcessingError(
                "No audio track was found in this file. Please upload a file that contains speech."
            )

        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        chunks = []

        for frame in container.decode(audio_streams[0]):
            for out_frame in resampler.resample(frame):
                chunks.append(out_frame.to_ndarray().reshape(-1))
        # flush any samples still inside the resampler
        for out_frame in resampler.resample(None):
            chunks.append(out_frame.to_ndarray().reshape(-1))
    except AudioProcessingError:
        raise
    except Exception as exc:
        raise AudioProcessingError(f"Error while decoding the audio: {exc}") from exc
    finally:
        container.close()

    if not chunks:
        raise AudioProcessingError("The audio track is empty.")

    samples = np.concatenate(chunks).astype(np.int16)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)          # mono
        wav_file.setsampwidth(2)          # 16-bit
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())

    return len(samples) / TARGET_SAMPLE_RATE


def _convert_with_ffmpeg(input_path: str, output_path: str) -> float:
    """Fallback: use the ffmpeg.exe program installed on the computer."""
    if shutil.which("ffmpeg") is None:
        raise AudioProcessingError(
            "Audio conversion is not available: the 'av' package is missing and FFmpeg is not installed. "
            "Run 'pip install av' or install FFmpeg (see README > Troubleshooting)."
        )
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",                      # drop video
        "-ac", "1",                 # mono
        "-ar", str(TARGET_SAMPLE_RATE),
        "-sample_fmt", "s16",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise AudioProcessingError(
            "FFmpeg could not convert this file. It may be corrupted or have no audio track."
        )
    return get_wav_duration(output_path)


def get_wav_duration(wav_path: str) -> float:
    """Return the length of a WAV file in seconds."""
    with wave.open(wav_path, "rb") as wav_file:
        return wav_file.getnframes() / float(wav_file.getframerate())


def extract_audio(input_path: str, output_dir: str) -> tuple:
    """
    Convert an uploaded audio/video file to a 16 kHz mono WAV file.

    Returns:
        (wav_path, duration_in_seconds)
    """
    if not os.path.exists(input_path):
        raise AudioProcessingError("Uploaded file not found on disk.")
    if os.path.getsize(input_path) == 0:
        raise AudioProcessingError("The uploaded file is empty (0 bytes).")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_16k.wav")

    try:
        import av  # noqa: F401  (just checking that PyAV is installed)
        duration = _convert_with_pyav(input_path, output_path)
    except ImportError:
        duration = _convert_with_ffmpeg(input_path, output_path)

    return output_path, duration


def is_ffmpeg_available() -> bool:
    """True if either PyAV (bundled FFmpeg) or the ffmpeg program is available."""
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return shutil.which("ffmpeg") is not None
