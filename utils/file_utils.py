"""
file_utils.py
-------------
Helper functions for handling uploaded files safely.
"""

import os
import re
import time

import config


class InvalidFileError(Exception):
    """Raised for unsupported, empty or too-large uploads."""


def get_extension(filename: str) -> str:
    """'Meeting.MP3' -> 'mp3'"""
    return os.path.splitext(filename)[1].lower().lstrip(".")


def is_video(filename: str) -> bool:
    return get_extension(filename) in config.SUPPORTED_VIDEO_FORMATS


def safe_filename(filename: str) -> str:
    """Remove characters that are not allowed / risky in Windows file names."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(filename))
    return name[-80:] or "upload"


def validate_upload(filename: str, size_bytes: int) -> None:
    """Check extension and size before we try to process the file."""
    ext = get_extension(filename)
    if ext not in config.SUPPORTED_FORMATS:
        raise InvalidFileError(
            f"Unsupported file format '.{ext}'. Please upload one of: "
            + ", ".join(f".{e}" for e in config.SUPPORTED_FORMATS)
        )
    if size_bytes == 0:
        raise InvalidFileError("The uploaded file is empty.")
    if size_bytes > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidFileError(f"The file is larger than {config.MAX_UPLOAD_MB} MB.")


def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile object into the temp folder.
    Returns the full path of the saved file.
    """
    data = uploaded_file.getvalue()
    validate_upload(uploaded_file.name, len(data))
    path = os.path.join(config.TEMP_DIR, f"{int(time.time())}_{safe_filename(uploaded_file.name)}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def delete_file(path: str) -> None:
    """Delete a file, ignoring errors (e.g. file already removed)."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def cleanup_temp_folder(max_age_hours: int = 6) -> None:
    """Remove old temporary files so the project folder does not keep growing."""
    now = time.time()
    for name in os.listdir(config.TEMP_DIR):
        path = os.path.join(config.TEMP_DIR, name)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_hours * 3600:
            delete_file(path)


def read_text_file(uploaded_file) -> str:
    """Read an uploaded .txt transcript (tries UTF-8 first, then Windows encoding)."""
    data = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InvalidFileError("Could not read the text file. Please save it as UTF-8.")
