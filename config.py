"""
config.py
---------
Central configuration for the AI-Based Smart Meeting Summarization project.
Change the values here to switch models, speed up the app, or change folders.
"""

import os

# Hide harmless Hugging Face warnings on Windows and keep the tokenizer quiet
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Folder paths (all relative to this project folder)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")            # downloaded AI models are stored here
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "whisper")  # speech-to-text models
HF_MODELS_DIR = os.path.join(MODELS_DIR, "huggingface")   # summarization models
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")          # saved results (TXT / JSON / PDF)
TEMP_DIR = os.path.join(BASE_DIR, "outputs", "temp")     # temporary uploaded/converted files
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")

for _folder in (MODELS_DIR, WHISPER_MODELS_DIR, HF_MODELS_DIR, OUTPUTS_DIR, TEMP_DIR):
    os.makedirs(_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Speech-to-Text (faster-whisper) settings
# ---------------------------------------------------------------------------
# Available sizes (smaller = faster, bigger = more accurate):
#   "tiny"  (~75 MB)  -> fastest, good enough for clear English audio
#   "base"  (~145 MB) -> RECOMMENDED for a normal laptop demo
#   "small" (~480 MB) -> more accurate, slower on CPU
WHISPER_MODEL_SIZES = ["tiny", "base", "small"]
DEFAULT_WHISPER_MODEL = "base"

# "cpu" works on every laptop. "cuda" only if you have an NVIDIA GPU + CUDA.
WHISPER_DEVICE = "cpu"
# "int8" is the fastest and uses the least RAM on CPU.
WHISPER_COMPUTE_TYPE = "int8"

# Language of the meeting. None = auto-detect. Use "en" for English.
DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Summarization settings
# ---------------------------------------------------------------------------
# "abstractive" -> uses a transformer model (writes new sentences, needs model download)
# "extractive"  -> picks the most important sentences (no download, very fast)
DEFAULT_SUMMARY_MODE = "abstractive"

# DistilBART fine-tuned on SAMSum (a dataset of chat/dialogue summaries).
# Chosen because meetings are dialogues and this model is ~40% smaller/faster than BART-large.
SUMMARIZATION_MODEL = "philschmid/distilbart-cnn-12-6-samsum"

# Other models you can try (just replace the name above):
#   "philschmid/bart-large-cnn-samsum"  -> better quality, slower, ~1.6 GB
#   "sshleifer/distilbart-cnn-12-6"      -> news-style summaries, ~1.2 GB
#   "facebook/bart-large-cnn"            -> news-style summaries, ~1.6 GB

# The transformer can only read ~1024 tokens at once, so long transcripts
# are split into chunks of this many words and summarized piece by piece.
CHUNK_WORD_LIMIT = 550
SUMMARY_MAX_TOKENS = 160
SUMMARY_MIN_TOKENS = 40

# Number of sentences for extractive key points / short summary
NUM_KEY_POINTS = 6                 # detailed key points (Key Points tab)
NUM_MAIN_DISCUSSION_POINTS = 4     # discussion points shown in 'Main Key Points' with the summary
NUM_SHORT_SUMMARY_SENTENCES = 4
NUM_TOPICS = 8

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "m4a"]
SUPPORTED_VIDEO_FORMATS = ["mp4"]
SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS + SUPPORTED_VIDEO_FORMATS

MAX_UPLOAD_MB = 500                  # also set in .streamlit/config.toml
MIN_AUDIO_SECONDS = 2                # shorter audio is rejected
LONG_AUDIO_WARNING_MINUTES = 30      # show a "this will take time" warning
MAX_AUDIO_MINUTES = 120              # hard limit for a laptop demo
MIN_TRANSCRIPT_WORDS = 15            # shorter text is too small to summarize

# ---------------------------------------------------------------------------
# Spacy model for names / dates / organisations
# ---------------------------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"

NOT_SPECIFIED = "Not specified"
