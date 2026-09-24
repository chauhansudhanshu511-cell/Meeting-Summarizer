"""
download_models.py
------------------
Run this ONCE (with internet) before your demonstration:

    python download_models.py            -> downloads Whisper "base" + summarizer + spaCy
    python download_models.py tiny small -> downloads the Whisper sizes you list

After this, the application works completely offline.
"""

import sys

import config


def main():
    whisper_sizes = sys.argv[1:] or [config.DEFAULT_WHISPER_MODEL]

    # 1. Whisper speech-to-text models
    from services.speech_to_text import load_whisper_model
    for size in whisper_sizes:
        if size not in config.WHISPER_MODEL_SIZES:
            print(f"  Skipping unknown Whisper size '{size}'. Use: {config.WHISPER_MODEL_SIZES}")
            continue
        print(f"[1/3] Downloading Whisper '{size}' model...")
        try:
            load_whisper_model(size)
            print(f"      OK - saved in {config.WHISPER_MODELS_DIR}")
        except Exception as exc:
            print(f"      FAILED: {exc}")

    # 2. Summarization transformer
    print(f"[2/3] Downloading summarization model '{config.SUMMARIZATION_MODEL}' (~1.2 GB)...")
    try:
        from services.summarizer import AbstractiveSummarizer
        summarizer = AbstractiveSummarizer(config.SUMMARIZATION_MODEL)
        test = summarizer.summarize("Priya: Let's finish the report by Friday. "
                                    "Rahul: Sure, I will write the introduction today.")
        print(f"      OK - test summary: {test}")
    except Exception as exc:
        print(f"      FAILED: {exc}")

    # 3. spaCy English model
    print(f"[3/3] Checking spaCy model '{config.SPACY_MODEL}'...")
    try:
        import spacy
        try:
            spacy.load(config.SPACY_MODEL)
            print("      OK - already installed")
        except OSError:
            from spacy.cli import download
            download(config.SPACY_MODEL)
            print("      OK - installed (restart the app if it is running)")
    except Exception as exc:
        print(f"      FAILED: {exc}")

    print("\nDone. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
