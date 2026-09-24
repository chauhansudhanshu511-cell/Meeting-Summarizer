"""
preprocessing.py
----------------
Step 3 of the pipeline: clean the raw transcript before the AI reads it.

What we clean (carefully - we never delete real content like decisions or tasks):
    * transcription noise tags such as [Music], (inaudible), [BLANK_AUDIO]
    * filler words: um, uh, erm, hmm ...
    * stuttered word repetitions: "the the the plan" -> "the plan"
    * the same sentence repeated back-to-back (a common Whisper glitch)
    * extra spaces, spaces before punctuation, missing capital letters

We also detect speaker names when the transcript is written as
    "Priya: We should finish the report by Friday."
so that later steps can tell WHO said what.
"""

import re

# Pure filler sounds - safe to remove because they carry no meaning.
FILLER_PATTERN = re.compile(
    r"(?<![\w'])(?:u+m+|u+h+m*|e+r+m+|e+r+|a+h+|h+m+|m+h*m+|you know,|i mean,)(?![\w'])[,.]?\s*",
    flags=re.IGNORECASE,
)

# Things like [Music], (laughs), [inaudible], <noise>, ♪
NOISE_PATTERN = re.compile(
    r"\[[^\]]{0,40}\]|\((?:inaudible|laughs?|laughter|music|noise|applause|crosstalk|silence)[^)]*\)"
    r"|<[^>]{0,20}>|[♪♫]+",
    flags=re.IGNORECASE,
)

# "the the the" -> "the"  (same word repeated 2+ times)
REPEATED_WORD_PATTERN = re.compile(r"\b(\w+)(?:[\s,]+\1\b)+", flags=re.IGNORECASE)

# "Priya: some text"   or   "Dr. Mehta: some text"
SPEAKER_LINE_PATTERN = re.compile(r"^\s*([A-Z][A-Za-z.'\- ]{0,30}?)\s*:\s+(.+)$")

# Split after . ! ? when followed by a space and a capital letter / digit / quote
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")


# Abbreviations whose full stop does NOT end a sentence ("Dr. Mehta")
ABBREVIATIONS = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|approx)\.", flags=re.IGNORECASE)


def split_sentences(text: str) -> list:
    """Split a paragraph into sentences using punctuation rules."""
    text = text.strip()
    if not text:
        return []
    protected = ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = SENTENCE_SPLIT_PATTERN.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _clean_fragment(text: str) -> tuple:
    """Clean one line / utterance. Returns (cleaned_text, fillers_removed_count)."""
    text = NOISE_PATTERN.sub(" ", text)

    fillers_removed = len(FILLER_PATTERN.findall(text))
    text = FILLER_PATTERN.sub(" ", text)

    text = REPEATED_WORD_PATTERN.sub(r"\1", text)

    # whitespace and punctuation tidy-up
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)      # "word ," -> "word,"
    text = re.sub(r"([,.!?;:])\1+", r"\1", text)       # "!!" -> "!"
    text = re.sub(r"^[,.;:\s]+", "", text)              # leading punctuation
    text = re.sub(r",\s*([.!?])", r"\1", text)          # ", ." -> "."
    text = text.strip()

    # Capitalise the first letter of every sentence
    sentences = split_sentences(text)
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]

    # Remove a sentence that is exactly the same as the previous one
    deduped = []
    for sentence in sentences:
        if not deduped or sentence.lower() != deduped[-1].lower():
            deduped.append(sentence)

    return " ".join(deduped), fillers_removed


def parse_speakers(raw_text: str) -> list:
    """
    Turn the transcript into a list of utterances:
        [{"speaker": "Priya", "text": "..."}, {"speaker": None, "text": "..."}]
    Lines without a "Name:" prefix get speaker None. Lines that continue the
    previous speaker are merged into that speaker's utterance.
    """
    utterances = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = SPEAKER_LINE_PATTERN.match(line)
        if match and len(match.group(1).split()) <= 3:
            utterances.append({"speaker": match.group(1).strip(), "text": match.group(2).strip()})
        elif utterances:
            utterances[-1]["text"] += " " + line
        else:
            utterances.append({"speaker": None, "text": line})
    return utterances


def preprocess_transcript(raw_text: str) -> dict:
    """
    Main function of this module.

    Returns:
        {
          "cleaned_text": str        (speaker labels kept, one utterance per line),
          "plain_text": str          (no speaker labels, single paragraph - for the AI model),
          "utterances": list         [{"speaker", "text"}],
          "sentences": list          [{"speaker", "text"}],
          "speakers": list           unique speaker names,
          "stats": dict
        }
    """
    if raw_text is None or not raw_text.strip():
        raise ValueError("The transcript is empty. Please upload audio or paste some text.")

    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    original_words = len(raw_text.split())

    utterances = []
    total_fillers = 0
    for utt in parse_speakers(raw_text):
        cleaned, fillers = _clean_fragment(utt["text"])
        total_fillers += fillers
        if cleaned:
            utterances.append({"speaker": utt["speaker"], "text": cleaned})

    if not utterances:
        raise ValueError("After cleaning, the transcript has no meaningful text.")

    sentences = []
    for utt in utterances:
        for sentence in split_sentences(utt["text"]):
            sentences.append({"speaker": utt["speaker"], "text": sentence})

    cleaned_lines = [
        f"{u['speaker']}: {u['text']}" if u["speaker"] else u["text"] for u in utterances
    ]
    speakers = []
    for u in utterances:
        if u["speaker"] and u["speaker"] not in speakers:
            speakers.append(u["speaker"])

    cleaned_text = "\n".join(cleaned_lines)
    plain_text = " ".join(u["text"] for u in utterances)

    return {
        "cleaned_text": cleaned_text,
        "plain_text": plain_text,
        "utterances": utterances,
        "sentences": sentences,
        "speakers": speakers,
        "stats": {
            "original_words": original_words,
            "cleaned_words": len(plain_text.split()),
            "fillers_removed": total_fillers,
            "sentences": len(sentences),
            "speakers_detected": len(speakers),
        },
    }
