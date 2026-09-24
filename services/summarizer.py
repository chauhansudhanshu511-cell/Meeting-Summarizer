"""
summarizer.py
-------------
Step 5 of the pipeline: generate the summary text.

Two methods are provided:

1. ABSTRACTIVE summarization (default)
   A pre-trained transformer model (DistilBART fine-tuned on the SAMSum
   dialogue dataset) READS the conversation and WRITES a new, shorter text
   in its own words - similar to how a human writes minutes.

2. EXTRACTIVE summarization (TextRank - no download needed)
   The TextRank algorithm (inspired by Google's PageRank) builds a graph of
   sentences, connects sentences that share words, and picks the most
   "central" sentences. It copies sentences exactly from the transcript.

The extractive method is also used as an automatic FALLBACK if the transformer
model cannot be loaded (no internet on first run, low RAM, etc.), so the app
always produces a result.
"""

import math
import re

import numpy as np

import config

# A small list of common English words that do not carry meaning by themselves.
STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further get got had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's
like me more most mustn't my myself need no nor not now of off on once only or other ought our ours ourselves
out over own okay ok same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's will with won't would wouldn't yeah yes you you'd you'll you're you've your
yours yourself yourselves really right think going go well actually basically sure thing things one also
""".split())


class SummarizationError(Exception):
    """Raised when the transformer summarizer fails."""


# ===========================================================================
# 1. EXTRACTIVE SUMMARIZATION - TextRank
# ===========================================================================
def _tokenize(sentence: str) -> list:
    words = re.findall(r"[a-zA-Z][a-zA-Z']+", sentence.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def textrank_scores(sentences: list, damping: float = 0.85, iterations: int = 50) -> np.ndarray:
    """
    Score every sentence with the TextRank algorithm.

    1. Similarity between two sentences = shared words / (log|S1| + log|S2|)
    2. Build an N x N similarity matrix (graph).
    3. Run PageRank on the graph: important sentences are the ones that are
       similar to many other important sentences.
    """
    n = len(sentences)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    token_sets = [set(_tokenize(s)) for s in sentences]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or not token_sets[i] or not token_sets[j]:
                continue
            overlap = len(token_sets[i] & token_sets[j])
            if overlap:
                denom = math.log(len(token_sets[i]) + 1) + math.log(len(token_sets[j]) + 1)
                matrix[i][j] = overlap / denom

    # normalise each row so it sums to 1 (probability of "jumping" to another sentence)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(iterations):  # PageRank power iteration
        scores = (1 - damping) / n + damping * matrix.T.dot(scores)

    # small bonus for informative (longer) sentences, penalty for tiny ones
    lengths = np.array([len(_tokenize(s)) for s in sentences])
    scores = scores * np.clip(lengths / 6.0, 0.2, 1.5)
    return scores


def extractive_summary(sentences: list, num_sentences: int = 5, min_words: int = 6) -> list:
    """Return the top-N sentences (kept in their original order)."""
    candidates = [(i, s) for i, s in enumerate(sentences) if len(s.split()) >= min_words]
    if not candidates:
        candidates = list(enumerate(sentences))
    if not candidates:
        return []

    scores = textrank_scores([s for _, s in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:num_sentences]
    ranked.sort(key=lambda x: x[1][0])  # back to original order
    return [s for _, (_, s) in ranked]


# ===========================================================================
# 2. ABSTRACTIVE SUMMARIZATION - Transformer (DistilBART-SAMSum)
# ===========================================================================
class AbstractiveSummarizer:
    """Wraps a Hugging Face sequence-to-sequence model (BART family)."""

    def __init__(self, model_name: str = config.SUMMARIZATION_MODEL):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise SummarizationError(
                "The 'transformers' and 'torch' packages are required for AI summarization. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.tokenizer, self.model = None, None

        # 1) try the local copy first (works offline), 2) otherwise download once
        last_error = None
        for local_only in (True, False):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=config.HF_MODELS_DIR, local_files_only=local_only)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, cache_dir=config.HF_MODELS_DIR, local_files_only=local_only)
                break
            except MemoryError as exc:
                raise SummarizationError(
                    "Not enough RAM to load the summarization model. Use 'Extractive' mode instead."
                ) from exc
            except Exception as exc:  # not downloaded yet / no internet
                last_error = exc

        if self.model is None:
            raise SummarizationError(
                f"Could not load the summarization model '{model_name}'. "
                "Internet is required the first time to download it (~1.2 GB). "
                f"Details: {last_error}"
            )

        self.model.eval()  # inference mode (no training)
        self.max_input_tokens = min(getattr(self.tokenizer, "model_max_length", 1024) or 1024, 1024)

    def summarize(self, text: str, max_tokens: int = config.SUMMARY_MAX_TOKENS,
                  min_tokens: int = config.SUMMARY_MIN_TOKENS) -> str:
        """Summarize ONE chunk of text (must fit into the model's 1024-token window)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=self.max_input_tokens)
        input_len = inputs["input_ids"].shape[1]
        # never ask for a summary longer than the input itself
        max_tokens = max(20, min(max_tokens, int(input_len * 0.8)))
        min_tokens = min(min_tokens, max_tokens // 2)

        try:
            with self.torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_new_tokens=min_tokens,
                    num_beams=4,               # beam search = better sentences
                    no_repeat_ngram_size=3,    # avoid repeating phrases
                    length_penalty=1.0,
                    early_stopping=True,
                )
        except (MemoryError, RuntimeError) as exc:
            raise SummarizationError(f"The model ran out of memory or failed: {exc}") from exc

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def chunk_utterances(lines: list, word_limit: int = config.CHUNK_WORD_LIMIT) -> list:
    """
    Split the meeting into chunks small enough for the transformer.
    We split between speaker lines (never in the middle of a sentence when possible).
    """
    chunks, current, count = [], [], 0
    for line in lines:
        words = line.split()
        # a single huge line (e.g. Whisper output without speakers): cut it by words
        while len(words) > word_limit:
            if current:
                chunks.append("\n".join(current))
                current, count = [], 0
            chunks.append(" ".join(words[:word_limit]))
            words = words[word_limit:]
        if count + len(words) > word_limit and current:
            chunks.append("\n".join(current))
            current, count = [], 0
        if words:
            current.append(" ".join(words))
            count += len(words)
    if current:
        chunks.append("\n".join(current))
    return chunks


def abstractive_summary(summarizer: AbstractiveSummarizer, cleaned_text: str,
                        progress_callback=None) -> str:
    """Summarize a (possibly long) meeting: summarize each chunk, then join the results."""
    chunks = chunk_utterances(cleaned_text.split("\n"))
    partial_summaries = []
    for index, chunk in enumerate(chunks):
        partial_summaries.append(summarizer.summarize(chunk))
        if progress_callback:
            progress_callback((index + 1) / len(chunks))

    combined = " ".join(s for s in partial_summaries if s)

    # For very long meetings the combined text can still be long -> summarize once more.
    if len(chunks) > 4 and len(combined.split()) > config.CHUNK_WORD_LIMIT:
        combined = abstractive_summary(summarizer, combined)
    return combined
