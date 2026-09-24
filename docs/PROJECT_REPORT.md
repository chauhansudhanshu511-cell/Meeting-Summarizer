# AI-Based Smart Meeting Summarization
### B.Tech Mini Project — Academic Documentation

---

## 1. Abstract

Meetings, lectures and online conferences generate a large amount of spoken information, yet the important outcomes — decisions, tasks, responsibilities and deadlines — are often lost because manual note-taking is slow, incomplete and depends on one person's attention. This project presents **AI-Based Smart Meeting Summarization**, a locally executed system that automatically converts meeting audio or video into a structured summary. The system accepts MP3, WAV, M4A and MP4 files, extracts and normalises the audio track, and transcribes speech using **faster-whisper**, an optimised implementation of OpenAI's Whisper Automatic Speech Recognition (ASR) model that runs efficiently on a CPU. The transcript is cleaned by a rule-based preprocessing module that removes filler words, transcription noise and repetitions while preserving meaningful content. A hybrid Natural Language Processing (NLP) stage then identifies topics, key points, decisions, action items, responsible persons, deadlines and named entities using spaCy Named Entity Recognition, pattern-based rules and the graph-based TextRank algorithm. Finally, a transformer model, **DistilBART fine-tuned on the SAMSum dialogue dataset**, generates an abstractive overview of the meeting, with TextRank as an offline fallback. Results are presented in a Streamlit web interface and can be exported as TXT, JSON and PDF. The system uses only free, open-source models, requires no GPU or paid API, and works offline after the initial model download, making it a practical and privacy-friendly tool for students, teams and organisations.

**Keywords:** Automatic Speech Recognition, Whisper, Natural Language Processing, Text Summarization, Transformer, BART, TextRank, Information Extraction, Named Entity Recognition.

---

## 2. Problem Statement

In colleges and workplaces, a large number of meetings are conducted every day — project reviews, classroom lectures, interviews and online conferences. After the meeting, participants usually depend on handwritten or typed notes prepared by one person. These notes are frequently incomplete, delayed, biased towards what the note-taker considered important, and often miss **who** is responsible for **what** and **by when**. Listening to a full recording again to recover this information is time-consuming.

Existing automatic solutions (for example, AI features in commercial meeting platforms) are usually paid, require uploading private recordings to cloud servers, and need a constant internet connection.

**The problem:** *to design a low-cost system that automatically converts a meeting recording or transcript into an accurate, structured summary — including key points, decisions, action items, responsible persons and deadlines — that runs locally on an ordinary laptop without a GPU, paid APIs or permanent internet access.*

---

## 3. Objectives

1. To accept meeting recordings in common audio/video formats (MP3, WAV, M4A, MP4) and automatically extract the audio from video files.
2. To convert speech into text accurately using an open-source Automatic Speech Recognition model (Whisper) that runs on a CPU.
3. To preprocess the transcript by removing filler words, noise tags and repetitions without losing important information.
4. To automatically identify key discussion points, decisions, action items, responsible persons, deadlines, topics and named entities.
5. To generate a concise, human-readable summary of the meeting using a transformer-based abstractive summarization model, with an extractive fallback.
6. To present the results in a simple, professional web interface and allow export in TXT, JSON and PDF formats.
7. To ensure the complete system runs locally, uses only free/open-source tools and works offline after the first setup.

---

## 4. Existing System

In the traditional approach, meeting minutes are prepared **manually**:

* One member (secretary, team lead or student) listens and writes notes during the meeting, or
* The meeting is recorded and someone later listens to the entire recording and types a summary.

**Limitations of the existing system:**

| Limitation | Explanation |
|---|---|
| Time-consuming | Writing or re-listening to a one-hour meeting can take one to two hours. |
| Incomplete notes | The note-taker cannot write everything while also participating. |
| Human bias and errors | Important points may be missed or recorded incorrectly. |
| Missing accountability | Action items are often recorded without a clear owner or deadline. |
| Delayed distribution | Minutes are shared hours or days later. |
| Not searchable | Handwritten notes cannot easily be searched or reused. |
| Commercial tools are costly | Cloud AI assistants need subscriptions, internet and uploading private data. |

---

## 5. Proposed System

The proposed system automates the complete process from recording to structured minutes:

* The user uploads an audio/video file **or** pastes a transcript into a local web application.
* The system extracts audio, transcribes it with Whisper, cleans the text, extracts important information and generates a summary.
* The output is displayed in separate sections — **Overview, Key Points, Decisions, Action Items (Task | Responsible Person | Deadline), Important Topics, Deadlines, Entities, Short Summary** — and can be downloaded.

**Key features / advantages over the existing system:**

* Fully automatic and fast (a 3-minute meeting is processed in about a minute on a normal laptop).
* Structured output with clear accountability (who does what by when).
* Does not invent information: if a person or deadline is not mentioned, it shows **"Not specified"**.
* Runs locally → recordings never leave the computer (privacy).
* No GPU, no paid API, offline after first model download.
* Two summarization modes: AI (abstractive) for quality, TextRank (extractive) for speed/low-resource laptops.

---

## 6. Methodology

```
Meeting Audio / Video
        │
        ▼
[1] Audio Extraction & Conversion ── PyAV (FFmpeg) → 16 kHz, mono, 16-bit WAV
        │
        ▼
[2] Speech Recognition (ASR) ────── faster-whisper (Whisper base, int8, CPU) + VAD
        │
        ▼
[3] Text Preprocessing ──────────── noise tags, fillers, stutters, repeats, spacing, speaker labels
        │
        ▼
[4] NLP / Information Extraction ── spaCy NER + noun chunks, rule patterns, TextRank
        │
        ▼
[5] Summarization ───────────────── DistilBART-SAMSum (abstractive) / TextRank (extractive)
        │
        ▼
[6] Structured Output ───────────── Overview, Key Points, Decisions, Action Items,
                                     Topics, Deadlines, Short Summary → TXT / JSON / PDF
```

**Step 1 – Audio extraction.** Uploaded files are validated (type, size, empty). The PyAV library decodes the media and, for MP4, selects only the audio stream. The signal is resampled to 16,000 Hz mono because Whisper was trained on 16 kHz audio.

**Step 2 – Speech recognition.** Whisper is a transformer encoder–decoder trained on 680,000 hours of multilingual speech. The audio is converted into a log-Mel spectrogram, the encoder learns acoustic features, and the decoder generates text tokens. faster-whisper runs the same model through the CTranslate2 engine with 8-bit (int8) quantisation, making it about four times faster on a CPU. Voice Activity Detection skips silent regions. The output is a list of timestamped segments.

**Step 3 – Preprocessing.** Regular expressions remove noise tags such as `[Music]` and `(inaudible)`, filler words (um, uh, hmm), stuttered repetitions ("the the"), duplicated consecutive sentences (a known Whisper artefact), and fix spacing/capitalisation. Lines written as `Name: text` are parsed so that each sentence is linked to its speaker. The cleaning is deliberately conservative so that decisions and tasks are not deleted.

**Step 4 – Information extraction.**
* *Named entities* — spaCy's statistical NER model labels PERSON, ORG, DATE, TIME, GPE etc.
* *Topics* — noun phrases (noun chunks) are lemmatised, generic words are filtered and the most frequent phrases are selected.
* *Key points* — TextRank builds a graph where sentences are nodes and word overlap is the edge weight, then runs PageRank; the highest-ranked sentences are the key points.
* *Decisions* — sentences containing decision expressions ("we decided", "let's go with", "agreed", "finalised").
* *Action items* — commitment/request patterns: "I will…" (owner = speaker), "Rahul will…", "Sneha, can you…", "assign X to Y", "we need to…" (owner = team). If a request is followed by "Sure/Okay/I'll…" from another speaker, that speaker becomes the owner. Duplicate tasks are merged using word-overlap similarity.
* *Deadlines* — patterns for weekdays, dates ("25th March", "25/03"), relative expressions ("tomorrow", "within five days", "by next Monday", "one week before…").

**Step 5 – Summarization.** The cleaned dialogue is split into chunks that fit the model's 1,024-token limit. DistilBART-SAMSum (a distilled BART sequence-to-sequence transformer fine-tuned on messenger-style dialogues) generates a summary for each chunk using beam search (4 beams) with repeated-trigram blocking; chunk summaries are concatenated (and summarised again for very long meetings). If the model is unavailable, TextRank produces an extractive summary.

**Step 6 – Structured output.** A fact-based opening sentence (participants and main topics) is combined with the model's summary to form the Overview. A Short Summary (3–5 sentences) adds counts of decisions and action items. All results are stored in a Python dictionary and exported as TXT, JSON and PDF.

---

## 7. Technologies Used

| Technology | Purpose in this project |
|---|---|
| **Python 3.11** | Main programming language; rich AI/NLP libraries. |
| **Streamlit** | Builds the local web interface (upload, buttons, progress, tabs, downloads) using only Python. |
| **PyAV (FFmpeg)** | Decodes MP3/M4A/WAV/MP4, extracts audio from video, resamples to 16 kHz. Bundled FFmpeg → no separate install. |
| **faster-whisper / OpenAI Whisper** | Open-source Automatic Speech Recognition; converts speech to text with timestamps. |
| **CTranslate2** | Fast inference engine used by faster-whisper; int8 quantisation for CPU. |
| **Hugging Face Transformers** | Loads and runs the pre-trained DistilBART summarization model. |
| **PyTorch** | Deep learning framework that executes the transformer on the CPU. |
| **DistilBART-CNN-12-6-SAMSum** | Transformer model that generates abstractive summaries of dialogues. |
| **spaCy (en_core_web_sm)** | NLP pipeline: tokenisation, part-of-speech tags, noun chunks, Named Entity Recognition. |
| **Regular expressions (re)** | Rule-based detection of fillers, decisions, action items and deadlines. |
| **NumPy** | Matrix operations for the TextRank/PageRank algorithm. |
| **Pandas** | Tables for action items and deadlines in the UI. |
| **fpdf2** | Generates the downloadable PDF report. |

---

## 8. System Requirements

**Hardware**

| Component | Minimum | Recommended |
|---|---|---|
| Processor | Dual/quad-core 64-bit (Intel i3 / Ryzen 3) | Intel i5 / Ryzen 5 or better |
| RAM | 4 GB (tiny model + extractive mode) | 8 GB or more |
| Storage | 4 GB free | 6 GB free |
| GPU | Not required | Optional NVIDIA GPU |
| Other | Microphone/recorder for creating meeting recordings | — |

**Software**

| Component | Version |
|---|---|
| Operating system | Windows 10/11 (also works on Linux/macOS) |
| Python | 3.10 – 3.12 (3.11 recommended) |
| Python libraries | streamlit, faster-whisper, av, torch, transformers, spacy, fpdf2, numpy, pandas |
| Web browser | Chrome / Edge / Firefox |
| Internet | Only for installation and first-time model download |

---

## 9. Advantages

1. **Saves time** — minutes are ready within about a minute of the meeting ending.
2. **Structured, actionable output** — clear list of decisions and tasks with owners and deadlines.
3. **No hallucinated facts in tasks** — names and dates are copied from the transcript; missing information is shown as "Not specified".
4. **Privacy** — recordings are processed on the local computer, not uploaded to any server.
5. **Zero cost** — only free, open-source software and models.
6. **Works without GPU and offline** after first setup.
7. **Supports audio, video and text input**; exports TXT, JSON and PDF.
8. **Configurable** — model size and summary method can be chosen according to hardware.
9. **Modular design** — each step is a separate module, easy to test, explain and upgrade.

---

## 10. Limitations

1. **No speaker identification from audio** — Whisper produces text without speaker names, so "I will…" statements in audio transcripts cannot be assigned to a person (they show "Not specified"). Pasted transcripts with `Name:` labels work fully.
2. **Rule-based extraction depends on phrasing** — unusual or indirect wording ("maybe someone could look into…") may be missed or mis-assigned.
3. **Accuracy depends on audio quality** — background noise, overlapping speakers, strong accents or poor microphones increase transcription errors, which then affect the summary.
4. **Abstractive model limitations** — DistilBART can occasionally produce generic sentences or merge details; it has a 1,024-token input window, so long meetings are summarised in chunks.
5. **Processing speed on CPU** — long meetings (1 hour+) can take several minutes; the `small` model is slow on weak laptops.
6. **English-focused** — Whisper can transcribe many languages, but the summarization model and extraction rules are designed for English.
7. **Relative deadlines are not converted to calendar dates** ("next Friday" is kept as text).
8. **Not real-time** — the system processes a recording after the meeting.

---

## 11. Applications

* **Business meetings** — automatic minutes of meetings (MoM), task tracking.
* **College classrooms** — lecture summaries and key points for revision.
* **Project team discussions** — tracking decisions and assignments in student/industry projects.
* **Interviews** — summarising candidate responses and key observations.
* **Online conferences and webinars** — quick recap for people who missed the session.
* **Government / committee meetings** — drafting official minutes.
* **Healthcare / legal consultations** — first-draft notes (with human review).
* **Podcasts and YouTube videos** — creating show notes.

---

## 12. Future Scope

1. **Real-time meeting summarization** — live transcription from the microphone with a rolling summary.
2. **Automatic meeting minutes** — formal MoM templates with agenda, attendees, and approval sections, shared by email.
3. **Better deadline extraction** — convert relative expressions ("next Friday") into exact calendar dates and create calendar reminders (.ics / Google Calendar).
4. **Speaker identification (diarization)** — detect *who spoke when* using models such as pyannote.audio.
5. **Speaker-wise summaries** — separate summaries and task lists for each participant.
6. **Multilingual meeting support** — Hindi, Hinglish and regional languages using multilingual models (mBART, mT5, IndicBART).
7. **Local LLM integration** — use small open-source LLMs (e.g. Llama/Phi/Qwen via Ollama) for richer, more natural summaries on better hardware.
8. **Integration with Zoom / Google Meet / Microsoft Teams** recordings and task tools like Trello or Jira.
9. **Sentiment and engagement analysis** of discussions.
10. **Searchable meeting archive** with semantic search across past meetings.

---

## 13. Conclusion

This project successfully demonstrates an end-to-end AI pipeline that transforms raw meeting recordings into structured, actionable summaries. By combining a state-of-the-art open-source speech recognition model (Whisper, via faster-whisper), careful text preprocessing, a hybrid information-extraction approach (spaCy NER, rule-based patterns and the TextRank graph algorithm) and a transformer-based abstractive summarizer (DistilBART fine-tuned on dialogue data), the system identifies key points, decisions, action items, responsible persons and deadlines with minimal human effort. The implementation runs entirely on an ordinary Windows laptop without a GPU, paid APIs or permanent internet access, which makes it affordable, private and practical for students and small organisations. The modular architecture allows each component to be improved independently, and the project clearly shows how speech processing and NLP can reduce manual effort, improve accountability and ensure that important meeting outcomes are not lost. With future enhancements such as speaker diarization, real-time processing and multilingual support, the system can evolve into a complete intelligent meeting assistant.

---

### References
1. A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)," OpenAI, 2022.
2. M. Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension," ACL, 2020.
3. B. Gliwa et al., "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization," 2019.
4. R. Mihalcea and P. Tarau, "TextRank: Bringing Order into Texts," EMNLP, 2004.
5. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
6. S. Shleifer and A. Rush, "Pre-trained Summarization Distillation," 2020.
7. spaCy documentation — https://spacy.io ; faster-whisper — https://github.com/SYSTRAN/faster-whisper ; Streamlit — https://streamlit.io
