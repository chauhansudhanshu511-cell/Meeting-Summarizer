# Presentation & Viva Preparation
### AI-Based Smart Meeting Summarization

---

## A. 2-minute project explanation

> Good morning, ma'am/sir. Our mini project is **AI-Based Smart Meeting Summarization**.
>
> In colleges and offices, many meetings happen every day, but the important outcomes — what was decided, who has to do which task, and by when — are often lost because someone has to write notes manually. Listening to the full recording again takes a lot of time.
>
> Our system solves this problem. The user uploads a meeting recording — MP3, WAV, M4A or even an MP4 video — or simply pastes a transcript. First, the system extracts the audio and converts it into text using **Whisper**, an open-source speech recognition model by OpenAI. We use **faster-whisper**, which runs quickly on a normal laptop CPU.
>
> Then the transcript is **cleaned** — filler words like "um" and "uh", noise tags and repetitions are removed. After that, our NLP module finds the **key points, decisions, action items, responsible persons, deadlines and topics** using spaCy, rule-based patterns and the TextRank algorithm. Finally, a transformer model called **DistilBART**, trained on dialogue summaries, writes a short overview of the meeting.
>
> The output is shown in a clean **Streamlit** web page and can be downloaded as TXT, JSON or PDF.
>
> The main advantages are: it is free, it runs completely **offline on our laptop without a GPU**, the data stays private, and it never invents information — if a deadline is not mentioned, it shows "Not specified". Let me show you a demo.

---

## B. 5-minute project explanation (with demo)

**1. Introduction (30 s)**
"Our project is AI-Based Smart Meeting Summarization. It converts meeting audio or video into a structured summary with key points, decisions and action items using AI. It is built fully in Python and runs locally."

**2. Problem (40 s)**
"Manual minutes are slow, incomplete and depend on one person. Often tasks are noted without an owner or deadline. Commercial AI tools exist but are paid, need internet and upload private recordings to the cloud. So our goal was a free, private, offline system that works on an ordinary laptop."

**3. Architecture — show the flow diagram (1 min 30 s)**
"The system follows six steps:
1. **Processing file** — we use PyAV, which contains FFmpeg, to decode the file. For MP4 we take only the audio track and convert it to 16 kHz mono WAV, the format Whisper expects.
2. **Speech-to-text** — faster-whisper, the optimised version of OpenAI Whisper. Whisper is a transformer encoder–decoder trained on 680,000 hours of speech. We use the *base* model with int8 quantisation on the CPU, and voice activity detection to skip silence.
3. **Preprocessing** — regular expressions remove fillers, noise like [Music], stutters and repeated sentences, and detect speaker names written as 'Name: text'.
4. **Information extraction** — spaCy finds names and dates; noun phrases become topics; TextRank ranks important sentences as key points; pattern rules find decisions like 'we decided' or 'let's go with', action items like 'I will…' or 'Rahul, can you…', and deadlines like 'by Friday' or '25th March'.
5. **Summarization** — DistilBART fine-tuned on the SAMSum dialogue dataset writes an abstractive summary. Long meetings are split into chunks because the model reads 1,024 tokens at a time. If the model is not available, TextRank is used as a fallback.
6. **Final output** — overview, key points, decisions, action items table, topics, deadlines and a short summary, downloadable as TXT, JSON and PDF."

**4. Live demo (1 min 30 s)**
* Click **Load sample transcript** → **Generate Summary**. Show the progress steps.
* Show **Summary**, **Decisions** (SQLite instead of MySQL, Streamlit for UI, email feature as future scope).
* Show **Action Items** table — Rahul: collect images by Friday; Sneha: database tables by next Monday; Arjun: dashboard within five days; Priya: slides by 18th March.
* Upload `sample_meeting.mp4` to show video → audio → transcript.
* Show the **Transcript** tab (original vs cleaned) and download the **PDF**.

**5. Conclusion (30 s)**
"The project shows how speech recognition and NLP can automate meeting minutes. It is free, private and runs without GPU. In future we can add speaker identification, real-time summarization and multilingual support. Thank you."

---

## C. Viva questions and answers

**1. What is the objective of your project?**
To automatically convert meeting audio/video or transcripts into a structured summary — overview, key points, decisions, action items with responsible person and deadline, and topics — using free AI models that run locally on a laptop.

**2. Why did you choose this topic?**
Meetings happen everywhere, and manual note-taking is slow and error-prone. Important tasks and deadlines get lost. It is a real-world problem, and it lets us combine two major AI areas — speech recognition and natural language processing — in one practical project.

**3. What is Speech-to-Text?**
Speech-to-Text is the process of converting spoken audio into written text automatically. The computer analyses the sound waves, recognises the words and outputs a transcript.

**4. What is ASR?**
ASR stands for **Automatic Speech Recognition** — the technology behind speech-to-text. Modern ASR systems use deep learning: audio is converted into a spectrogram (a picture of frequencies over time), a neural network learns acoustic patterns, and a decoder predicts the most likely sequence of words.

**5. What is Whisper and why is it used?**
Whisper is an open-source ASR model released by OpenAI. It is a transformer encoder–decoder trained on 680,000 hours of multilingual audio from the internet, so it handles accents, noise and many languages well. We chose it because it is free, very accurate, works offline and has small model sizes suitable for laptops.

**6. What is faster-whisper? Why not the original Whisper?**
faster-whisper is a re-implementation of Whisper using the CTranslate2 inference engine. It gives the same accuracy but is about 4× faster and uses less memory on a CPU, especially with int8 quantisation. Since we don't have a GPU, it is the best choice.

**7. What is quantisation (int8)?**
Quantisation stores model weights in 8-bit integers instead of 32-bit floating point numbers. This makes the model about 4× smaller in memory and faster on CPU, with only a very small loss in accuracy.

**8. What is NLP?**
Natural Language Processing is a branch of AI that enables computers to understand, analyse and generate human language. In our project NLP is used for cleaning text, finding names and dates, detecting decisions and tasks, and generating the summary.

**9. What is a transformer?**
A transformer is a deep-learning architecture introduced in the paper "Attention Is All You Need" (2017). It uses a **self-attention** mechanism, which lets the model look at all words in a sentence at once and decide which words are related to each other. Transformers process text in parallel and understand long-range context better than older RNN/LSTM models. Both Whisper and BART are transformers.

**10. What is abstractive summarization?**
Abstractive summarization generates **new sentences** that capture the meaning of the original text, like a human writing a summary in their own words. Example: from a long discussion it may write "The team decided to use SQLite and assigned tasks for the report and dashboard." Our DistilBART model does this.

**11. What is extractive summarization?**
Extractive summarization **selects the most important existing sentences** from the text and combines them, without changing words. It is faster and never introduces wrong facts, but it may be less fluent. Our TextRank method is extractive.

**12. Which algorithm/model is used in your project?**
* Speech recognition: **Whisper (base)** via faster-whisper.
* Abstractive summary: **DistilBART-CNN-12-6 fine-tuned on SAMSum** (transformer, sequence-to-sequence).
* Extractive summary and key points: **TextRank** (graph-based, uses the PageRank algorithm).
* Entities and topics: **spaCy en_core_web_sm** (statistical NER, noun chunks).
* Decisions, action items and deadlines: **rule-based pattern matching** using regular expressions.

**13. Why did you choose DistilBART-SAMSum?**
BART is a strong summarization model. DistilBART is a "distilled" (compressed) version — about 40% smaller and faster while keeping most of the quality — so it runs on a laptop CPU. The SAMSum version is fine-tuned on **dialogue** summaries (chat conversations), which is closer to meetings than news articles. Large LLMs like Llama need 8–16 GB RAM or a GPU and would be too slow for our demo.

**14. What is TextRank?**
TextRank is an unsupervised, graph-based ranking algorithm inspired by Google's PageRank. Each sentence is a node; two sentences are connected if they share words (weighted by similarity). A sentence that is similar to many other important sentences gets a high score. We select the top-scoring sentences as key points.

**15. How does the system extract action items?**
We use rule-based NLP on each sentence:
* "**I will / I'll / I can** …" → task owner is the **speaker**.
* "**Rahul will** …" → owner is Rahul (name must be a speaker or detected person).
* "**Sneha, can you** …" → owner is Sneha.
* "**Can you** …?" followed by "Sure, I'll…" from another person → owner is the person who agreed.
* "**We need to / we must** …" → owner is the entire team.
* "**assign X to Y**", "**Y is responsible for X**".
The task text is cleaned (the deadline phrase is removed), and similar duplicates are merged. If no owner is found, we write "Not specified".

**16. How are deadlines identified?**
Using regular-expression patterns for:
* Days — "by Friday", "next Monday", "this Thursday".
* Dates — "25th March", "March 25", "25/03/2026".
* Relative time — "tomorrow", "within five days", "by next week", "end of the month".
* Relative to another date — "one week before 25th March".
spaCy's DATE entities are also shown in the entities section. If no date is mentioned, we show "Not specified".

**17. How are decisions detected?**
Sentences containing decision phrases such as "we decided", "we agreed", "let's go with", "final decision is", "finalised", "approved", and that are not questions, are listed as decisions along with who said them.

**18. What preprocessing do you perform?**
Removing transcription noise ([Music], (inaudible)), filler words (um, uh, hmm), stuttered words ("the the"), repeated consecutive sentences, extra spaces and punctuation errors, capitalising sentences, and detecting speaker labels. We do not remove normal words, so decisions and tasks remain intact.

**19. Does the project require internet?**
Only for installing libraries and the **first-time download** of models (Whisper ~145 MB, DistilBART ~1.2 GB, spaCy ~12 MB). The models are saved in the `models` folder. After that, the project runs **completely offline**.

**20. Can it run without GPU?**
Yes. CPU mode is the default. We use faster-whisper with int8 quantisation and a distilled summarization model, both designed to be light enough for CPU. A GPU would only make it faster.

**21. What happens if the audio quality is poor?**
Whisper is fairly robust to noise, and voice activity detection skips silence, but heavy noise, overlapping speakers or a distant microphone will cause transcription mistakes, and those mistakes pass into the summary. We can improve it by choosing the larger `small` model, recording closer to the microphone, or selecting the correct language. If no speech is detected, the app shows a clear error message.

**22. How does the system handle long meetings?**
Whisper processes audio in 30-second windows internally, so any length works (we limit to 120 minutes for a laptop demo and show a warning above 30 minutes). The summarization model reads at most 1,024 tokens, so the transcript is split into chunks of about 550 words; each chunk is summarised and the results are combined (and summarised again if very long).

**23. What are the limitations of your project?**
No automatic speaker identification from audio; rule-based extraction depends on how people phrase things; accuracy depends on audio quality; the summary model can sometimes be generic; processing long meetings on CPU takes time; mainly English; relative dates are not converted to calendar dates.

**24. What is the future scope?**
Real-time summarization during live meetings, automatic formal minutes emailed to participants, better deadline extraction with calendar integration, speaker identification (diarization) and speaker-wise summaries, multilingual support (Hindi/Hinglish), integration with Zoom/Meet/Teams, and using small local LLMs for richer summaries.

**25. Why Streamlit for the interface?**
Streamlit turns a Python script into an interactive web app with a few lines of code — file upload, buttons, progress bars, tabs and download buttons are built in. No HTML/CSS/JavaScript or separate backend server is needed, and it runs on `localhost`, so it is ideal for a mini project demo.

**26. What is Named Entity Recognition (NER)?**
NER is an NLP task that finds and classifies names in text into categories such as PERSON, ORGANIZATION, DATE, TIME and LOCATION. We use spaCy's pre-trained English model for this.

**27. What is the difference between your system and ChatGPT-like tools?**
ChatGPT-style tools are large cloud LLMs that need internet and often a subscription, and the data is sent to a server. Our system uses small open-source models that run on the laptop, is free, private, works offline and gives a fixed structured output. Large LLMs may write more fluent summaries, but they need much more computing power.

**28. How do you ensure the system does not give wrong information?**
Names, tasks and deadlines are extracted with rules that copy text exactly from the transcript; if something is not mentioned we show "Not specified" instead of guessing. The transcript and cleaned transcript are displayed so the user can verify every point.

**29. What is the role of the virtual environment?**
A virtual environment (`venv`) is an isolated Python installation for the project, so the library versions it needs do not conflict with other projects on the computer.

**30. What output formats are supported?**
On-screen cards and tables, plus TXT (readable minutes), JSON (structured data for other programs) and PDF (printable report). All results are also auto-saved in the `outputs` folder.

**31. How are the "Main Key Points" shown with the summary generated?**
They combine three sources: (1) the top discussion sentences chosen by TextRank, after removing greetings, questions, recap lines and sentences that are already tasks or decisions, and cleaning words like "So," or "Okay," from the start; (2) every detected decision; and (3) every action item written as "Person: task (deadline)". This gives a one-glance list of what was discussed, what was decided and who has to do what.

---

## Demo checklist (day before the viva)
- [ ] Run `python download_models.py` at home with internet.
- [ ] Turn Wi-Fi off and test once → everything should still work.
- [ ] Keep the laptop charged / plugged in (battery saver makes the CPU slow).
- [ ] Pre-open the app (`run_app.bat`) before your turn — first load takes ~20 s.
- [ ] Keep `sample_meeting.mp3` / `.mp4` and your own recording ready.
- [ ] If anything fails, use **Load sample transcript** + **Fast (Extractive)** mode — it needs no models.
