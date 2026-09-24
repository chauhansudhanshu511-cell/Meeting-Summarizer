"""
information_extractor.py
------------------------
Step 4 of the pipeline: pull out the IMPORTANT INFORMATION from the meeting.

We use a HYBRID approach (easy to explain and reliable on a laptop):

  * spaCy NLP model (en_core_web_sm) - Named Entity Recognition (NER) finds
    people, organisations, dates and times; noun-chunk detection finds topics.
  * Rule-based NLP (regular expressions / keyword patterns) - finds
    decisions ("we decided", "let's go with"), action items ("I will",
    "Rahul, can you ...") and deadlines ("by Friday", "25th March").
  * TextRank (from summarizer.py) - ranks the most important sentences as key points.

Why rules for tasks and deadlines? A small summarization model often drops
or changes names and dates. Rules copy them EXACTLY from the transcript,
so we never invent a person or a deadline. If something is not mentioned,
we write "Not specified".
"""

import re
from collections import Counter

import config
from services.summarizer import STOPWORDS, extractive_summary

# ---------------------------------------------------------------------------
# spaCy loading (optional - the app still works without it)
# ---------------------------------------------------------------------------
_NLP = None
_NLP_LOADED = False


def get_nlp():
    """Load the spaCy English model once. Returns None if not installed."""
    global _NLP, _NLP_LOADED
    if not _NLP_LOADED:
        _NLP_LOADED = True
        try:
            import spacy
            _NLP = spacy.load(config.SPACY_MODEL)
        except Exception:
            _NLP = None
    return _NLP


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
ORDINAL = r"\d{1,2}(?:st|nd|rd|th)?"

DEADLINE_PATTERNS = [
    # "by the end of this week", "before end of the month", "by EOD"
    r"\b(?:by|before|till|until|within|on)\s+(?:the\s+)?end\s+of\s+(?:the\s+|this\s+|next\s+)?(?:day|week|month|semester|sprint)\b",
    r"\b(?:by|before)\s+(?:EOD|EOW|end of day)\b",
    # "by 25th March", "on March 25th", "before 10 April 2026"
    rf"\b(?:by|on|before|till|until|due)?\s*(?:the\s+)?{ORDINAL}\s+(?:of\s+)?{MONTHS}(?:,?\s+\d{{4}})?\b",
    rf"\b(?:by|on|before|till|until|due)?\s*{MONTHS}\s+{ORDINAL}(?:,?\s+\d{{4}})?\b",
    # numeric dates 25/03, 25-03-2026, 2026-03-25
    r"\b(?:by|on|before|due)?\s*\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    # "by next Friday", "on Monday", "this Thursday"
    rf"\b(?:by|on|before|till|until|this|next|coming)\s+(?:this\s+|next\s+|coming\s+)?{WEEKDAYS}(?:\s+(?:morning|evening|afternoon|night))?\b",
    rf"\b{WEEKDAYS}\b",
    # relative deadlines
    r"\b(?:by|before)?\s*(?:tomorrow|tonight|today)(?:\s+(?:morning|evening|afternoon|night))?\b",
    r"\b(?:by|before|within|in)\s+(?:the\s+)?(?:next|coming)\s+(?:\w+\s+)?(?:days?|weeks?|months?)\b",
    r"\b(?:within|in)\s+(?:\d+|one|two|three|four|five|six|seven|ten)\s+(?:days?|weeks?|months?|hours?)\b",
    r"\b(?:by|before)\s+next\s+(?:week|month|meeting|class|review)\b",
    r"\bnext\s+(?:week|month)\b",
    r"\b(?:by|before|at)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)",
]
DEADLINE_REGEX = re.compile("|".join(f"(?:{p})" for p in DEADLINE_PATTERNS), re.IGNORECASE)

DECISION_REGEX = re.compile(
    r"\b(?:we(?:'ve| have)?\s+(?:all\s+)?(?:decided|agreed|finali[sz]ed|concluded|chosen|settled)"
    r"|(?:it's|it is|that's|that is)\s+(?:decided|final|settled|agreed)"
    r"|let'?s\s+(?:go\s+with|use|finali[sz]e|stick\s+with|keep|choose|fix|freeze|go\s+ahead)"
    r"|(?:the\s+)?(?:final\s+)?decision\s+is"
    r"|decided\s+to|agreed\s+(?:to|on|that|upon)|we\s+will\s+go\s+with|we'll\s+go\s+with"
    r"|(?:is|are|has been|have been)\s+(?:approved|finali[sz]ed|confirmed)"
    r"|final(?:ly)?,?\s+we|we\s+(?:will|'ll)\s+(?:use|adopt|switch\s+to|keep)"
    r"|go\s+ahead\s+with|consensus)\b",
    re.IGNORECASE,
)

# Words that look like names at the start of a sentence but are not people
NOT_NAMES = {
    "We", "I", "You", "It", "This", "That", "They", "He", "She", "There", "Everyone", "Everybody",
    "Someone", "Somebody", "Anyone", "Nobody", "Also", "So", "And", "But", "Then", "Okay", "Ok",
    "Yes", "Yeah", "No", "Sure", "Great", "Good", "Fine", "Please", "Maybe", "Now", "Next", "Today",
    "Tomorrow", "The", "Our", "My", "Your", "Let", "Let's", "What", "Who", "When", "Where", "Why",
    "How", "Alright", "Right", "Well", "Thanks", "Thank", "Hello", "Hi", "Team", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Action", "Task", "One", "First", "Finally",
}

TASK_VERBS = r"(?:will|'ll|shall|is going to|are going to|am going to|can|should|needs? to|has to|have to|must)"
# Replies that accept a request ("Sure, I'll do it")
ACK_WORDS = re.compile(r"^(?:sure|okay|ok|yes|yeah|yep|alright|all right|will do|no problem|done|got it|i can|i will|i'll)\b",
                       re.IGNORECASE)
# Recap sentences repeat tasks already found earlier, so they are skipped for action items
RECAP_WORDS = re.compile(r"^(?:so,?\s+)?(?:to summari[sz]e|in summary|to conclude|to recap|summing up)\b", re.IGNORECASE)
# Sentences about the meeting itself ("Today we need to review ...") are not tasks
MEETING_META = re.compile(r"\b(?:this meeting|the meeting|today we|agenda|let'?s start|let'?s begin)\b", re.IGNORECASE)
# "one week before that" -> relative deadline
RELATIVE_DEADLINE = re.compile(r"\b(?:a|one|two|three|\d+)\s+(?:days?|weeks?)\s+before(?:\s+(?:that|it|this))?\b", re.IGNORECASE)
# Generic nouns that are never useful as "topics"
GENERIC_TOPIC_WORDS = {
    "time", "progress", "thing", "idea", "way", "lot", "week", "day", "everyone", "meeting", "today",
    "morning", "version", "detail", "point", "people", "guy", "problem", "question", "part", "bit", "sense",
    "work", "stuff", "minute", "hour", "month", "year", "moment", "end", "start", "case", "place", "kind",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_deadlines(sentence: str) -> list:
    """Return all deadline phrases found in a sentence (exact text)."""
    found = []
    for match in DEADLINE_REGEX.finditer(sentence):
        phrase = match.group(0).strip(" ,.")
        if phrase and phrase.lower() not in [f.lower() for f in found]:
            found.append(phrase)
    # keep the most specific phrase (e.g. "by next Friday" instead of "Friday")
    found = [f for f in found if not any(f.lower() != g.lower() and f.lower() in g.lower() for g in found)]
    return found


def _clean_task_text(task: str, deadlines: list) -> str:
    """Tidy the task description: remove the deadline phrase, filler starts, trailing punctuation."""
    for d in deadlines:
        task = re.sub(re.escape(d), "", task, flags=re.IGNORECASE)
    task = re.sub(r"^(?:also|then|please|definitely|probably|just|actually|basically)\s+", "", task.strip(),
                  flags=re.IGNORECASE)
    task = re.sub(r"\s+(?:okay|ok|right|please|then)\s*$", "", task.strip(" ,.?!"), flags=re.IGNORECASE)
    # cut off extra clauses that are not part of the task itself
    task = re.split(r",?\s+(?:and then|so that|because|since|but)\s+", task, maxsplit=1, flags=re.IGNORECASE)[0]
    task = re.sub(r"\s+([,.;:])", r"\1", task)
    task = re.sub(r"\s{2,}", " ", task).strip(" ,.?!;:-")
    # "I will" -> first-person words inside the task are changed to neutral form
    task = re.sub(r"\bmy\b", "their", task, flags=re.IGNORECASE)
    return task[0].upper() + task[1:] if task else task


# ---------------------------------------------------------------------------
# Entities and topics
# ---------------------------------------------------------------------------
def extract_entities(text: str, speakers: list) -> tuple:
    """
    Find people, organisations, dates/times, places and other named things.
    Returns (entities_dict, person_candidates). person_candidates = every name spaCy
    tagged as PERSON (used to detect task owners such as "Rahul will ...").
    """
    person_candidates = set(speakers)
    entities = {"People": [], "Organizations": [], "Dates & Times": [], "Places": [], "Other": []}
    # "Other" = tools, products, technologies and other named terms
    for speaker in speakers:
        if speaker not in entities["People"]:
            entities["People"].append(speaker)

    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(text[:100000])
        mapping = {"PERSON": "People", "ORG": "Organizations", "DATE": "Dates & Times",
                   "TIME": "Dates & Times", "GPE": "Places", "LOC": "Places", "FAC": "Places",
                   "PRODUCT": "Other", "EVENT": "Other", "WORK_OF_ART": "Other", "LAW": "Other",
                   "LANGUAGE": "Other", "NORP": "Other"}
        for ent in doc.ents:
            group = mapping.get(ent.label_)
            value = ent.text.strip(" ,.'\"")
            if not group or len(value) < 2:
                continue
            if group == "People":
                value = re.sub(r"'s$", "", value)
                if value in NOT_NAMES or not value[0].isupper():
                    continue
                person_candidates.add(value.split()[0])
                # small models sometimes tag tools (e.g. "Streamlit") as people. We accept a
                # name only if it is a speaker, has a title (Dr./Mr./Prof.) or is a full name.
                has_title = re.search(rf"\b(?:Dr|Mr|Mrs|Ms|Prof|Sir|Madam)\.?\s+{re.escape(value)}", text)
                if value not in speakers and not has_title and " " not in value:
                    group = "Other"
            if group == "Dates & Times" and value.lower() in {"morning", "evening", "afternoon", "night", "now"}:
                continue
            existing = [e.lower() for e in entities[group]]
            if value.lower() not in existing:
                entities[group].append(value)
        # Whisper transcripts have no "Name:" labels and the small spaCy model can miss
        # Indian / uncommon names, so proper nouns (PROPN) with no other entity type are
        # also kept as possible task owners ("Rahul will prepare the slides").
        for token in doc:
            if (token.pos_ == "PROPN" and token.ent_type_ in ("", "PERSON") and token.text.istitle()
                    and token.text.isalpha() and token.text not in NOT_NAMES and len(token.text) > 2):
                person_candidates.add(token.text)
    else:
        # Fallback without spaCy: capitalised words in the middle of sentences = likely names
        candidates = re.findall(r"(?<=[a-z,] )([A-Z][a-z]{2,})\b", text)
        for word, _ in Counter(candidates).most_common(10):
            if word not in NOT_NAMES and word not in entities["People"]:
                entities["Other"].append(word)
        for d in find_deadlines(text):
            if d not in entities["Dates & Times"]:
                entities["Dates & Times"].append(d)

    return entities, person_candidates


def extract_topics(text: str, people: list, top_n: int = config.NUM_TOPICS) -> list:
    """
    Find the main topics = the most frequently discussed noun phrases.
    Example: "database design", "user interface", "project report".
    """
    people_lower = {p.lower() for p in people}
    counter = Counter()
    display = {}

    nlp = get_nlp()
    if nlp is not None:
        doc = nlp(text[:100000])
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ == "PRON":
                continue
            # keep only meaningful words (drop "the", "our", "some" ...)
            words = [t for t in chunk if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "ADJ")]
            if not words or all(t.pos_ == "ADJ" for t in words):
                continue
            phrase = " ".join(t.lemma_.lower() if t.pos_ == "NOUN" else t.text.lower() for t in words)
            if len(phrase) < 3 or phrase in people_lower or phrase in STOPWORDS or phrase in GENERIC_TOPIC_WORDS:
                continue
            if any(w in people_lower for w in phrase.split()) or find_deadlines(phrase):
                continue
            counter[phrase] += 1
            display.setdefault(phrase, " ".join(t.text for t in words))
    else:
        tokens = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
                  if w not in STOPWORDS and len(w) > 3 and w not in people_lower and w not in GENERIC_TOPIC_WORDS]
        counter.update(tokens)
        counter.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))

    # score: frequency, with a bonus for multi-word phrases (they are more specific)
    scored = sorted(counter.items(), key=lambda kv: kv[1] * (1.5 if " " in kv[0] else 1.0), reverse=True)

    topics = []
    for phrase, count in scored:
        if count < 2 and len(topics) >= 3:
            continue  # after 3 topics, only keep phrases mentioned more than once
        # skip a phrase already covered by a chosen topic ("database" vs "database design")
        if any(phrase in t.lower() or t.lower() in phrase for t in topics):
            continue
        label = display.get(phrase, phrase)
        # keep original capitals for names like "SQLite"; otherwise use Title Case
        topics.append(label if any(c.isupper() for c in label[1:]) else label[0].upper() + label[1:])
        if len(topics) >= top_n:
            break
    return topics


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------
def extract_decisions(sentences: list) -> list:
    """Sentences that contain decision language (and are not questions)."""
    decisions = []
    for item in sentences:
        text = item["text"]
        if text.endswith("?") or len(text.split()) < 4:
            continue
        if DECISION_REGEX.search(text):
            cleaned = re.sub(r"^(?:okay|ok|so|alright|great|good|fine|right|then|yes|yeah)[,.]?\s+", "", text,
                             flags=re.IGNORECASE)
            cleaned = cleaned[0].upper() + cleaned[1:]
            if cleaned.lower() not in [d["decision"].lower() for d in decisions]:
                decisions.append({"decision": cleaned, "said_by": item["speaker"] or config.NOT_SPECIFIED})
    return decisions


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------
def _match_action(text: str, speaker, known_names: set):
    """
    Try the action-item rules on ONE sentence.
    Returns (owner, task_text, needs_acknowledgement) or None.
    """
    # Rule 1: "Rahul, can you / could you / please / you will / you should ..."
    m = re.search(r"\b([A-Z][a-z]+),?\s+(?:can you|could you|would you|will you|please|you(?:'ll| will| should| need to| have to| can))\s+(?P<task>.+)",
                  text)
    if m and m.group(1) in known_names:
        return m.group(1), m.group("task"), False

    # Rule 2: "... assign the testing to Rahul" / "Rahul is responsible for testing"
    m = re.search(r"\bassign(?:ed|ing)?\s+(?P<task>.+?)\s+to\s+([A-Z][a-z]+)", text)
    if m and m.group(2) in known_names:
        return m.group(2), m.group("task"), False
    m = re.search(r"\b([A-Z][a-z]+)\s+(?:is|will be)\s+(?:responsible for|in charge of|handling|taking care of)\s+(?P<task>.+)", text)
    if m and m.group(1) in known_names:
        return m.group(1), m.group("task"), False

    # Rule 3: "Rahul will prepare the slides" / "Sneha and Arjun will ..."
    m = re.search(rf"\b([A-Z][a-z]+)(?:\s+and\s+([A-Z][a-z]+))?\s+{TASK_VERBS}\s+(?:also\s+)?(?P<task>.+)", text)
    if m and m.group(1) in known_names:
        owner = m.group(1) + (f" & {m.group(2)}" if m.group(2) and m.group(2) in known_names else "")
        return owner, m.group("task"), False

    # Rule 4: "I will / I'll / I can take care of / I am going to ..." -> the speaker
    m = re.search(r"\b(?:I(?:'ll| will| shall| am going to|'m going to| can| need to| have to| must)|"
                  r"let me)\s+(?:also\s+)?(?P<task>.+)", text)
    if m:
        return (speaker or config.NOT_SPECIFIED), m.group("task"), False

    # Rule 5: "We need to / We have to / We must / Everyone should ..." -> the team
    m = re.search(r"\b(?:we(?:'ll| will| need to| have to| must| should)|everyone (?:should|must|needs to|has to)|"
                  r"all of us (?:should|must|need to))\s+(?P<task>.+)", text, re.IGNORECASE)
    if m and not DECISION_REGEX.search(text) and not MEETING_META.search(text):
        return "Entire team", m.group("task"), False

    # Rule 6: "Can you send ... ?" / "Someone needs to ..." -> owner = whoever says "Sure" next
    m = re.search(r"\b(?:can you|could you|would you|please|someone (?:needs|has) to|somebody should)\s+(?P<task>.+)",
                  text, re.IGNORECASE)
    if m:
        return config.NOT_SPECIFIED, m.group("task"), True

    # Rule 7: explicit "action item: ..." / "task: ..."
    m = re.search(r"\b(?:action item|todo|to-do|task)\s*(?:is|:|-)\s*(?P<task>.+)", text, re.IGNORECASE)
    if m:
        return config.NOT_SPECIFIED, m.group("task"), False
    return None


def extract_action_items(sentences: list, known_names: set) -> list:
    """Find tasks, who is responsible, and deadlines."""
    actions = []
    for index, item in enumerate(sentences):
        text = item["text"]
        if len(text.split()) < 4:
            continue
        # skip pure questions unless they are requests ("can you ...?")
        if text.endswith("?") and not re.search(r"\b(?:can|could|would|will) you\b", text, re.IGNORECASE):
            continue
        # skip past-tense progress reports ("I have completed the ...")
        if re.search(r"\bI(?:'ve| have| had)\s+(?:already\s+)?\w+ed\b", text) and not re.search(r"\bwill\b|'ll\b", text):
            continue

        if RECAP_WORDS.search(text):
            continue

        result = _match_action(text, item["speaker"], known_names)
        if not result:
            continue
        owner, task, needs_ack = result
        is_request = needs_ack or bool(re.search(r"\b(?:can|could|would|will) you\b|\byou(?:'ll| will| should)\b", text, re.I))

        # Who accepted the request? Look at the next sentence from a DIFFERENT speaker.
        if needs_ack and item["speaker"]:
            for nxt in sentences[index + 1:index + 3]:
                if nxt["speaker"] and nxt["speaker"] != item["speaker"] and ACK_WORDS.search(nxt["text"]):
                    owner = nxt["speaker"]
                    break

        deadlines = find_deadlines(text)
        # for a request, the deadline may be given in the reply: "Sure, I'll send it by Monday."
        if not deadlines and is_request and index + 1 < len(sentences) \
                and ACK_WORDS.search(sentences[index + 1]["text"]):
            deadlines = find_deadlines(sentences[index + 1]["text"])

        task_text = _clean_task_text(task, deadlines)
        deadline_text = ", ".join(deadlines) if deadlines else config.NOT_SPECIFIED

        # "... is on 25th March, so we must finish testing one week before that"
        relative = RELATIVE_DEADLINE.search(task_text)
        if relative:
            base = re.sub(r"^(?:by|on|before|till|until|due)\s+", "", deadlines[0], flags=re.I) if deadlines else ""
            phrase = re.sub(r"\s+(?:that|it|this)$", "", relative.group(0), flags=re.I)
            deadline_text = f"{phrase} {base}".strip() if base else relative.group(0)
            task_text = _clean_task_text(task_text.replace(relative.group(0), ""), [])
        if len(task_text.split()) < 2:
            continue
        # skip vague tasks such as "Do that" / "Discuss it"
        if re.fullmatch(r"(?:do|discuss|check|see|think about|talk about)\s+(?:it|that|this|them)", task_text, re.I):
            continue

        # merge duplicates: the same task is often said twice
        # ("Rahul, can you collect the images?" -> "Yes, I will collect the images by Friday")
        duplicate = next((a for a in actions if _similar(a["task"], task_text)
                          and (a["responsible"] in (owner, config.NOT_SPECIFIED) or owner == config.NOT_SPECIFIED)), None)
        if duplicate:
            if duplicate["responsible"] == config.NOT_SPECIFIED:
                duplicate["responsible"] = owner
            if duplicate["deadline"] == config.NOT_SPECIFIED:
                duplicate["deadline"] = deadline_text
            if len(task_text) > len(duplicate["task"]):   # keep the more detailed wording
                duplicate["task"] = task_text
            continue

        actions.append({
            "task": task_text,
            "responsible": owner or config.NOT_SPECIFIED,
            "deadline": deadline_text,
            "source": text,
        })
    return actions


def _similar(task_a: str, task_b: str, threshold: float = 0.6) -> bool:
    """Two tasks are 'the same' if most words of the shorter task appear in the other one
    (overlap coefficient = shared words / words in the shorter task)."""
    def words(t):
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", t.lower())
                if w not in STOPWORDS and len(w) > 2 and w not in {"finish", "complete", "start", "also"}}
    a, b = words(task_a), words(task_b)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= threshold


def extract_all_deadlines(sentences: list) -> list:
    """Every deadline/date mentioned in the meeting, with the sentence it came from."""
    results = []
    for item in sentences:
        for d in find_deadlines(item["text"]):
            results.append({"deadline": d, "context": item["text"], "said_by": item["speaker"] or config.NOT_SPECIFIED})
    return results


# ---------------------------------------------------------------------------
# Key points and "Main Key Points"
# ---------------------------------------------------------------------------
# Sentences that are only greetings / small talk are never key points
SMALL_TALK = re.compile(
    r"^(?:good (?:morning|afternoon|evening)|hello|hi\b|thank(?:s| you)|let'?s (?:start|begin|meet again)|"
    r"see you|bye|that'?s all|any (?:other )?questions)", re.IGNORECASE)
# Words at the start of a spoken sentence that add nothing ("So, ...", "Okay, ...")
LEADING_WORDS = re.compile(
    r"^(?:so|okay|ok|also|now|well|yes|yeah|sure|right|alright|perfect|great|good|fine|and|but|actually|"
    r"basically|anyway|hmm|that makes sense|that'?s good progress(?:, \w+)?)[,.!]?\s+", re.IGNORECASE)


def clean_key_point(sentence: str) -> str:
    """Make a spoken sentence read like a note: remove 'So,', 'Okay,' etc. at the start."""
    text = sentence.strip()
    previous = None
    while previous != text:            # remove several leading words: "Okay, so, ..."
        previous = text
        text = LEADING_WORDS.sub("", text)
    text = text.strip(" ,")
    if not text:
        return sentence
    if text[-1] not in ".!?":
        text += "."
    return text[0].upper() + text[1:]


def extract_key_points(sentences: list, decisions: list, action_items: list) -> list:
    """
    Key discussion points = the most important sentences (TextRank) that are NOT
    small talk, questions, recaps, or already listed as a decision / action item.
    """
    already_used = {d["decision"].lower() for d in decisions} | {a["source"].lower() for a in action_items}
    candidates, speaker_of = [], {}
    for item in sentences:
        text = item["text"]
        if text.endswith("?") or SMALL_TALK.search(text) or RECAP_WORDS.search(text) or MEETING_META.search(text):
            continue
        if text.lower() in already_used or any(text.lower() in u or u in text.lower() for u in already_used):
            continue
        if _match_action(text, None, set()):   # task sentences are shown as action items instead
            continue
        candidates.append(text)
        speaker_of[text] = item["speaker"]

    top = extractive_summary(candidates, config.NUM_KEY_POINTS, min_words=7)
    return [
        f"{clean_key_point(kp)} ({speaker_of[kp]})" if speaker_of.get(kp) else clean_key_point(kp)
        for kp in top
    ]


def build_main_points(key_points: list, decisions: list, action_items: list) -> list:
    """
    "Main Key Points" shown next to the summary: a short, mixed list of the most
    important discussion points, every decision, and every task with its owner/deadline.
    Returns [{"type": "Discussion" | "Decision" | "Action", "text": "..."}]
    """
    points = []
    for kp in key_points[:config.NUM_MAIN_DISCUSSION_POINTS]:
        points.append({"type": "Discussion", "text": re.sub(r"\s\([^)]*\)$", "", kp)})
    for d in decisions:
        points.append({"type": "Decision", "text": clean_key_point(d["decision"])})
    for a in action_items:
        text = a["task"]
        if a["responsible"] != config.NOT_SPECIFIED:
            text = f"{a['responsible']}: {text[0].lower() + text[1:]}"
        if a["deadline"] != config.NOT_SPECIFIED:
            text += f" ({a['deadline']})"
        points.append({"type": "Action", "text": text})
    return points


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def extract_information(preprocessed: dict) -> dict:
    """
    Run every extractor on the cleaned transcript.

    Args:
        preprocessed: output of services.preprocessing.preprocess_transcript()
    """
    sentences = preprocessed["sentences"]
    plain_text = preprocessed["plain_text"]
    speakers = preprocessed["speakers"]

    entities, person_candidates = extract_entities(plain_text, speakers)
    known_names = {n for n in person_candidates if n and n not in NOT_NAMES}

    decisions = extract_decisions(sentences)
    action_items = extract_action_items(sentences, known_names)
    key_points = extract_key_points(sentences, decisions, action_items)

    return {
        "topics": extract_topics(plain_text, entities["People"]),
        "key_points": key_points,
        "main_points": build_main_points(key_points, decisions, action_items),
        "decisions": decisions,
        "action_items": action_items,
        "deadlines": extract_all_deadlines(sentences),
        "entities": entities,
    }
