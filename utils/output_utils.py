"""
output_utils.py
---------------
Turn the final result dictionary into downloadable files: TXT, JSON and PDF.
"""

import json
import os
from datetime import datetime

import config


def _duration_text(seconds: float) -> str:
    if not seconds:
        return "-"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes} min {secs} sec"


def group_main_points(result: dict) -> list:
    """[(label, [texts]), ...] for Discussion / Decisions / Action items (empty groups skipped)."""
    points = result.get("main_points")
    if not points:  # results saved by the older version only had "key_points"
        return [("Key discussion points", result["key_points"])] if result["key_points"] else []
    groups = []
    for kind, label in (("Discussion", "Key discussion points"), ("Decision", "Decisions"),
                        ("Action", "Action items")):
        items = [p["text"] for p in points if p["type"] == kind]
        if items:
            groups.append((label, items))
    return groups


# ---------------------------------------------------------------------------
# TXT (formatted like the "Meeting Summary" template)
# ---------------------------------------------------------------------------
def result_to_text(result: dict, include_transcript: bool = True) -> str:
    meta = result["meta"]
    lines = []
    lines.append("=" * 70)
    lines.append("MEETING SUMMARY  -  AI-Based Smart Meeting Summarization")
    lines.append("=" * 70)
    lines.append(f"Generated on    : {meta['generated_at']}")
    lines.append(f"Source          : {meta['source']}")
    if meta.get("audio_duration_seconds"):
        lines.append(f"Audio duration  : {_duration_text(meta['audio_duration_seconds'])}")
    lines.append(f"Summary method  : {meta['summary_method']}")
    if result.get("participants"):
        lines.append(f"Participants    : {', '.join(result['participants'])}")
    lines.append("")

    lines.append("1. OVERVIEW")
    lines.append("-" * 70)
    lines.append(result["summary"]["overview"])
    lines.append("")

    lines.append("2. MAIN KEY POINTS")
    lines.append("-" * 70)
    for label, items in group_main_points(result):
        lines.append(f"  {label}:")
        lines.extend(f"    * {item}" for item in items)
    if not result.get("main_points") and not result["key_points"]:
        lines.append("  None identified")
    lines.append("")

    lines.append("3. DECISIONS MADE")
    lines.append("-" * 70)
    if result["decisions"]:
        lines.extend(f"  * {d['decision']}" for d in result["decisions"])
    else:
        lines.append("  No explicit decisions were detected.")
    lines.append("")

    lines.append("4. ACTION ITEMS")
    lines.append("-" * 70)
    if result["action_items"]:
        for i, a in enumerate(result["action_items"], 1):
            lines.append(f"  {i}. Task       : {a['task']}")
            lines.append(f"     Responsible: {a['responsible']}")
            lines.append(f"     Deadline   : {a['deadline']}")
    else:
        lines.append("  No action items were detected.")
    lines.append("")

    lines.append("5. IMPORTANT TOPICS")
    lines.append("-" * 70)
    if result["topics"]:
        lines.extend(f"  * {t}" for t in result["topics"])
    else:
        lines.append("  None identified")
    lines.append("")

    lines.append("6. SHORT SUMMARY")
    lines.append("-" * 70)
    lines.append(result["summary"]["short_summary"])
    lines.append("")

    if result["deadlines"]:
        lines.append("DEADLINES / DATES MENTIONED")
        lines.append("-" * 70)
        lines.extend(f"  * {d['deadline']}  -  \"{d['context']}\"" for d in result["deadlines"])
        lines.append("")

    if include_transcript:
        lines.append("CLEANED TRANSCRIPT")
        lines.append("-" * 70)
        lines.append(result["transcript"]["cleaned"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# PDF (uses fpdf2 - pure Python, no extra software needed)
# ---------------------------------------------------------------------------
def _pdf_safe(text: str) -> str:
    """The built-in PDF fonts only support Latin-1 characters, so replace the rest."""
    replacements = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                    "—": "-", "…": "...", "•": "*", "₹": "Rs."}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def result_to_pdf(result: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def heading(text):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(width, 8, _pdf_safe(text), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(31, 78, 121)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10.5)

    def paragraph(text):
        pdf.multi_cell(width, 5.5, _pdf_safe(text), new_x="LMARGIN", new_y="NEXT")

    def bullets(items, empty_message):
        if not items:
            paragraph(empty_message)
        for item in items:
            pdf.multi_cell(width, 5.5, _pdf_safe(f"-  {item}"), new_x="LMARGIN", new_y="NEXT")

    meta = result["meta"]
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(31, 78, 121)
    pdf.cell(width, 10, "Meeting Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(90, 90, 90)
    info = f"Generated: {meta['generated_at']}   |   Source: {meta['source']}   |   Method: {meta['summary_method']}"
    pdf.multi_cell(width, 5, _pdf_safe(info), align="C", new_x="LMARGIN", new_y="NEXT")
    if result.get("participants"):
        pdf.multi_cell(width, 5, _pdf_safe("Participants: " + ", ".join(result["participants"])),
                       align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    heading("1. Overview")
    paragraph(result["summary"]["overview"])

    heading("2. Main Key Points")
    groups = group_main_points(result)
    if not groups:
        paragraph("None identified.")
    for label, items in groups:
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.cell(width, 6, _pdf_safe(label), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10.5)
        bullets(items, "")

    heading("3. Decisions Made")
    bullets([d["decision"] for d in result["decisions"]], "No explicit decisions were detected.")

    heading("4. Action Items")
    if result["action_items"]:
        pdf.set_font("Helvetica", "", 9.5)
        with pdf.table(col_widths=(60, 20, 20), text_align="LEFT", line_height=5.5) as table:
            header = table.row()
            for title in ("Task", "Responsible Person", "Deadline"):
                header.cell(title)
            for a in result["action_items"]:
                row = table.row()
                row.cell(_pdf_safe(a["task"]))
                row.cell(_pdf_safe(a["responsible"]))
                row.cell(_pdf_safe(a["deadline"]))
        pdf.set_font("Helvetica", "", 10.5)
    else:
        paragraph("No action items were detected.")

    heading("5. Important Topics")
    bullets(result["topics"], "None identified.")

    heading("6. Short Summary")
    paragraph(result["summary"]["short_summary"])

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Save all formats into the "outputs" folder
# ---------------------------------------------------------------------------
def save_outputs(result: dict) -> dict:
    """Save TXT, JSON (and PDF if possible). Returns {format: file_path}."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(config.OUTPUTS_DIR, f"meeting_summary_{stamp}")
    paths = {}

    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(result_to_text(result))
    paths["txt"] = base + ".txt"

    with open(base + ".json", "w", encoding="utf-8") as f:
        f.write(result_to_json(result))
    paths["json"] = base + ".json"

    try:
        with open(base + ".pdf", "wb") as f:
            f.write(result_to_pdf(result))
        paths["pdf"] = base + ".pdf"
    except Exception:
        pass  # PDF is optional
    return paths
