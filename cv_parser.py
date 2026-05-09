"""
cv_parser.py
------------
Extracts text from a PDF or DOCX CV, then uses Groq LLM to parse it into
a structured dictionary for the JobScout AI agent.
"""

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from llm_provider import chat_with_fallback

# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to return strict JSON only
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert CV / resume parser.

Given the raw text of a CV, extract the following information and return it as
a single, valid JSON object — no markdown, no code fences, no extra text.

JSON schema:
{
  "name":             "<full name of the candidate>",
  "skills":           ["<skill1>", "<skill2>", ...],
  "job_titles":       ["<role1>", "<role2>", ...],
  "experience_years": <integer or float>,
  "location":         "<city, country or Remote>",
  "education":        "<highest degree and institution>",
  "summary":          "<exactly 2-sentence professional summary of the candidate>"
}

Rules:
- skills: include technical tools, languages, frameworks, and soft skills.
- job_titles: include both past roles and the role they appear to be targeting.
- experience_years: best estimate as a number (e.g. 3, 5.5). Use 0 if unknown.
- location: if not found, use "Not specified".
- education: if multiple degrees, list the highest one.
- Return ONLY the JSON object, nothing else.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cv(file_path: str) -> dict:
    """
    Parse a CV file (PDF or DOCX) and return structured candidate data.

    Args:
        file_path: Path to the uploaded CV file.

    Returns:
        dict with keys: name, skills, job_titles, experience_years,
                        location, education, summary.
        On failure: {"error": "<reason>"}
    """
    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        # ---- Extract raw text ----------------------------------------------
        if ext == ".pdf":
            raw_text = _extract_pdf(path)
        elif ext == ".docx":
            raw_text = _extract_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only PDF and DOCX accepted.")

        if not raw_text.strip():
            raise ValueError("CV file appears to be empty or unreadable.")

        # ---- Send to Groq LLM ----------------------------------------------
        parsed = _call_groq(raw_text)
        return _normalize_parsed_cv(parsed)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Private helpers — text extraction
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    """Extract plain text from a PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


def _extract_docx(path: Path) -> str:
    """Extract plain text from a DOCX using python-docx."""
    from docx import Document

    doc = Document(str(path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Private helper — LLM call
# ---------------------------------------------------------------------------

def _call_groq(cv_text: str) -> dict:
    """
    Send CV text to Groq LLM and parse the JSON response.

    Args:
        cv_text: Raw extracted text from the CV.

    Returns:
        Parsed dict matching the schema defined in SYSTEM_PROMPT.

    Raises:
        ValueError: If the LLM response cannot be parsed as JSON.
    """
    # Truncate very long CVs to avoid token limits (~12k chars ≈ ~3k tokens)
    truncated_text = cv_text[:12000]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Here is the CV text:\n\n{truncated_text}"),
    ]

    raw_content = chat_with_fallback(messages)
    parsed = _extract_first_json_object(raw_content.strip())
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned JSON but not an object.")
    return parsed


def _extract_first_json_object(text: str) -> dict:
    """
    Extract the first JSON object from model output.
    The prompt requests JSON-only, but this makes parsing resilient.
    """
    cleaned = text.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    # Fast path: direct JSON
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: find first {...} block and parse
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("LLM did not return a JSON object.")

    candidate = cleaned[first : last + 1].strip()
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {exc}")


def _normalize_parsed_cv(parsed: dict) -> dict:
    """
    Ensure required keys exist and types are sane.
    """
    defaults = {
        "name": "Unknown",
        "skills": [],
        "job_titles": [],
        "experience_years": 0,
        "location": "Not specified",
        "education": "Not specified",
        "summary": "",
    }

    out = dict(defaults)
    out.update(parsed or {})

    out["name"] = str(out.get("name") or defaults["name"]).strip() or defaults["name"]
    out["location"] = str(out.get("location") or defaults["location"]).strip() or defaults["location"]
    out["education"] = str(out.get("education") or defaults["education"]).strip() or defaults["education"]
    out["summary"] = str(out.get("summary") or defaults["summary"]).strip()

    skills = out.get("skills")
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    if not isinstance(skills, list):
        skills = []
    out["skills"] = [str(s).strip() for s in skills if str(s).strip()]

    titles = out.get("job_titles")
    if isinstance(titles, str):
        titles = [t.strip() for t in titles.split(",") if t.strip()]
    if not isinstance(titles, list):
        titles = []
    out["job_titles"] = [str(t).strip() for t in titles if str(t).strip()]

    exp = out.get("experience_years", 0)
    try:
        out["experience_years"] = float(exp)
        if out["experience_years"] < 0:
            out["experience_years"] = 0
    except Exception:
        out["experience_years"] = 0

    return out
