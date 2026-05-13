"""
agent.py
--------
AI agent that takes structured CV data and returns a ranked list of job results.

TODO: Implement run_agent() using LangChain / LangGraph + DuckDuckGo search tool:
  1. Build a prompt from cv_data (skills, experience, title).
  2. Use DuckDuckGo search to find relevant jobs.
  3. Use Groq LLM (GROQ_API_KEY from .env) to rank and score results.
  4. Return a list of ranked job dicts.
"""

import os
import json
import re
import httpx
import traceback
from typing import Any

from dotenv import load_dotenv
from llm_provider import chat_with_fallback

load_dotenv()




def run_agent(cv_data: dict) -> list[dict]:
    """
    Run the AI job-search agent against the parsed CV data.

    Args:
        cv_data: Structured CV dict produced by cv_parser.parse_cv().
                 Expected keys: skills, job_titles, experience_years, location.

    Returns:
        A list of ranked job dicts, each containing:
        {
            "title":       str,
            "company":     str,
            "location":    str,
            "url":         str,
            "score":       float,   # 0 – 100 match score
            "description": str,
        }
    """

    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()

    skills = cv_data.get("skills") or []
    job_titles = cv_data.get("job_titles") or []
    experience_years = cv_data.get("experience_years")
    location = cv_data.get("location") or ""

    queries = _generate_queries(
        skills=skills,
        job_titles=job_titles,
        experience_years=experience_years,
        location=location,
    )

    raw_results: list[dict[str, Any]] = []
    for q in queries:
        try:
            results = search_tool.run(q)
        except Exception:
            traceback.print_exc()
            results = None

        raw_results.extend(_coerce_search_results(results))

    seen_urls: set[str] = set()
    scored_jobs: list[dict[str, Any]] = []

    for r in raw_results:
        url = (r.get("url") or "").strip()
        if not url or not url.startswith("http"):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = (r.get("title") or "").strip() or "Unknown"
        snippet = (r.get("content") or r.get("snippet") or "").strip()

        job_eval = _score_job(
            cv_data={
                "skills": skills,
                "job_titles": job_titles,
                "experience_years": experience_years,
                "location": location,
            },
            job_data={
                "title": title,
                "url": url,
                "snippet": snippet[:2500],
                "source": r.get("source") or "tavily",
            },
        )

        scored_jobs.append(
            {
                "title": job_eval.get("title") or title,
                "company": job_eval.get("company") or "Unknown",
                "location": job_eval.get("location") or "Not specified",
                "url": url,
                "score": float(job_eval.get("score", 0)),
                "description": job_eval.get("reasoning") or "",
            }
        )

    scored_jobs.sort(key=lambda j: float(j.get("score", 0)), reverse=True)
    top = scored_jobs[:10]

    return [
        {
            "title": j["title"],
            "company": j["company"],
            "location": j["location"],
            "url": j["url"],
            "score": float(j["score"]),
            "description": j.get("description") or None,
        }
        for j in top
    ]


def _generate_queries(
    *,
    skills: list[Any],
    job_titles: list[Any],
    experience_years: Any,
    location: str,
) -> list[str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    skills_s = ", ".join(str(s) for s in skills[:20] if str(s).strip())
    titles_s = ", ".join(str(t) for t in job_titles[:8] if str(t).strip())
    exp_s = "" if experience_years is None else str(experience_years)
    location_s = (location or "").strip()

    system = SystemMessage(
        content=(
            "You generate job search queries for real job listings.\n"
            "Return ONLY a JSON array of exactly 3 strings. No extra text.\n"
            "Each query should be something someone would paste into Google.\n"
            "Include role keywords, key skills, and location/remote if appropriate."
        )
    )

    user = HumanMessage(
        content=(
            "Create 3 job search queries based on this CV data:\n"
            f"- job_titles: {titles_s or 'Not specified'}\n"
            f"- skills: {skills_s or 'Not specified'}\n"
            f"- experience_years: {exp_s or 'Not specified'}\n"
            f"- location: {location_s or 'Not specified'}\n"
            "\nConstraints:\n"
            "- Prefer queries that surface real listings (include 'jobs', 'hiring', 'apply').\n"
            "- Make the 3 queries meaningfully different."
        )
    )

    content = chat_with_fallback([system, user])
    arr = _parse_json_from_text(content)
    if not isinstance(arr, list):
        raise RuntimeError("Query generation failed (LLM did not return a JSON array).")

    queries = [str(x).strip() for x in arr if str(x).strip()]
    queries = queries[:3]
    while len(queries) < 3:
        queries.append(queries[-1] if queries else "software engineer jobs remote")
    return queries


def _score_job(*, cv_data: dict[str, Any], job_data: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(
        content=(
            "You are a strict job matching evaluator.\n"
            "Return ONLY a single JSON object with exactly these keys:\n"
            '{ "score": number, "reasoning": string, "company": string, "location": string, "title": string }\n'
            "Rules:\n"
            "- score is 0-100.\n"
            "- reasoning is 2-3 sentences.\n"
            "- company/location/title: extract from provided snippet if possible; otherwise use 'Unknown'/'Not specified'.\n"
            "- Output must be valid JSON only (no markdown, no extra text)."
        )
    )

    user = HumanMessage(
        content=(
            "CV data:\n"
            f"{json.dumps(cv_data, ensure_ascii=False)}\n\n"
            "Job listing data:\n"
            f"{json.dumps(job_data, ensure_ascii=False)}"
        )
    )

    content = chat_with_fallback([system, user])
    obj = _parse_json_from_text(content)
    if not isinstance(obj, dict):
        raise RuntimeError("Scoring failed (LLM did not return a JSON object).")

    score = obj.get("score", 0)
    try:
        score_f = float(score)
    except Exception:
        score_f = 0.0
    score_f = max(0.0, min(100.0, score_f))

    reasoning = str(obj.get("reasoning") or "").strip()
    company = str(obj.get("company") or "Unknown").strip() or "Unknown"
    location = str(obj.get("location") or "Not specified").strip() or "Not specified"
    title = str(obj.get("title") or job_data.get("title") or "Unknown").strip() or "Unknown"

    return {
        "score": score_f,
        "reasoning": reasoning,
        "company": company,
        "location": location,
        "title": title,
    }


def _coerce_search_results(results: Any) -> list[dict[str, Any]]:
    if results is None:
        return []

    if isinstance(results, str):
        try:
            results = json.loads(results)
        except Exception:
            return []

    if isinstance(results, dict):
        if isinstance(results.get("results"), list):
            return [r for r in results["results"] if isinstance(r, dict)]
        if isinstance(results.get("data"), list):
            return [r for r in results["data"] if isinstance(r, dict)]
        if "url" in results:
            return [results]  # type: ignore[list-item]
        return []

    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]

    return []


def _parse_json_from_text(text: str) -> Any:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        start = start_arr
        end = cleaned.rfind("]")
    else:
        start = start_obj
        end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON found in LLM output.")

    candidate = cleaned[start : end + 1].strip()
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    return json.loads(candidate)
