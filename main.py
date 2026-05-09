import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm_provider import chat_with_fallback

# Import CV parser and agent — implementations live in their respective files
from cv_parser import parse_cv      # TODO: implement parse_cv(file_path: str) -> dict
from agent import run_agent         # TODO: implement run_agent(cv_data: dict) -> list[dict]

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()  # Load variables from .env (GROQ_API_KEY, TAVILY_API_KEY, etc.)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory session store  { session_id: { "file_path": str, "cv_data": dict | None } }
sessions: dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="JobScout AI",
    description="AI-powered job search backend — upload a CV and get ranked job results.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (Next.js frontend on any port)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class UploadCVResponse(BaseModel):
    session_id: str
    filename: str
    message: str


class SearchJobsRequest(BaseModel):
    session_id: str


class JobResult(BaseModel):
    title: str
    company: str
    location: str
    url: str
    score: float
    description: str | None = None


class SearchJobsResponse(BaseModel):
    session_id: str
    jobs: list[JobResult]
    total: int


class ChatHistoryItem(BaseModel):
    role: str  # "user" | "ai" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    chat_history: list[ChatHistoryItem] | None = None


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
async def health_check():
    """Simple liveness probe used by the frontend and deployment platforms."""
    return {"status": "ok"}


@app.post("/upload-cv", response_model=UploadCVResponse, tags=["CV"])
async def upload_cv(file: UploadFile = File(...)):
    """
    Accept a PDF or DOCX CV upload.

    - Validates the file extension.
    - Saves the file to `uploads/{session_id}.{ext}`.
    - Creates an in-memory session keyed by the generated UUID.
    - Returns the `session_id` for use in subsequent requests.
    """
    # ---- Validate extension ------------------------------------------------
    original_name = file.filename or ""
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Only PDF and DOCX are accepted.",
        )

    # ---- Generate session & persist file -----------------------------------
    session_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{session_id}{ext}"

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {exc}",
        )
    finally:
        await file.close()

    # ---- Store session data ------------------------------------------------
    sessions[session_id] = {
        "file_path": str(save_path),
        "original_filename": original_name,
        "cv_data": None,   # Populated lazily on /search-jobs
        "jobs": None,
    }

    return UploadCVResponse(
        session_id=session_id,
        filename=original_name,
        message="CV uploaded successfully. Use the session_id to search for jobs.",
    )


@app.post("/search-jobs", response_model=SearchJobsResponse, tags=["Jobs"])
async def search_jobs(body: SearchJobsRequest):
    """
    Trigger the AI agent pipeline for a previously uploaded CV.

    1. Looks up the session by `session_id`.
    2. Parses the CV if not already parsed.
    3. Runs the AI agent to fetch and rank job listings.
    4. Caches results in the session for repeat calls.
    """
    session_id = body.session_id

    # ---- Validate session --------------------------------------------------
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Please upload a CV first.",
        )

    session = sessions[session_id]

    # ---- Return cached results if available --------------------------------
    if session.get("jobs") is not None:
        return SearchJobsResponse(
            session_id=session_id,
            jobs=session["jobs"],
            total=len(session["jobs"]),
        )

    # ---- Parse CV ----------------------------------------------------------
    if session["cv_data"] is None:
        try:
            # TODO: parse_cv reads the file at file_path and returns structured data
            # Expected return shape:
            # {
            #   "name": str,
            #   "email": str,
            #   "skills": list[str],
            #   "experience": list[dict],
            #   "education": list[dict],
            #   "summary": str,
            # }
            cv_data = parse_cv(session["file_path"])
            session["cv_data"] = cv_data
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"CV parsing failed: {exc}",
            )

    # ---- Run AI agent ------------------------------------------------------
    try:
        # TODO: run_agent takes cv_data dict and returns a ranked list of job dicts.
        # Expected return shape (list of):
        # {
        #   "title": str,
        #   "company": str,
        #   "location": str,
        #   "url": str,
        #   "score": float,        # 0.0 – 1.0 relevance score
        #   "description": str,
        # }
        raw_jobs = run_agent(session["cv_data"])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {exc}",
        )

    # ---- Validate & cache --------------------------------------------------
    jobs = [JobResult(**job) for job in raw_jobs]
    session["jobs"] = jobs

    return SearchJobsResponse(
        session_id=session_id,
        jobs=jobs,
        total=len(jobs),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(body: ChatRequest):
    """
    Chat with the AI using the uploaded CV + matched job results as context.
    The frontend may optionally provide `chat_history` to preserve conversation context.
    """
    session_id = body.session_id

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Please upload a CV first.",
        )

    session = sessions[session_id]
    cv_data = session.get("cv_data") or {}
    jobs = session.get("jobs") or []

    # Normalize jobs into plain dicts (they may be Pydantic models)
    normalized_jobs: list[dict] = []
    for j in jobs:
        if hasattr(j, "model_dump"):
            normalized_jobs.append(j.model_dump())
        elif isinstance(j, dict):
            normalized_jobs.append(j)

    # Keep the prompt compact to avoid token bloat
    cv_context = {
        "skills": cv_data.get("skills", []),
        "job_titles": cv_data.get("job_titles", []),
        "experience_years": cv_data.get("experience_years", 0),
        "location": cv_data.get("location", "Not specified"),
        "summary": cv_data.get("summary", ""),
        "education": cv_data.get("education", ""),
    }
    jobs_context = [
        {
            "title": j.get("title"),
            "company": j.get("company"),
            "location": j.get("location"),
            "url": j.get("url"),
            "score": j.get("score"),
            "description": j.get("description"),
        }
        for j in normalized_jobs[:10]
    ]

    system_prompt = (
        "You are JobScout AI, a career assistant. Use the provided candidate CV data and the matched job results.\n"
        "Be specific, practical, and concise. If you make assumptions, say they're assumptions.\n"
        "When referencing jobs, cite their title + company and why.\n\n"
        "=== Candidate CV Data (structured) ===\n"
        f"{cv_context}\n\n"
        "=== Matched Jobs (top results) ===\n"
        f"{jobs_context}\n"
    )

    messages = [SystemMessage(content=system_prompt)]

    # Optional conversation history, provided by the client
    if body.chat_history:
        for item in body.chat_history[-20:]:
            role = (item.role or "").lower().strip()
            content = (item.content or "").strip()
            if not content:
                continue

            if role in {"user"}:
                messages.append(HumanMessage(content=content))
            elif role in {"ai", "assistant"}:
                messages.append(AIMessage(content=content))
            elif role in {"system"}:
                # Usually avoid letting clients override system context,
                # but we include it as a normal assistant message to preserve intent.
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=body.message))

    try:
        reply = chat_with_fallback(messages)
        if not reply:
            raise RuntimeError("Empty reply from model.")
        return ChatResponse(reply=reply)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")
