from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_pipeline import ask, build_index, has_session, retrieve

app = FastAPI(
    title="Doc Chat RAG Explorer",
    description="Mini project: the full RAG pipeline (chunk -> embed -> store -> "
    "retrieve -> augment -> generate), backed by Chroma + Groq.",
)


# ---------- SECTION 1: REQUEST MODELS ----------
class TextBuildRequest(BaseModel):
    text: str


class QuestionRequest(BaseModel):
    session_id: str
    question: str


def _require_session(session_id: str):
    if not has_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id -- build an index first.")


# ---------- SECTION 2: BUILD ROUTES (chunk -> embed -> store) ----------
@app.post("/api/build-from-text")
def build_from_text(req: TextBuildRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    return build_index(req.text)


@app.post("/api/build-from-file")
async def build_from_file(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Only plain-text (.txt/.md) files are supported.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")
    return build_index(text)


# ---------- SECTION 3: RETRIEVE ROUTE (retrieval only, no LLM call) ----------
@app.post("/api/retrieve")
def retrieve_route(req: QuestionRequest):
    _require_session(req.session_id)
    return {"retrieved": retrieve(req.session_id, req.question)}


# ---------- SECTION 4: ASK ROUTE (full RAG: retrieve + generate) ----------
@app.post("/api/ask")
def ask_route(req: QuestionRequest):
    _require_session(req.session_id)
    return ask(req.session_id, req.question)


# ---------- SECTION 5: FRONTEND (served at "/") ----------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
