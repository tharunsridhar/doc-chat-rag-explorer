from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import llm
import rag_pipeline as rag
import store

app = FastAPI(
    title="Doc Chat RAG Explorer",
    description="A RAG debugger: chunk, embed, retrieve, generate, and see inside every stage.",
)

Strategy = Literal["dense", "mmr", "bm25", "hybrid"]
Embedder = Literal["minilm", "bge-small", "paraphrase-l3"]
Experiment = Literal["no_context", "distractor", "shuffle_middle"]


# ---------- SECTION 1: REQUEST MODELS ----------
class ChunkParams(BaseModel):
    chunk_size: int = Field(store.DEFAULT_CHUNK_SIZE, ge=50, le=4000)
    chunk_overlap: int = Field(store.DEFAULT_CHUNK_OVERLAP, ge=0, le=2000)


class ChunkPreviewRequest(ChunkParams):
    text: str


class TextBuildRequest(ChunkParams):
    text: str
    embedder: Embedder = "minilm"


class RebuildRequest(ChunkParams):
    session_id: str
    embedder: Embedder = "minilm"


class QuestionRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1)
    k: int = Field(4, ge=1, le=20)
    strategy: Strategy = "dense"


class ExperimentRequest(QuestionRequest):
    mode: Experiment
    distractor: str | None = None


class DiagnoseRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    retrieved_ids: list[int]


class SideConfig(ChunkParams):
    embedder: Embedder = "minilm"
    k: int = Field(4, ge=1, le=20)
    strategy: Strategy = "dense"


class CompareRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1)
    a: SideConfig
    b: SideConfig
    skip_llm: bool = False


# ---------- SECTION 2: ERROR MAPPING ----------
@app.exception_handler(KeyError)
async def unknown_session(_: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": "Unknown session_id. Build an index first."})


@app.exception_handler(ValueError)
async def bad_value(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(llm.LLMUnavailable)
async def llm_unavailable(_: Request, exc: llm.LLMUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(llm.LLMError)
async def llm_error(_: Request, exc: llm.LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# ---------- SECTION 3: CONFIG + CHUNKING LAB ----------
@app.get("/api/config")
def get_config():
    return rag.app_config()


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": store.session_count(), "llm_configured": llm.is_configured()}


@app.post("/api/chunk-preview")
def chunk_preview(req: ChunkPreviewRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    return rag.chunk_preview(req.text, req.chunk_size, req.chunk_overlap)


# ---------- SECTION 4: BUILD ROUTES (chunk -> embed -> store) ----------
@app.post("/api/build-from-text")
def build_from_text(req: TextBuildRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    return rag.build(req.text, req.chunk_size, req.chunk_overlap, req.embedder)


@app.post("/api/build-from-file")
async def build_from_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(store.DEFAULT_CHUNK_SIZE, ge=50, le=4000),
    chunk_overlap: int = Form(store.DEFAULT_CHUNK_OVERLAP, ge=0, le=2000),
    embedder: Embedder = Form("minilm"),
):
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Only plain-text (.txt/.md) files are supported.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")
    return rag.build(text, chunk_size, chunk_overlap, embedder)


@app.post("/api/rebuild")
def rebuild(req: RebuildRequest):
    """Re-index the session's stored text with new chunking / embedding settings."""
    store.get_session(req.session_id)
    return rag.build("", req.chunk_size, req.chunk_overlap, req.embedder, session_id=req.session_id)


# ---------- SECTION 5: RETRIEVE / ASK / DIAGNOSE ----------
@app.post("/api/retrieve")
def retrieve_route(req: QuestionRequest):
    return rag.retrieve(req.session_id, req.question, req.k, req.strategy)


@app.post("/api/ask")
def ask_route(req: QuestionRequest):
    return rag.ask(req.session_id, req.question, req.k, req.strategy)


@app.post("/api/diagnose")
def diagnose_route(req: DiagnoseRequest):
    return rag.diagnose(req.session_id, req.question, req.answer, req.retrieved_ids)


# ---------- SECTION 6: DEBUGGING TOOLS ----------
@app.post("/api/embedding-space")
def embedding_space_route(req: QuestionRequest):
    return rag.embedding_space(req.session_id, req.question, req.k, req.strategy)


@app.post("/api/experiment")
def experiment_route(req: ExperimentRequest):
    return rag.experiment(req.session_id, req.question, req.k, req.strategy, req.mode, req.distractor)


@app.post("/api/compare")
def compare_route(req: CompareRequest):
    return rag.compare(req.session_id, req.question, req.a.model_dump(), req.b.model_dump(), req.skip_llm)


# ---------- SECTION 7: FRONTEND (served at "/") ----------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
