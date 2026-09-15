import os
import uuid

import chromadb
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_MODEL = "qwen/qwen3.8-27b"
llm = ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

_sessions: dict[str, dict] = {}  # session_id -> {"vectorstore": Chroma, "chunks": [str, ...]}


def has_session(session_id: str) -> bool:
    return session_id in _sessions


# ---------- SECTION 1: BUILD PHASE (chunk -> embed -> store) ----------
def build_index(text: str) -> dict:
    chunks = splitter.split_text(text)

    # a fresh in-memory chroma client per document -- nothing touches disk or the
    # other scripts' persisted chroma_db in this folder
    client = chromadb.EphemeralClient()
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, client=client)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"vectorstore": vectorstore, "chunks": chunks}
    return {"session_id": session_id, "chunk_count": len(chunks), "chunk_preview": chunks[:3]}


# ---------- SECTION 2: RETRIEVE PHASE (question -> top-k chunks + similarity scores) ----------
def retrieve(session_id: str, question: str, k: int = 4) -> list[dict]:
    vectorstore = _sessions[session_id]["vectorstore"]
    results = vectorstore.similarity_search_with_score(question, k=k)
    return [{"text": doc.page_content, "score": float(score)} for doc, score in results]


# ---------- SECTION 3: GENERATE PHASE ((context + question) -> LLM -> answer) ----------
PROMPT_TEMPLATE = """You are a helpful assistant answering questions about an uploaded document.
Answer using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""


def ask(session_id: str, question: str, k: int = 4) -> dict:
    retrieved = retrieve(session_id, question, k=k)
    context = "\n\n".join(c["text"] for c in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm.invoke(prompt).content
    return {"answer": answer, "retrieved": retrieved, "prompt_sent": prompt}
