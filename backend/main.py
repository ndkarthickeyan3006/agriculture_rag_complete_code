from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from backend.ingestion import (
    SOURCE_URL,
    ingest_url,
    search_index,
    list_sources,
    get_index_stats,
)

app = FastAPI(
    title="TNAU Agricultural Schemes URL RAG API",
    version="2.0.0",
    description="Fetch TNAU scheme webpages, index them in FAISS, and retrieve passages.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    url: HttpUrl = SOURCE_URL


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def health():
    return {"status": "ok", "service": "TNAU URL RAG API", "default_url": SOURCE_URL}


@app.post("/ingest-url")
def ingest(request: IngestRequest):
    try:
        result = ingest_url(str(request.url))
        return {"message": "Website fetched and indexed", **result}
    except requests_error_types() as exc:
        raise HTTPException(status_code=502, detail=f"Website request failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


def requests_error_types():
    # Keep requests import isolated to the backend API module.
    import requests
    return (requests.RequestException,)


@app.post("/search")
def search(request: SearchRequest):
    try:
        results = search_index(request.question, request.top_k)
        return {
            "question": request.question,
            "count": len(results),
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@app.get("/sources")
def sources():
    try:
        return {"sources": list_sources()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not list sources: {exc}")


@app.get("/stats")
def stats():
    try:
        return get_index_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read index stats: {exc}")
