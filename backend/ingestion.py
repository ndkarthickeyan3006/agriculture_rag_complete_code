from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import hashlib
import json
import re

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import HashingVectorizer

SOURCE_URL = (
    "https://agritech.tnau.ac.in/"
    "agricultural_engineering/agriengg_govt_schemes.html"
)

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "data" / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
INFO_FILE = INDEX_DIR / "index_info.json"

VECTOR_DIMENSION = 384
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35

# Offline lexical vectorizer. No Hugging Face/model download is needed.
# The same vectorizer settings MUST be used for ingestion and queries.
VECTORIZER = HashingVectorizer(
    n_features=VECTOR_DIMENSION,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    stop_words="english",
)

def fetch_webpage(url: str = SOURCE_URL) -> dict[str, Any]:
    """Fetch a webpage and extract readable headings, paragraphs, lists and tables."""
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; TNAU-RAG/1.0)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    page_title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "TNAU Agricultural Engineering Government Schemes"
    )

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find(id=re.compile("content|main", re.I))
    if main is None:
        main = soup.body or soup

    extracted: list[str] = []
    for element in main.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "tr", "td", "th"]
    ):
        value = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if value and value not in extracted:
            extracted.append(value)

    text = "\n".join(extracted)

    # Fallback for older pages with unusual HTML markup.
    if len(text) < 300:
        text = main.get_text("\n", strip=True)
        text = "\n".join(
            re.sub(r"\s+", " ", line).strip()
            for line in text.splitlines()
            if line.strip()
        )

    if len(text) < 300:
        raise ValueError(
            "The page returned too little readable text. "
            "The website may have changed or blocked automated access."
        )

    return {
        "title": page_title,
        "url": url,
        "text": text,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def split_into_chunks(text: str) -> list[str]:
    """Split extracted webpage text into overlapping word-based chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def load_faiss_index():
    """Load the persistent FAISS index and matching metadata."""
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

        if index.ntotal != len(metadata):
            raise RuntimeError(
                "FAISS vector count does not match metadata count. "
                "Back up data/faiss_index and rebuild the index."
            )
        return index, metadata

    return None, []


def save_faiss_index(index, metadata: list[dict[str, Any]]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    INFO_FILE.write_text(
        json.dumps(
            {
                "vector_method": "scikit-learn HashingVectorizer",
                "vector_dimension": VECTOR_DIMENSION,
                "source_url": SOURCE_URL,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "note": "Lexical vector retrieval; not a neural semantic embedding model.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def ingest_url(url: str = SOURCE_URL) -> dict[str, Any]:
    """Fetch URL, extract text, chunk, vectorize and append unique chunks to FAISS."""
    page = fetch_webpage(url)
    chunks = split_into_chunks(page["text"])

    if not chunks:
        raise ValueError("No text chunks were produced from the webpage.")

    matrix = VECTORIZER.transform(chunks).toarray().astype("float32")
    index, metadata = load_faiss_index()

    if index is None:
        index = faiss.IndexFlatIP(VECTOR_DIMENSION)
    elif index.d != VECTOR_DIMENSION:
        raise ValueError("Existing FAISS index has an incompatible vector dimension.")

    existing_hashes = {item.get("content_hash") for item in metadata}
    new_vectors: list[np.ndarray] = []
    new_items: list[dict[str, Any]] = []

    for chunk_number, chunk in enumerate(chunks, start=1):
        digest = hashlib.sha256(
            f"{url}|{chunk}".encode("utf-8")
        ).hexdigest()

        if digest in existing_hashes:
            continue

        new_vectors.append(matrix[chunk_number - 1])
        new_items.append(
            {
                "text": chunk,
                "source": page["title"],
                "url": url,
                "chunk": chunk_number,
                "fetched_at": page["fetched_at"],
                "content_hash": digest,
            }
        )

    if new_vectors:
        index.add(np.vstack(new_vectors).astype("float32"))
        metadata.extend(new_items)
        save_faiss_index(index, metadata)

    return {
        "title": page["title"],
        "url": url,
        "characters_extracted": len(page["text"]),
        "chunks_detected": len(chunks),
        "chunks_added": len(new_items),
        "total_chunks": len(metadata),
        "fetched_at": page["fetched_at"],
        "vector_method": "Offline HashingVectorizer",
    }


def search_index(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Vectorize the question and return the nearest matching chunks."""
    index, metadata = load_faiss_index()

    if index is None or index.ntotal == 0:
        return []

    query_vector = VECTORIZER.transform([question]).toarray().astype("float32")
    scores, ids = index.search(query_vector, min(top_k, index.ntotal))

    results: list[dict[str, Any]] = []
    for score, item_id in zip(scores[0], ids[0]):
        if item_id < 0:
            continue
        result = dict(metadata[int(item_id)])
        result["score"] = float(score)
        results.append(result)

    return results


def list_sources() -> list[dict[str, Any]]:
    """Return one summary per indexed source URL."""
    _, metadata = load_faiss_index()
    grouped: dict[str, dict[str, Any]] = {}

    for item in metadata:
        url = item.get("url", "")
        if url not in grouped:
            grouped[url] = {
                "title": item.get("source", "Unknown source"),
                "url": url,
                "chunks": 0,
                "last_fetched": item.get("fetched_at"),
            }
        grouped[url]["chunks"] += 1

    return list(grouped.values())


def get_index_stats() -> dict[str, Any]:
    index, metadata = load_faiss_index()
    return {
        "faiss_ready": index is not None,
        "vector_count": int(index.ntotal) if index is not None else 0,
        "metadata_count": len(metadata),
        "index_directory": str(INDEX_DIR),
    }
