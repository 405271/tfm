from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
from google import genai
import numpy as np
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai


try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover - optional dependency at runtime
    OllamaEmbeddings = None


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
RAG_DIR = BASE_DIR / "data" / os.getenv("RAG_INDEX_DIR", "rag")
INDEX_PATH = RAG_DIR / "docs.index"
CHUNKS_PATH = RAG_DIR / "chunks.json"
QUERY_CACHE_PATH = RAG_DIR / "query_cache.json"


@dataclass
class RAGChunk:
    chunk_id: str
    title: str
    url: str
    text: str


class GenericVectorStore:
    def __init__(self, embeddings: Embeddings, index: faiss.Index, chunks: list[RAGChunk]) -> None:
        self.embeddings = embeddings
        self.index = index
        self.chunks = chunks
        self.query_cache = _load_query_cache()

    @classmethod
    def build(cls, chunks: list[RAGChunk], embeddings: Embeddings) -> "GenericVectorStore":
        if not chunks:
            raise ValueError("No chunks available to build the vector store.")

        vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        d = 768  # Dimensión de los embeddings, ajusta según tu modelo
        index = faiss.IndexFlatIP(d) # matrix.shape[1]
        index.add(matrix)
        return cls(embeddings=embeddings, index=index, chunks=chunks)

    @classmethod
    def load(cls) -> "GenericVectorStore | None":
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            return None

        embeddings = _build_embeddings()
        index = faiss.read_index(str(INDEX_PATH))
        raw_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        chunks = [RAGChunk(**item) for item in raw_chunks]
        return cls(embeddings=embeddings, index=index, chunks=chunks)

    def save(self) -> None:
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        CHUNKS_PATH.write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def search(self, query: str, k: int = 4) -> list[RAGChunk]:
        if not query.strip():
            return []

        embedding = self.query_cache.get(query)
        if embedding is None:
            embedding = self.embeddings.embed_query(query)
            self.query_cache[query] = embedding
            _save_query_cache(self.query_cache)

        vector = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vector)
        print(vector.shape)
        print(self.index.d)

        # Añade esto justo ANTES de: distances, indices = self.index.search(vector, k)
        print("--- DEBUG FAISS ---")
        print(f"Dimensión esperada por FAISS (self.index.d): {self.index.d}")
        print(f"Dimensión de tu vector de búsqueda (vector.shape): {vector.shape}")
        print("-------------------")

        distances, indices = self.index.search(vector, k)

        results: list[RAGChunk] = []
        seen_urls: set[str] = set()
        for idx, score in zip(indices[0], distances[0], strict=False):
            if idx < 0 or score <= 0:
                continue
            chunk = self.chunks[idx]
            if chunk.url in seen_urls:
                continue
            seen_urls.add(chunk.url)
            results.append(chunk)
        return results


def _build_embeddings() -> Embeddings:
    provider = os.getenv("EMBEDDINGS_PROVIDER", "").strip().lower()
    if not provider:
        provider = "ollama" if os.getenv("LLM_PROVIDER", "openai").strip().lower() == "ollama" else "openai"

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        return OpenAIEmbeddings(
            api_key=api_key,
            model=os.getenv("EMBEDDINGS_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.getenv("EMBEDDINGS_BASE_URL") or None,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print("Modelos disponibles:")
        for model in client.models.list():
            # Filtramos por nombre para mostrar los de embedding
            if "embed" in model.name.lower():
                print(f"- {model.name}")

        return GoogleGenerativeAIEmbeddings(
            api_key=api_key,
            model=os.getenv("EMBEDDINGS_MODEL") or os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.getenv("EMBEDDINGS_BASE_URL") or None,
        )
    
    print("***provider")
    print(provider)
    if provider == "ollama":
        if OllamaEmbeddings is None:
            raise RuntimeError(
                "Ollama embeddings require the 'langchain-ollama' package to be installed."
            )

        print("***")
        print(os.getenv("EMBEDDINGS_MODEL"))
        print(os.getenv("EMBEDDINGS_BASE_URL"))
        
        return OllamaEmbeddings(
            model=os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text"),
            base_url=os.getenv("EMBEDDINGS_BASE_URL") or os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434"),
            
        )

    if provider == "deepseek":
        raise RuntimeError(
            "DeepSeek embeddings are not configured in this project. "
            "Use EMBEDDINGS_PROVIDER=openai or EMBEDDINGS_PROVIDER=ollama."
        )

    raise RuntimeError(
        "Unsupported EMBEDDINGS_PROVIDER. Use one of: openai, ollama."
    )


def _load_query_cache() -> dict[str, list[float]]:
    if not QUERY_CACHE_PATH.exists():
        return {}
    return json.loads(QUERY_CACHE_PATH.read_text(encoding="utf-8"))


def _save_query_cache(cache: dict[str, list[float]]) -> None:
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
