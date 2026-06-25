from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader

import trafilatura

from app.rag.vector_store import (
    CHUNKS_PATH,
    INDEX_PATH,
    QUERY_CACHE_PATH,
    GenericVectorStore,
    RAGChunk,
    _build_embeddings,
)


load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parents[2]
DOCS_ROOT = os.getenv("DOCS_ROOT_URL", "").strip()
DOCS_SOURCE_DIR = os.getenv("DOCS_SOURCE_DIR", "").strip()
DOCS_ALLOWED_DOMAINS = tuple(
    domain.strip()
    for domain in os.getenv("DOCS_ALLOWED_DOMAINS", "").split(",")
    if domain.strip()
)
DOCS_PATH_PREFIX = os.getenv("DOCS_PATH_PREFIX", "/")
DOCS_URL_SUFFIX = os.getenv("DOCS_URL_SUFFIX", "")
DOCS_CONTENT_SELECTOR = os.getenv("DOCS_CONTENT_SELECTOR", "article#content")
MAX_PAGES = int(os.getenv("DOCS_MAX_PAGES", "40"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
LOCAL_DOC_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".pdf"}


@dataclass
class PageDocument:
    title: str
    url: str
    text: str

def _parse_travel_page(html: str, url: str) -> PageDocument | None:
    # Trafilatura extrae el texto principal ignorando menús y banners publicitarios
    extracted_text = trafilatura.extract(html, include_links=False, favor_precision=True)
    
    if not extracted_text or len(extracted_text) < 100:
        return None

    # Intentar obtener el título
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find(["h1", "title"])
    title = title_tag.get_text(strip=True) if title_tag else url

    return PageDocument(title=title, url=url, text=extracted_text)

def crawl_docs_trafilatura() -> list[PageDocument]:
    if not DOCS_ROOT:
        return []

    seen: set[str] = set()
    queue: deque[str] = deque([DOCS_ROOT])
    pages: list[PageDocument] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while queue and len(seen) < MAX_PAGES:
            url = queue.popleft()
            if url in seen: continue
            seen.add(url)

            try:
                response = client.get(url)
                response.raise_for_status()
                html = response.text

                # Usar el nuevo parser de viajes
                page = _parse_travel_page(html, url)
                if page:
                    pages.append(page)

                for link in _extract_links(html, url):
                    if link not in seen:
                        queue.append(link)
            except Exception as e:
                print(f"Error parseando {url}: {e}")

    return pages

def _is_doc_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and (not DOCS_ALLOWED_DOMAINS or parsed.netloc in DOCS_ALLOWED_DOMAINS)
        and parsed.path.startswith(DOCS_PATH_PREFIX)
        and parsed.path.endswith(DOCS_URL_SUFFIX)
    )


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_page(html: str, url: str) -> PageDocument | None:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one(DOCS_CONTENT_SELECTOR)
    if article is None:
        return None

    title_tag = article.find(["h1", "title"])
    title = _clean_text(title_tag.get_text(" ", strip=True)) if title_tag else url

    for element in article.find_all(["script", "style", "nav", "aside"]):
        element.decompose()

    text = _clean_text(article.get_text("\n", strip=True))
    if len(text) < 80:
        return None

    return PageDocument(title=title, url=url, text=text)


def _parse_local_file(path: Path) -> PageDocument | None:
    if path.suffix.lower() == ".pdf":
        return _parse_pdf_file(path)

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        return _parse_page(raw_text, path.as_uri())

    text = _clean_text(raw_text)
    if len(text) < 80:
        return None

    return PageDocument(title=path.stem, url=path.as_uri(), text=text)


def _parse_pdf_file(path: Path) -> PageDocument | None:
    reader = PdfReader(str(path))
    pages_text: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages_text.append(extracted)

    text = _clean_text("\n".join(pages_text))
    if len(text) < 80:
        return None

    return PageDocument(title=path.stem, url=path.as_uri(), text=text)


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        full_url = urljoin(base_url, anchor["href"])
        if _is_doc_url(full_url):
            links.append(full_url)
    return links


def _split_document(doc: PageDocument) -> list[RAGChunk]:
    chunks: list[RAGChunk] = []
    text = doc.text
    start = 0
    position = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                RAGChunk(
                    chunk_id=f"{doc.title}-{position}",
                    title=doc.title,
                    url=doc.url,
                    text=chunk_text,
                )
            )
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        position += 1

    return chunks


def crawl_docs_beatiful() -> list[PageDocument]:
    if not DOCS_ROOT:
        return []

    seen: set[str] = set()
    queue: deque[str] = deque([DOCS_ROOT])
    pages: list[PageDocument] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while queue and len(seen) < MAX_PAGES:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            response = client.get(url)
            response.raise_for_status()
            html = response.text

            page = _parse_page(html, url)
            if page:
                pages.append(page)

            for link in _extract_links(html, url):
                if link not in seen:
                    queue.append(link)

    return pages


def _resolve_source_dir() -> Path | None:
    if not DOCS_SOURCE_DIR:
        return None

    path = Path(DOCS_SOURCE_DIR).expanduser()
    if not path.is_absolute():
        path = (PROJECT_DIR / path).resolve()
    return path


def load_local_docs() -> list[PageDocument]:
    source_dir = _resolve_source_dir()
    if source_dir is None:
        return []
    if not source_dir.exists() or not source_dir.is_dir():
        raise RuntimeError(
            f"DOCS_SOURCE_DIR={str(source_dir)!r} no existe o no es una carpeta valida."
        )

    pages: list[PageDocument] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in LOCAL_DOC_SUFFIXES:
            continue
        page = _parse_local_file(path)
        if page:
            pages.append(page)
        if len(pages) >= MAX_PAGES:
            break
    return pages


def build_index() -> str:
    for path in (INDEX_PATH, CHUNKS_PATH, QUERY_CACHE_PATH):
        if path.exists():
            path.unlink()

    if not DOCS_ROOT and not DOCS_SOURCE_DIR:
        raise RuntimeError(
            "Debes configurar DOCS_ROOT_URL, DOCS_SOURCE_DIR o ambos para construir el indice RAG."
        )

    pages = []
    if DOCS_ROOT:
        pages.extend(crawl_docs_trafilatura())
    if DOCS_SOURCE_DIR:
        pages.extend(load_local_docs())

    if not pages:
        raise RuntimeError(
            "No se pudo extraer ningun documento util para el indice RAG. "
            f"Revisa DOCS_ROOT_URL={DOCS_ROOT!r}, DOCS_SOURCE_DIR={DOCS_SOURCE_DIR!r}, "
            f"DOCS_ALLOWED_DOMAINS={DOCS_ALLOWED_DOMAINS!r}, DOCS_PATH_PREFIX={DOCS_PATH_PREFIX!r}, "
            f"DOCS_URL_SUFFIX={DOCS_URL_SUFFIX!r} y DOCS_CONTENT_SELECTOR={DOCS_CONTENT_SELECTOR!r}."
        )

    chunks: list[RAGChunk] = []
    for page in pages:
        chunks.extend(_split_document(page))

    if not chunks:
        raise RuntimeError(
            "Se encontraron paginas, pero no se generaron chunks para el indice RAG. "
            f"Verifica el selector DOCS_CONTENT_SELECTOR={DOCS_CONTENT_SELECTOR!r}, "
            f"el tamano minimo de contenido util y la estructura HTML real de las paginas."
        )

    store = GenericVectorStore.build(chunks=chunks, embeddings=_build_embeddings())
    store.save()
    return str(INDEX_PATH.parent.relative_to(CHUNKS_PATH.parents[2]))


if __name__ == "__main__":
    output_dir = build_index()
    print(f"RAG index created in {output_dir}")
