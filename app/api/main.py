from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from app.rag.vector_store import GenericVectorStore

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatOllama = None

try:
    from memori import Memori
except ImportError:  # pragma: no cover - optional dependency at runtime
    Memori = None


load_dotenv()

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
FAQ_PATH = Path(os.getenv("FAQ_PATH", str(BASE_DIR / "data" / "faq.json")))
GLOSSARY_PATH = Path(os.getenv("GLOSSARY_PATH", str(BASE_DIR / "data" / "glossary.json")))
SESSION_STORE: dict[str, list["ChatMessage"]] = {}
VECTOR_STORE: GenericVectorStore | None = None
MEMORI_SESSION_FACTORY = None
MEMORI_STORAGE_READY = False
# Adaptador del núcleo Rust de Memori compartido entre peticiones. Cada instancia de
# Memori crea el suyo, y cada adaptador carga su propio EngineHandle (modelo ONNX de
# embeddings) en la primera recuperación: con una instancia por petición la memoria del
# proceso crecía sin límite (~7 MB/s en la evaluación, hasta el OOM killer). Sus callbacks
# solo usan la conexión a la BD, que es la misma para todas las peticiones, y la entidad
# viaja en cada consulta, así que se puede reutilizar.
_SHARED_RUST_CORE: object | None = None
PRODUCT_NAME = os.getenv("PRODUCT_NAME", "tu producto")
SUPPORT_CONTACT = os.getenv(
    "SUPPORT_CONTACT",
    "soporte@example.com | https://example.com/soporte",
)
GLOSSARY_TITLE = os.getenv("GLOSSARY_TITLE", "Glosario de Terminos")
GLOSSARY_URL = os.getenv(
    "GLOSSARY_URL",
    "https://example.com/docs/glosario.html",
)
ISSUE_CONTEXT_REQUEST = os.getenv(
    "ISSUE_CONTEXT_REQUEST",
    "Para ayudarte necesito mas contexto del caso: describe que intentabas hacer, que ocurrio, "
    "cual fue el mensaje o comportamiento observado y que resultado esperabas. "
    "Si aplica, comparte tambien el identificador de la reserva, solicitud, transaccion o registro relacionado.",
)
ESCALATION_REQUEST = os.getenv(
    "ESCALATION_REQUEST",
    "Puedo dejar este caso listo para revision humana. "
    "Comparte por favor el contexto del caso, la accion realizada, el mensaje o comportamiento observado, "
    "tu usuario o correo de contacto y, si aplica, el identificador de la reserva, solicitud, transaccion o registro relacionado.",
)
ESCALATION_KEYWORDS = {
    "demanda",
    "fraude",
    "reclamo",
    "supervisor",
    "abogado",
    "cancelacion",
    "cancelación",
    "molesto",
    "enojado",
    "queja",
    "incidente",
    "caido",
    "caído",
    "no funciona",
}


def _load_json_entries(path: Path, required_keys: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        return []

    raw_content = path.read_text(encoding="utf-8").strip()
    if not raw_content:
        return []

    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    entries: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if not all(isinstance(item.get(key), str) and item.get(key, "").strip() for key in required_keys):
            continue
        entries.append({key: item[key].strip() for key in required_keys})
    return entries


def _load_faq() -> list[dict[str, str]]:
    return _load_json_entries(FAQ_PATH, ("question", "answer"))


def _load_glossary() -> list[dict[str, str]]:
    return _load_json_entries(GLOSSARY_PATH, ("term", "definition"))


FAQ_ENTRIES = _load_faq()
GLOSSARY_ENTRIES = _load_glossary()


def _tokenize(text: str) -> set[str]:
    return {word.strip(".,:;!?()[]{}").lower() for word in text.split() if word.strip()}


def _entry_score(message: str, entry: dict[str, str]) -> int:
    words = _tokenize(message)
    question_words = _tokenize(entry["question"])
    answer_words = _tokenize(entry["answer"])
    return (2 * len(words & question_words)) + len(words & answer_words)


def _llm_temperature() -> float | None:
    """Temperatura configurada; None si LLM_TEMPERATURE está vacía.

    Algunos modelos (p. ej. los de razonamiento) solo aceptan la temperatura
    por defecto y rechazan cualquier otro valor; con LLM_TEMPERATURE= vacía
    no se envía el parámetro y el proveedor usa su valor por defecto.
    """
    raw = os.getenv("LLM_TEMPERATURE", "0.2").strip()
    return float(raw) if raw else None


def _llm_reasoning_effort() -> str | None:
    """Esfuerzo de razonamiento (none, low, medium, high…); None si no se indica.

    Solo se envía cuando LLM_REASONING_EFFORT tiene valor, porque los modelos
    sin razonamiento rechazan el parámetro. Con 'none', algunos modelos de
    razonamiento vuelven a aceptar una temperatura distinta de la por defecto.
    """
    raw = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
    return raw or None


def _build_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    temperature = _llm_temperature()
    reasoning_effort = _llm_reasoning_effort()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL") or None,
        )

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing DEEPSEEK_API_KEY environment variable."
            )

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        )

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError(
                "Ollama support requires the 'langchain-ollama' package to be installed."
            )

        return ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3.1"),
            temperature=temperature,
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434"),
        )

    raise RuntimeError(
        "Unsupported LLM_PROVIDER. Use one of: openai, deepseek, ollama."
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _memori_storage_url() -> str:
    configured = os.getenv("MEMORI_STORAGE_URL")
    if configured:
        return configured

    storage_path = BASE_DIR / "data" / "memori.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_path}"


def _get_memori_session_factory():
    global MEMORI_SESSION_FACTORY
    if MEMORI_SESSION_FACTORY is None:
        url = _memori_storage_url()
        # Cada instancia de Memori retiene su conexión mientras dura la petición (incluida la
        # generación). Con el QueuePool por defecto (5 + 10) la concurrencia > 15 agota el pool;
        # en SQLite abrir una conexión es barato, así que NullPool elimina ese límite artificial.
        # MEMORI_DB_POOL=queue restaura el comportamiento por defecto (para medirlo en H4).
        if url.startswith("sqlite") and os.getenv("MEMORI_DB_POOL", "null").strip().lower() != "queue":
            engine = create_engine(url, poolclass=NullPool)
        else:
            engine = create_engine(url)
        MEMORI_SESSION_FACTORY = sessionmaker(bind=engine)
    return MEMORI_SESSION_FACTORY


_FACT_SCOPES = {
    # Soporte de agencia de viajes (dataset propio). El texto no debe cambiar: forma parte
    # de la clave de la caché de extracción de las corridas ya hechas.
    "support": (
        "Extract ONLY specific facts about THIS user from the support conversation below.\n"
        "Include: name, company, job title, reference or booking numbers, reported problems, assigned contact persons, "
        "travel destination, travel dates, budget, number of travellers, preferences, restrictions "
        "(diet, mobility, health), and the user's own requests and events (bookings made, incidents reported, "
        "changes requested), with their dates when given.\n"
        "If the user corrects or updates a previously given datum, write the NEW value and say it replaces the old one.\n"
        "Do NOT include general questions the user asked about policies, procedures or services "
        "(e.g. how invoices work, what documents are needed): they are not facts about the user.\n"
        "Do NOT include generic product info, policies, or agent instructions.\n"
    ),
    # Charla general (LongMemEval, perfil neutral).
    "neutral": (
        "Extract ONLY specific facts about THIS user from the conversation below.\n"
        "Include personal facts, preferences, plans, possessions, relationships, activities, events and decisions "
        "of the user, with their dates when given (a session date may appear in brackets at the start of the user "
        "message), and concrete information the assistant gave that the user asked for or accepted "
        "(recommendations, answers, names, numbers).\n"
        "If the user corrects or updates a previously given datum, write the NEW value and say it replaces the old one.\n"
        "Do NOT include generic knowledge unrelated to this user.\n"
    ),
}

_FACT_EXTRACT_PROMPT = (
    "{scope}"
    "{style}"
    "Keep names, places, amounts, dates and codes exactly as the user wrote them.\n"
    'Return ONLY JSON with a string array: {{"facts": ["...", "..."]}}\n'
    'Return {{"facts": []}} if no user-specific facts are mentioned.\n\n'
    "User: {user}\nAssistant: {assistant}"
)

# Idioma de los hechos. Por defecto español: las consultas están en español y el modelo de
# embeddings local de Memori (all-MiniLM-L6-v2) apenas relaciona consultas en español con
# hechos en inglés (similitud ~0.0-0.14 frente a ~0.4-0.7 en el mismo idioma), por lo que
# con hechos en inglés casi nada superaba el umbral de relevancia. MEMORI_FACT_LANGUAGE=en
# conserva el comportamiento anterior como ablación.
_FACT_STYLES = {
    "es": (
        "Write each fact as a single Spanish sentence starting with 'El usuario', e.g.:\n"
        '  "El usuario se llama Carlos García."\n'
        '  "El usuario trabaja en Viajes Express como director de operaciones."\n'
        '  "El usuario tiene la reserva con referencia RES-2024-001."\n'
    ),
    "en": (
        "Write each fact as a single English sentence starting with 'The user', e.g.:\n"
        '  "The user is named Carlos Garcia."\n'
        '  "The user works at Viajes Express as director de operaciones."\n'
        '  "The user has booking reference RES-2024-001."\n'
    ),
}


def _fact_language() -> str:
    language = os.getenv("MEMORI_FACT_LANGUAGE", "es").strip().lower()
    return language if language in _FACT_STYLES else "es"


# Contadores de fallos de memoria. Antes se tragaban en silencio; ahora se
# registran como warning y se exponen en /health para que la evaluación
# detecte corridas en las que Memori no escribió o no recuperó nada.
MEMORY_FAILURES: dict[str, int] = {"extraction": 0, "embedding": 0, "write": 0, "recall": 0}

# Caché de extracción de hechos (solo evaluación). Permite sembrar el mismo
# historial varias veces sin repetir llamadas al LLM: la clave incluye el modelo
# y el prompt completo, así que cambiar cualquiera de los dos invalida la entrada.
_EXTRACTION_CACHE: dict[str, dict[str, Any]] | None = None


async def _offload(fn: Any, *args: Any) -> Any:
    """Ejecuta una llamada síncrona de Memori (BD, embeddings ONNX).

    Por defecto se ejecuta en el propio event loop, que es el comportamiento original y
    bloquea las demás peticiones mientras dura. Con MEMORI_OFFLOAD_SYNC=true se lleva a
    un hilo (asyncio.to_thread) para no bloquearlo; la prueba de carga compara ambas
    variantes para contrastar la afirmación sobre el procesamiento asíncrono.
    """
    if _env_flag("MEMORI_OFFLOAD_SYNC"):
        return await asyncio.to_thread(fn, *args)
    return fn(*args)


def _record_memory_failure(stage: str, exc: BaseException) -> None:
    MEMORY_FAILURES[stage] = MEMORY_FAILURES.get(stage, 0) + 1
    LOGGER.warning("Fallo de memoria en la etapa %s: %s", stage, exc)


def _extraction_cache_path() -> Path | None:
    raw = os.getenv("EVAL_EXTRACTION_CACHE", "").strip()
    return Path(raw) if raw else None


def _extraction_cache() -> dict[str, dict[str, Any]]:
    global _EXTRACTION_CACHE
    if _EXTRACTION_CACHE is None:
        _EXTRACTION_CACHE = {}
        path = _extraction_cache_path()
        if path is not None and path.exists():
            # Solo "\n": splitlines() también corta en U+2028 y similares, que pueden
            # aparecer sin escapar dentro de los hechos guardados.
            for line in path.read_text(encoding="utf-8").split("\n"):
                try:
                    entry = json.loads(line)
                    _EXTRACTION_CACHE[entry["key"]] = entry
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    return _EXTRACTION_CACHE


def _store_extraction(key: str, facts: list[str], usage: dict[str, int]) -> None:
    entry = {"key": key, "facts": facts, "usage": usage}
    _extraction_cache()[key] = entry
    path = _extraction_cache_path()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extraction_model_name() -> str:
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


async def _extract_facts(user_message: str, assistant_reply: str) -> dict[str, Any]:
    """Extrae hechos del usuario con el LLM configurado.

    Devuelve {"facts", "usage", "cached"}; lanza la excepción si la llamada falla.
    """
    from openai import AsyncOpenAI

    model_name = _extraction_model_name()
    content = _FACT_EXTRACT_PROMPT.format(
        scope=_FACT_SCOPES[_prompt_profile()], style=_FACT_STYLES[_fact_language()],
        user=user_message, assistant=assistant_reply,
    )
    use_cache = _env_flag("EVAL_MODE")
    key = hashlib.sha256(f"{model_name}\n{content}".encode("utf-8")).hexdigest()
    if use_cache and key in _extraction_cache():
        cached = _extraction_cache()[key]
        return {"facts": list(cached["facts"]), "usage": dict(cached.get("usage") or {}), "cached": True}

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY") if provider != "deepseek" else (
        os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    base_url = os.getenv("LLM_BASE_URL") or ("https://api.deepseek.com" if provider == "deepseek" else None)

    # Extracción determinista (temperature=0) salvo que el modelo solo admita
    # la temperatura por defecto (LLM_TEMPERATURE vacía): entonces no se envía.
    # El esfuerzo de razonamiento es el mismo que el del chat.
    extra: dict[str, Any] = {} if _llm_temperature() is None else {"temperature": 0}
    reasoning_effort = _llm_reasoning_effort()
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort

    oai = AsyncOpenAI(api_key=api_key, base_url=base_url)
    resp = await oai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        **extra,
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    raw = data.get("facts", []) if isinstance(data, dict) else []
    facts = [f.strip() for f in raw if isinstance(f, str) and f.strip()]
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
    }
    if use_cache:
        _store_extraction(key, facts, usage)
    return {"facts": facts, "usage": usage, "cached": False}


async def _trigger_memori_augmentation(
    memori_session: object, user_message: str, assistant_reply: str
) -> dict[str, Any]:
    """Store conversation turn and extract entity facts locally (no Memori cloud).

    Memori's built-in augmentation calls api.memorilabs.ai which has an anonymous
    IP quota. Instead we:
      1. Write conversation messages via Memori's Writer (sets conversation_id).
      2. Extract facts with our own OpenAI call.
      3. Embed facts locally using Memori's Rust ONNX core.
      4. Write entity_facts directly to the local SQLite DB.

    Devuelve un resumen (hechos, tokens, si vino de caché) para la traza de evaluación.
    """
    summary: dict[str, Any] = {"facts": [], "usage": {}, "cached": False, "written": False}

    try:
        from memori.memory._manager import Manager as MemoryManager
        from memori.storage._connection import connection_context
    except ImportError as exc:
        _record_memory_failure("write", exc)
        return summary

    config = memori_session.config  # type: ignore[union-attr]

    # 1. Store conversation messages so that conversation_id is created/cached.
    await _offload(
        MemoryManager(config).execute,
        {
            "messages": [
                {"role": "user", "type": "text", "text": user_message},
                {"role": "assistant", "type": "text", "text": assistant_reply},
            ]
        },
    )

    conversation_id = config.cache.conversation_id
    entity_id = config.entity_id
    if not conversation_id or not entity_id:
        _record_memory_failure("write", RuntimeError("sin conversation_id o entity_id"))
        return summary

    # 2. Extract facts with our configured LLM (OpenAI/DeepSeek), same key.
    try:
        extraction = await _extract_facts(user_message, assistant_reply)
    except Exception as exc:
        _record_memory_failure("extraction", exc)
        return summary

    facts: list[str] = extraction["facts"]
    summary.update(facts=facts, usage=extraction["usage"], cached=extraction["cached"])
    if not facts:
        return summary

    # 3. Embed locally via Memori's Rust ONNX core (all-MiniLM-L6-v2, no cloud).
    embeddings = None
    try:
        rust_core = getattr(memori_session.config, "rust_core", None)  # type: ignore[union-attr]
        if rust_core is not None:
            # Use Rust EngineHandle embedding which runs ONNX locally.
            embeddings = await _offload(memori_session.embed_texts, facts)  # type: ignore[union-attr]
    except Exception as exc:
        _record_memory_failure("embedding", exc)
        embeddings = None

    # 4. Write entity_facts to the local SQLite DB.
    def _write_facts() -> None:
        with connection_context(config.storage.conn_factory) as (_conn, adapter, driver):
            entity_db_id = driver.entity.create(entity_id)
            driver.entity_fact.create(
                entity_db_id,
                facts,
                fact_embeddings=embeddings,
                conversation_id=int(conversation_id),
            )
            adapter.flush()
            adapter.commit()

    try:
        await _offload(_write_facts)
        summary["written"] = True
    except Exception as exc:
        _record_memory_failure("write", exc)
    return summary


def _memori_recall(memori_session: object, user_query: str) -> tuple[str, list[dict[str, Any]]]:
    """Devuelve (bloque <memori_context>, memorias recuperadas).

    La lista incluye todas las memorias que devolvió Memori (top-k), cada una con su
    puntuación y si superó el umbral de relevancia, para poder calcular recall@k.
    """
    try:
        from memori._utils import format_date_created
        from memori.memory.recall import _score_for_recall_threshold

        facts = memori_session.recall(user_query)  # type: ignore[union-attr]
        if not facts:
            return "", []

        threshold = memori_session.config.recall_relevance_threshold  # type: ignore[union-attr]
        retrieved: list[dict[str, Any]] = []
        lines: list[str] = []
        for f in facts:
            if isinstance(f, str):
                content, date_created = f, None
            elif isinstance(f, dict):
                content, date_created = f.get("content"), f.get("date_created")
            else:
                content, date_created = getattr(f, "content", None), getattr(f, "date_created", None)
            if not content:
                continue
            score = float(_score_for_recall_threshold(f))
            relevant = score >= threshold
            retrieved.append({"content": str(content), "score": score, "above_threshold": relevant})
            if relevant:
                ts = format_date_created(date_created) if date_created else None
                lines.append(f"- {content}" + (f". Stated at {ts}" if ts else ""))

        if not lines:
            return "", retrieved

        block = (
            "<memori_context>Only use the relevant context if it is relevant to the user's query. "
            "Relevant context about the user:\n" + "\n".join(lines) + "\n</memori_context>"
        )
        return block, retrieved
    except Exception as exc:
        _record_memory_failure("recall", exc)
        return "", []


def _close_memori(memori_session: object | None) -> None:
    """Devuelve al pool la conexión que cada instancia de Memori retiene.

    Sin esto la conexión solo se libera con el recolector de basura: con 20 peticiones
    concurrentes se agotaba el pool de SQLAlchemy (5 + 10) y, en modo síncrono, la espera
    bloqueaba el event loop hasta el timeout de 30 s.
    """
    if memori_session is None:
        return
    try:
        memori_session.close()  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        _record_memory_failure("close", exc)


def _build_model_with_memory(session_id: str, entity_id: str) -> tuple[BaseChatModel, object | None]:
    global MEMORI_STORAGE_READY
    model = _build_model()

    if not _env_flag("MEMORI_ENABLED"):
        return model, None

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "ollama":
        raise RuntimeError(
            "Memori integration is not configured for Ollama in this project. "
            "Disable MEMORI_ENABLED or use openai/deepseek."
        )

    if Memori is None:
        raise RuntimeError(
            "Memori integration is enabled but the 'memori' package is not installed."
        )

    global _SHARED_RUST_CORE
    memori = Memori(conn=_get_memori_session_factory()).llm.register(chatopenai=model)
    if memori.config.rust_core is not None:
        if _SHARED_RUST_CORE is None:
            _SHARED_RUST_CORE = memori.config.rust_core
        else:
            memori.config.rust_core = _SHARED_RUST_CORE
    if not MEMORI_STORAGE_READY:
        memori.config.storage.build()
        MEMORI_STORAGE_READY = True
    memori.attribution(
        entity_id=entity_id,
        process_id=os.getenv("MEMORI_PROCESS_ID", "support-api"),
    )
    memori.set_session(session_id)
    return model, memori


def _load_vector_store() -> GenericVectorStore | None:
    global VECTOR_STORE
    if VECTOR_STORE is None:
        try:
            VECTOR_STORE = GenericVectorStore.load()
        except RuntimeError:
            VECTOR_STORE = None
    return VECTOR_STORE


def _to_langchain_messages(messages: list["ChatMessage"]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []

    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content))
        else:
            converted.append(HumanMessage(content=message.content))

    return converted


def _faq_context(message: str) -> str:
    if not FAQ_ENTRIES:
        return ""

    matches: list[tuple[int, dict[str, str]]] = []

    for entry in FAQ_ENTRIES:
        score = _entry_score(message, entry)
        if score:
            matches.append((score, entry))

    matches.sort(key=lambda item: item[0], reverse=True)
    selected = [entry for _, entry in matches[:3]]

    if not selected:
        selected = FAQ_ENTRIES[:2]

    return "\n\n".join(
        f"FAQ: {entry['question']}\nRespuesta: {entry['answer']}" for entry in selected
    )


def _should_escalate(message: str) -> bool:
    if not _env_flag("ESCALATION_ENABLED", default=True):
        return False
    lowered = message.lower()
    return any(keyword in lowered for keyword in ESCALATION_KEYWORDS)


def _is_definition_query(message: str) -> bool:
    lowered = message.lower().strip()
    patterns = (
        "que es ",
        "qué es ",
        "definicion de ",
        "definición de ",
        "que significa ",
        "qué significa ",
        "significado de ",
    )
    return any(lowered.startswith(pattern) for pattern in patterns)


def _extract_definition_target(message: str) -> str:
    lowered = message.lower().strip().rstrip("?.!")
    prefixes = (
        "que es ",
        "qué es ",
        "definicion de ",
        "definición de ",
        "que significa ",
        "qué significa ",
        "significado de ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return lowered[len(prefix):].strip()
    return lowered


def _glossary_answer(message: str) -> str | None:
    if not GLOSSARY_ENTRIES or not _is_definition_query(message):
        return None

    target = _extract_definition_target(message)
    target_tokens = _tokenize(target)
    best_score = 0
    best_definition: str | None = None

    for entry in GLOSSARY_ENTRIES:
        term = entry["term"]
        term_tokens = _tokenize(term)
        score = (3 if target == term else 0) + (2 if target in term or term in target else 0) + len(target_tokens & term_tokens)
        if score > best_score:
            best_score = score
            best_definition = entry["definition"]

    if best_score >= 2:
        return best_definition
    return None


def _needs_incident_context(message: str) -> bool:
    # ISSUE_CONTEXT_ENABLED=false desactiva la respuesta predefinida; la evaluación
    # de memoria la apaga para que todos los turnos lleguen al LLM.
    if not _env_flag("ISSUE_CONTEXT_ENABLED", default=True):
        return False
    lowered = message.lower()
    issue_terms = (
        "error",
        "falla",
        "no puedo",
        "no carga",
        "no abre",
        "permiso",
        "sin permisos",
        "acceso denegado",
        "no me deja",
        "problema al",
        "publicar",
        "guardar",
        "adjuntar",
        "anexar",
        "crear",
        "revisión",
        "revision",
        "firma",
        "visto bueno",
    )
    has_reference = any(char.isdigit() for char in message)
    return not _is_definition_query(message) and any(term in lowered for term in issue_terms) and not has_reference


def _trim_history(messages: list["ChatMessage"], limit: int = 12) -> list["ChatMessage"]:
    return messages[-limit:]


def _rag_context(message: str, k: int = 4) -> tuple[str, list["SourceItem"]]:
    store = _load_vector_store()
    if store is None:
        return "", []

    chunks = store.search(message, k=k)
    if not chunks:
        return "", []

    context = "\n\n".join(
        f"Fuente: {chunk.title}\nURL: {chunk.url}\nContenido: {chunk.text}" for chunk in chunks
    )
    sources = [SourceItem(title=chunk.title, url=chunk.url) for chunk in chunks]
    return context, sources


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., examples=["user"])
    content: str = Field(..., min_length=1, examples=["Hola, quien eres?"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Explicame que es FastAPI."])
    session_id: str | None = Field(default=None, examples=["cliente-demo-001"])
    entity_id: str | None = Field(
        default=None,
        examples=["usuario-001"],
        description=(
            "Identificador estable del usuario a través de sesiones. "
            "Memori usa este campo para recuperar memoria cross-session. "
            "Si no se proporciona, se usa session_id (sin memoria entre sesiones)."
        ),
    )
    history: list[ChatMessage] = Field(default_factory=list)
    eval_context: str | None = Field(
        default=None,
        description=(
            "Solo con EVAL_MODE=true. Contexto de sesiones previas que inyectan los scripts de evaluación "
            "en el mismo hueco que <memori_context> (condiciones C1 contexto completo, "
            "C2 RAG simple y C4 oracle). Se ignora fuera del modo de evaluación."
        ),
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    escalated: bool = False
    sources: list["SourceItem"] = Field(default_factory=list)
    debug: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Solo con EVAL_MODE=true: ruta seguida, memorias recuperadas, tokens y "
            "tiempos por etapa (recuperación, generación y escritura de memoria) en ms."
        ),
    )


class IngestTurn(BaseModel):
    user: str = Field(..., min_length=1)
    assistant: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    entity_id: str = Field(..., min_length=1, max_length=100)
    session_id: str = Field(..., min_length=1)
    turns: list[IngestTurn] = Field(default_factory=list)


class IngestResponse(BaseModel):
    entity_id: str
    session_id: str
    turns: int
    facts: list[str]
    extraction_calls: int
    cached_extractions: int
    usage: dict[str, int]
    write_ms: float


class SourceItem(BaseModel):
    title: str
    url: str


app = FastAPI(
    title="Simple LangChain Chat API",
    version="1.0.0",
    description="Backend sencillo de atencion al cliente con FastAPI, LangChain y OpenAI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Perfil "neutral": asistente general para la validación externa con LongMemEval,
# que es charla general en inglés. Mismo modelo, misma memoria y mismos huecos de contexto;
# solo cambia el rol, para que el dominio de soporte de viajes no sesgue las respuestas.
_NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful assistant that has had previous conversations with this user. "
    "Answer in the language of the user's question, concisely. "
    "When the memory or previous-conversation context contains information relevant to the question, "
    "use it to answer directly. If facts in that context contradict each other, prefer the most recent one. "
    "Use the dates given in the context and the current date given in the question for any temporal reasoning. "
    "If the answer is not in the context, say that you do not have that information instead of guessing."
)


def _prompt_profile() -> str:
    profile = os.getenv("EVAL_PROMPT_PROFILE", "support").strip().lower()
    return profile if profile in {"support", "neutral"} else "support"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _NEUTRAL_SYSTEM_PROMPT if _prompt_profile() == "neutral" else (
                "Eres un asistente de atencion al cliente. "
                f"Atiendes consultas funcionales del producto {PRODUCT_NAME}. "
                "Responde en espanol, con tono profesional, breve y claro. "
                "Usa solo la informacion del contexto FAQ cuando aplique. "
                "Cuando el bloque <memori_context> contenga informacion sobre el usuario o el caso "
                "(nombre, empresa, cargo, numero de referencia, estado, contacto tecnico), "
                "usa ESA informacion para responder directamente sin pedir mas contexto. "
                "Si el contexto de memoria contiene hechos contradictorios sobre el mismo caso, "
                "usa siempre el hecho con el timestamp mas reciente. "
                "Cuando el usuario te informe de su nombre, empresa, cargo o numero de referencia, "
                "menciona explicitamente esos datos en tu respuesta para confirmar que los has registrado. "
                "Si el usuario reporta un problema sin numero de referencia ni datos del caso, pide contexto: "
                "la accion realizada, el comportamiento observado y el resultado esperado. "
                "No inventes funcionalidades, configuraciones, endpoints ni estados internos que no esten en el contexto. "
                "Prioriza la informacion del contexto RAG y de las FAQs por encima de cualquier suposicion. "
                "Cuando hables de soporte humano, usa estos canales: "
                f"{SUPPORT_CONTACT}. "
                "Si el caso requiere revision humana, indicalo con claridad y ofrece escalarlo."
            ),
        ),
        ("system", "Contexto documental recuperado:\n{rag_context}"),
        ("system", "Contexto FAQ disponible:\n{faq_context}"),
        ("system", "{memory_context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


@app.get("/health")
async def health() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "rag_index_loaded": "yes" if _load_vector_store() else "no",
    }
    if _env_flag("EVAL_MODE"):
        # Configuración efectiva, para el manifiesto de cada corrida.
        payload["eval"] = {
            "memori_enabled": _env_flag("MEMORI_ENABLED"),
            "llm_provider": os.getenv("LLM_PROVIDER", "openai").strip().lower(),
            "llm_model": _extraction_model_name(),
            "llm_temperature": _llm_temperature(),
            "llm_reasoning_effort": _llm_reasoning_effort(),
            "memori_fact_language": _fact_language(),
            "memori_offload_sync": _env_flag("MEMORI_OFFLOAD_SYNC"),
            "memori_db_pool": os.getenv("MEMORI_DB_POOL", "null").strip().lower(),
            "prompt_profile": _prompt_profile(),
            "issue_context_enabled": _env_flag("ISSUE_CONTEXT_ENABLED", default=True),
            "escalation_enabled": _env_flag("ESCALATION_ENABLED", default=True),
            "faq_entries": len(FAQ_ENTRIES),
            "glossary_entries": len(GLOSSARY_ENTRIES),
            "memory_failures": dict(MEMORY_FAILURES),
        }
    return payload


def _usage_from_message(message: AIMessage) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid4())
    # entity_id identifica al usuario de forma estable entre sesiones.
    # Si no se proporciona, cada sesión es su propio "usuario" (sin cross-session recall).
    effective_entity_id = payload.entity_id or session_id
    history_source = payload.history or SESSION_STORE.get(session_id, [])
    history = _trim_history(list(history_source))
    escalated = _should_escalate(payload.message)
    eval_mode = _env_flag("EVAL_MODE")

    def _route_debug(route: str) -> dict[str, Any] | None:
        return {"route": route} if eval_mode else None

    if _needs_incident_context(payload.message):
        reply = ISSUE_CONTEXT_REQUEST
        updated_history = history + [
            ChatMessage(role="user", content=payload.message),
            ChatMessage(role="assistant", content=reply),
        ]
        SESSION_STORE[session_id] = _trim_history(updated_history)
        return ChatResponse(
            session_id=session_id, reply=reply, escalated=escalated, sources=[],
            debug=_route_debug("issue_context"),
        )

    if escalated:
        reply = f"{ESCALATION_REQUEST} Canales de soporte: {SUPPORT_CONTACT}."
        updated_history = history + [
            ChatMessage(role="user", content=payload.message),
            ChatMessage(role="assistant", content=reply),
        ]
        SESSION_STORE[session_id] = _trim_history(updated_history)
        return ChatResponse(
            session_id=session_id, reply=reply, escalated=True, sources=[],
            debug=_route_debug("escalation"),
        )

    glossary_answer = _glossary_answer(payload.message)
    if glossary_answer:
        updated_history = history + [
            ChatMessage(role="user", content=payload.message),
            ChatMessage(role="assistant", content=glossary_answer),
        ]
        SESSION_STORE[session_id] = _trim_history(updated_history)
        glossary_source = SourceItem(
            title=GLOSSARY_TITLE,
            url=GLOSSARY_URL,
        )
        return ChatResponse(
            session_id=session_id,
            reply=glossary_answer,
            escalated=False,
            sources=[glossary_source],
            debug=_route_debug("glossary"),
        )

    rag_context, rag_sources = _rag_context(payload.message)

    try:
        model, _memori_session = await _offload(_build_model_with_memory, session_id, effective_entity_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        # Etapa de recuperación: solo la llamada a Memori (H4 mide esta etapa aislada).
        retrieval_start = time.perf_counter()
        retrieved: list[dict[str, Any]] = []
        memory_context = ""
        if _memori_session is not None:
            memory_context, retrieved = await _offload(_memori_recall, _memori_session, payload.message)
        retrieval_ms = _elapsed_ms(retrieval_start) if _memori_session is not None else None

        # Condiciones C1/C2/C4: los scripts de evaluación aportan el contexto previo en el mismo hueco.
        if eval_mode and payload.eval_context:
            memory_context = payload.eval_context

        chain = prompt | model
        generation_start = time.perf_counter()
        ai_message = await chain.ainvoke(
            {
                "history": _to_langchain_messages(history),
                "input": payload.message,
                "faq_context": _faq_context(payload.message),
                "rag_context": rag_context or "Sin contexto recuperado.",
                "memory_context": memory_context or "Sin contexto de memoria.",
            }
        )
        generation_ms = _elapsed_ms(generation_start)
        reply = StrOutputParser().invoke(ai_message)

        write_ms: float | None = None
        augmentation: dict[str, Any] | None = None
        if _memori_session is not None:
            write_start = time.perf_counter()
            augmentation = await _trigger_memori_augmentation(_memori_session, payload.message, reply)
            write_ms = _elapsed_ms(write_start)

        updated_history = history + [
            ChatMessage(role="user", content=payload.message),
            ChatMessage(role="assistant", content=reply),
        ]
        SESSION_STORE[session_id] = _trim_history(updated_history)

        debug: dict[str, Any] | None = None
        if eval_mode:
            debug = {
                "route": "llm",
                "memory_source": "memori" if _memori_session is not None else (
                    "eval_context" if payload.eval_context else "none"
                ),
                "retrieved_memories": retrieved,
                "memory_context_chars": len(memory_context),
                "usage": _usage_from_message(ai_message),
                "extraction_usage": (augmentation or {}).get("usage", {}),
                "extracted_facts": (augmentation or {}).get("facts", []),
                "timings_ms": {
                    "retrieval": retrieval_ms,
                    "generation": generation_ms,
                    "memory_write": write_ms,
                },
            }

        return ChatResponse(
            session_id=session_id, reply=reply, escalated=False, sources=rag_sources, debug=debug,
        )
    finally:
        _close_memori(_memori_session)


@app.post("/eval/ingest", response_model=IngestResponse)
async def eval_ingest(payload: IngestRequest) -> IngestResponse:
    """Siembra sesiones previas con respuestas guionizadas (solo EVAL_MODE).

    Usa la misma ruta de escritura que /chat (_trigger_memori_augmentation), pero sin
    generar respuesta con el LLM. Cada llamada es una sesión: Memori abre una
    conversación nueva por session_id, igual que en uso real.
    """
    if not _env_flag("EVAL_MODE"):
        raise HTTPException(status_code=404, detail="Not Found")
    if not _env_flag("MEMORI_ENABLED"):
        raise HTTPException(status_code=400, detail="/eval/ingest requiere MEMORI_ENABLED=true.")

    try:
        _, memori_session = await _offload(_build_model_with_memory, payload.session_id, payload.entity_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if memori_session is None:
        raise HTTPException(status_code=500, detail="No se pudo iniciar la sesión de Memori.")

    try:
        facts: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        extraction_calls = 0
        cached = 0
        start = time.perf_counter()
        for turn in payload.turns:
            summary = await _trigger_memori_augmentation(memori_session, turn.user, turn.assistant)
            facts.extend(summary["facts"])
            if summary["cached"]:
                cached += 1
            elif summary["usage"]:
                extraction_calls += 1
            for key in usage:
                usage[key] += int(summary["usage"].get(key, 0) or 0)

        return IngestResponse(
            entity_id=payload.entity_id,
            session_id=payload.session_id,
            turns=len(payload.turns),
            facts=facts,
            extraction_calls=extraction_calls,
            cached_extractions=cached,
            usage=usage,
            write_ms=_elapsed_ms(start),
        )
    finally:
        _close_memori(memori_session)
