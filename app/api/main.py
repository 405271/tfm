from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4
import trafilatura
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.rag.vector_store import GenericVectorStore
from google import genai
from memori import Memori

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
FAQ_PATH = BASE_DIR / "data" / "faq.json"
GLOSSARY_PATH = BASE_DIR / "data" / "glossary.json"
SESSION_STORE: dict[str, list["ChatMessage"]] = {}
VECTOR_STORE: GenericVectorStore | None = None
MEMORI_SESSION_FACTORY = None
MEMORI_STORAGE_READY = False
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
    #"cancelacion",
    #"cancelación",
    "molesto",
    "enojado",
    "queja",
    "incidente",
    "caido",
    "caído",
    "no funciona",
}

def mask_pii(text: str) -> str:
    # Ocultar correos electrónicos
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[CORREO]', text)
    # Ocultar tarjetas de crédito o pasaportes (ejemplo genérico a numérico largo)
    text = re.sub(r'\b\d{9,16}\b', '[NUM_OCULTO]', text)
    return text

def _scrape_live(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded, include_links=False)
    return text if text else "No se pudo extraer contenido."

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


def _build_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=temperature,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL") or None,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print("Modelos disponibles:")
        for model in client.models.list():
            print(f"- {model.name}")

        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL", "gemini-pro"),
            temperature=temperature,
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
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        )

    if provider == "ollama":
        if ChatOllama is None:
            raise RuntimeError(
                "Ollama support requires the 'langchain-ollama' package to be installed."
            )

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "qwen:0.5b"),
            base_url="http://127.0.0.1:11434/v1", # El puerto local de Ollama
            api_key="ollama", # Requerido por la librería, aunque en local no se usa
            temperature=temperature,
            frequency_penalty=1.5,
            presence_penalty=0.5,
            #base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434"),
            #num_ctx=512 # Limita cuántos tokens (memoria) puede usar el modelo para leer
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
        engine = create_engine(_memori_storage_url())
        MEMORI_SESSION_FACTORY = sessionmaker(bind=engine)
    return MEMORI_SESSION_FACTORY


def _build_model_with_memory(session_id: str) -> tuple[BaseChatModel, object | None]:
    global MEMORI_STORAGE_READY
    model = _build_model()

    if not _env_flag("MEMORI_ENABLED"):
        return model, None

    if Memori is None:
        raise RuntimeError(
            "Memori integration is enabled but the 'memori' package is not installed."
        )

    # 2. Inicializa Memori (usando tu configuración de base de datos si la tienes)
    memori = Memori() 
    
    # 3. Registra el modelo directamente sin usar ProviderConfig
    memori = Memori(conn=_get_memori_session_factory()).llm.register(chatopenai=model)

    if not MEMORI_STORAGE_READY:
        memori.config.storage.build()
        MEMORI_STORAGE_READY = True
    
    # 4. Atribuir la memoria a la sesión actual
    memori.attribution(
        entity_id=session_id,
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


def _direct_faq_answer(message: str) -> str | None:
    if not FAQ_ENTRIES:
        return None

    best_score = 0
    best_answer: str | None = None

    for entry in FAQ_ENTRIES:
        score = _entry_score(message, entry)
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]

    if best_score >= 3:
        return best_answer
    return None


def _should_escalate(message: str) -> bool:
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


def _rag_context(message: str, k: int = 1) -> tuple[str, list["SourceItem"]]:
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
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    escalated: bool = False
    sources: list["SourceItem"] = Field(default_factory=list)


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

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente de atencion al cliente. "
            "REGLA ESTRICTA: Tu respuesta debe ser SIEMPRE en español. "
            "Si el contexto recuperado (RAG o Web en vivo) está en inglés u otro idioma, "
            "extrae la información, tradúcela internamente y redacta tu respuesta final únicamente en español. "
            "Utiliza un tono amable, persuasivo y profesional."
            f"Atiendes consultas funcionales del producto {PRODUCT_NAME}. "
            "Responde en espanol, con tono profesional, breve y claro. "
            "Usa solo la informacion del contexto FAQ cuando aplique. "
            "Si el usuario reporta un problema o caso incompleto, pide primero contexto suficiente para entender la situacion, "
            "la accion realizada, el comportamiento observado y el resultado esperado. "
            "No inventes funcionalidades, configuraciones, endpoints ni estados internos que no esten en el contexto. "
            "Prioriza la informacion del contexto RAG y de las FAQs por encima de cualquier suposicion. "
            "Cuando hables de soporte humano, usa estos canales: "
            f"{SUPPORT_CONTACT}. "
            "Si el caso requiere revision humana, indicalo con claridad y ofrece escalarlo."
            "Memoria del cliente:\n{memory_context}"
            "Contexto documental recuperado:\n{rag_context}"
            "Contexto FAQ disponible:\n{faq_context}"
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "rag_index_loaded": "yes" if _load_vector_store() else "no"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    start_time = time.time() # iniciar evaluacion de rendimiento
    session_id = payload.session_id or str(uuid4())
    history_source = payload.history or SESSION_STORE.get(session_id, [])
    history = _trim_history(list(history_source))
    escalated = _should_escalate(payload.message)
    safe_message = mask_pii(payload.message)
    
    if _needs_incident_context(safe_message):
        reply = ISSUE_CONTEXT_REQUEST
        updated_history = history + [
            ChatMessage(role="user", content=safe_message),
            ChatMessage(role="assistant", content=reply),
        ]
        SESSION_STORE[session_id] = _trim_history(updated_history)
        return ChatResponse(session_id=session_id, reply=reply, escalated=escalated, sources=[])

    if escalated:
        reply = f"{ESCALATION_REQUEST} Canales de soporte: {SUPPORT_CONTACT}."
        updated_history = history + [
            ChatMessage(role="user", content=safe_message),
            ChatMessage(role="assistant", content=reply),
        ]
        SESSION_STORE[session_id] = _trim_history(updated_history)
        return ChatResponse(session_id=session_id, reply=reply, escalated=True, sources=[])

    glossary_answer = _glossary_answer(safe_message)
    if glossary_answer:
        updated_history = history + [
            ChatMessage(role="user", content=safe_message),
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
        )

    direct_answer = _direct_faq_answer(safe_message)
    if direct_answer:
        updated_history = history + [
            ChatMessage(role="user", content=safe_message),
            ChatMessage(role="assistant", content=direct_answer),
        ]
        SESSION_STORE[session_id] = _trim_history(updated_history)
        return ChatResponse(session_id=session_id, reply=direct_answer, escalated=False, sources=[])

    # 1. Recuperar contexto RAG normal (PDFs y docs predefinidos)
    rag_context, rag_sources = _rag_context(safe_message, k = 1)
    live_context = ""

    # 2. Lógica simple para detectar URLs en el mensaje del usuario
    urls = re.findall(r'(https?://[^\s]+)', safe_message)
    if urls:
        # Extraer info en tiempo real de la primera URL encontrada
        live_context = _scrape_live(urls[0])
        rag_context += f"\n\nContexto Web en vivo:\n{live_context}"
        rag_sources.append(SourceItem(title="Web en tiempo real", url=urls[0]))

    try:
        model, memori_session = _build_model_with_memory(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Extraer memoria explícita si Memori está activo
    memory_context = "Sin memoria previa."
    if memori_session and hasattr(memori_session, 'get_memory'):
        memory_context = memori_session.get_memory(session_id) or memory_context
    
    chain = prompt | model | StrOutputParser()

    # Asegurar que los contextos nunca estén vacíos
    clean_memory = memory_context if memory_context and memory_context.strip() else "Ninguna todavía."
    clean_rag = rag_context if rag_context and rag_context.strip() else "Sin contexto recuperado."
    clean_faq = _faq_context(safe_message)
    clean_faq = clean_faq if clean_faq and clean_faq.strip() else "Sin FAQs relevantes."

    reply = await chain.ainvoke(
        {
            "history": _to_langchain_messages(history[-4:]), # Limita el historial a los últimos 4 mensajes
            "input": safe_message,
            "faq_context": clean_faq,
            "rag_context": clean_rag,
            "memory_context": clean_memory,
        }
    )

    # Evaluación y trazabilidad
    process_time = time.time() - start_time
    print(f"[EVALUACIÓN] Sesión: {session_id} | Tiempo: {process_time:.2f}s | Fuentes RAG: {len(rag_sources)}")

    updated_history = history + [
        ChatMessage(role="user", content=safe_message),
        ChatMessage(role="assistant", content=reply),
    ]
    SESSION_STORE[session_id] = _trim_history(updated_history)

    return ChatResponse(session_id=session_id, reply=reply, escalated=False, sources=rag_sources)
