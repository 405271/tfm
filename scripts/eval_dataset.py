"""
Esquema, validación e instanciación del dataset de evaluación de memoria (v2).

Lo usan el generador (generate_eval_dataset.py), el verificador (check_eval_dataset.py)
y el runner de condiciones. Reglas del dataset:

- Cinco categorías de LongMemEval, todas inter-sesión: la evidencia está siempre en
  sesiones anteriores a la de la pregunta.
- Cada caso tiene sesiones de evidencia propias; las sesiones distractoras salen de
  un pool compartido sin datos personales, para poder instanciar el mismo caso con
  5, 20 o 50 sesiones totales y comparar por longitud (H3).
- Las comprobaciones por palabra clave ignoran acentos y mayúsculas y exigen palabra
  completa, para evitar falsos positivos como "Ana" dentro de "mañana".

Formato de un caso (una línea JSONL):
{
  "id": "test-extraction-001", "split": "test" | "dev", "type": <CASE_TYPES>,
  "intent": <INTENTS>, "domain": "travel_agency",
  "persona": {"name": "..."},
  "evidence_sessions": [{"label": "...", "turns": [{"user": "...", "assistant": "..."}]}],
  "question": "...", "reference_answer": "...",
  "answer_keywords": ["..."],      # todas deben aparecer en una respuesta correcta
  "forbidden_keywords": ["..."],   # valor antiguo (actualización) o datos cercanos (abstención)
  "gold_facts": [{"text": "...", "keywords": ["..."]}],  # para recall@k
  "slots": {"destino": "...", ...},                     # datos ya entregados (RR)
  "task": {"goal": "...", "opening_message": "...", "required_slots": ["..."],
           "success_criteria": "..."}                   # usuario simulado (STR)
}
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

CASE_TYPES = ("extraction", "multi_session", "temporal", "knowledge_update", "abstention")

# Etiquetas en español para tablas y reportes.
CASE_TYPE_LABELS = {
    "extraction": "Extracción",
    "multi_session": "Multi-sesión",
    "temporal": "Temporal",
    "knowledge_update": "Actualización",
    "abstention": "Abstención",
}

# Mínimo de sesiones de evidencia por categoría.
MIN_EVIDENCE_SESSIONS = {
    "extraction": 1,
    "multi_session": 2,
    "temporal": 2,
    "knowledge_update": 2,
    "abstention": 1,
}

# Intenciones del contexto de uso de la agencia, más dos añadidas para la evaluación.
INTENTS = (
    "consulta sobre destinos turísticos",
    "búsqueda de paquetes vacacionales",
    "gestión de reservas",
    "seguimiento de itinerarios",
    "atención de incidencias",
    "recomendaciones personalizadas",
    "modificación de vuelos u hospedaje",
    "soporte postventa",
    "pagos y facturación",
    "documentación de viaje",
)

HISTORY_LENGTHS = (5, 20, 50)


# ---------------------------------------------------------------------------
# Normalización y coincidencia
# ---------------------------------------------------------------------------

def fold(text: str) -> str:
    """Minúsculas, sin acentos y con espacios colapsados."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.lower()).strip()


def contains_term(text: str, term: str) -> bool:
    """True si `term` aparece en `text` como palabra o frase completa."""
    folded_term = fold(term)
    if not folded_term:
        return False
    pattern = r"(?<!\w)" + re.escape(folded_term) + r"(?!\w)"
    return re.search(pattern, fold(text)) is not None


def contains_all(text: str, terms: Iterable[str]) -> bool:
    return all(contains_term(text, term) for term in terms)


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(contains_term(text, term) for term in terms)


# ---------------------------------------------------------------------------
# Carga y escritura
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    # Solo "\n": splitlines() también corta en U+2028, U+0085, etc., que json.dumps con
    # ensure_ascii=False deja sin escapar dentro de las cadenas (aparecen en LongMemEval).
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def session_text(session: dict[str, Any]) -> str:
    return "\n".join(f"{turn['user']}\n{turn['assistant']}" for turn in session.get("turns", []))


def evidence_text(case: dict[str, Any]) -> str:
    return "\n".join(session_text(s) for s in case.get("evidence_sessions", []))


def scenario_text(case: dict[str, Any]) -> str:
    """Texto que representa el escenario para medir diversidad."""
    user_turns = [
        turn["user"] for session in case.get("evidence_sessions", []) for turn in session.get("turns", [])
    ]
    return "\n".join(user_turns + [case.get("question", "")])


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def _str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def _turns_ok(session: Any) -> bool:
    turns = session.get("turns") if isinstance(session, dict) else None
    return isinstance(turns, list) and all(
        isinstance(t, dict)
        and isinstance(t.get("user"), str) and t["user"].strip()
        and isinstance(t.get("assistant"), str) and t["assistant"].strip()
        for t in turns
    )


def validate_case(case: dict[str, Any], min_turns: int = 10) -> list[str]:
    """Devuelve la lista de errores; vacía si el caso es válido."""
    errors: list[str] = []
    case_type = case.get("type")
    if case_type not in CASE_TYPES:
        return [f"type inválido: {case_type!r}"]

    for key in ("question", "reference_answer"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            errors.append(f"falta {key}")
    for key in ("answer_keywords", "forbidden_keywords"):
        if not isinstance(case.get(key), list) or (case[key] and not _str_list(case[key])):
            errors.append(f"{key} debe ser una lista de cadenas")

    sessions = case.get("evidence_sessions")
    if not isinstance(sessions, list) or not sessions:
        errors.append("faltan evidence_sessions")
        return errors
    if len(sessions) < MIN_EVIDENCE_SESSIONS[case_type]:
        errors.append(
            f"{case_type} necesita al menos {MIN_EVIDENCE_SESSIONS[case_type]} sesiones de evidencia"
        )
    for index, session in enumerate(sessions):
        if not _turns_ok(session):
            errors.append(f"sesión {index}: turnos mal formados")
        elif len(session["turns"]) < min_turns:
            errors.append(f"sesión {index}: {len(session['turns'])} turnos (< {min_turns})")
    if errors:
        return errors

    question = case["question"]
    evidence = evidence_text(case)
    answer_keywords: list[str] = case.get("answer_keywords") or []
    forbidden: list[str] = case.get("forbidden_keywords") or []

    if case_type == "abstention":
        if answer_keywords:
            errors.append("abstention no lleva answer_keywords")
        if not forbidden:
            errors.append("abstention necesita forbidden_keywords (datos cercanos del historial)")
    else:
        if not answer_keywords:
            errors.append("faltan answer_keywords")
        # Sin eco: la pregunta no puede contener la respuesta (fallo de los casos v1).
        echoed = [k for k in answer_keywords if contains_term(question, k)]
        if echoed:
            errors.append(f"la pregunta contiene la respuesta: {echoed}")
        missing = [k for k in answer_keywords if not contains_term(evidence, k)]
        if missing:
            errors.append(f"answer_keywords ausentes de la evidencia: {missing}")

    if case_type == "knowledge_update":
        if not forbidden:
            errors.append("knowledge_update necesita forbidden_keywords (valor antiguo)")
        else:
            first = session_text(sessions[0])
            last = session_text(sessions[-1])
            if not contains_any(first, forbidden):
                errors.append("el valor antiguo no aparece en la primera sesión de evidencia")
            if not contains_all(last, answer_keywords):
                errors.append("el valor nuevo no aparece en la última sesión de evidencia")
            if contains_any(question, forbidden):
                errors.append("la pregunta menciona el valor antiguo")

    if case_type == "temporal":
        # La pregunta no puede ofrecer las opciones ("¿A o B?"): se resolvería por eco.
        offered = [k for k in forbidden if contains_term(question, k)]
        if offered:
            errors.append(f"la pregunta temporal ofrece las opciones: {offered}")

    gold = case.get("gold_facts")
    if not isinstance(gold, list) or not gold:
        errors.append("faltan gold_facts")
    else:
        for fact in gold:
            if not isinstance(fact, dict) or not _str_list(fact.get("keywords")):
                errors.append("gold_fact sin keywords")
                continue
            absent = [k for k in fact["keywords"] if not contains_term(evidence, k)]
            if absent:
                errors.append(f"keywords de gold_fact ausentes de la evidencia: {absent}")

    slots = case.get("slots")
    if not isinstance(slots, dict) or not slots:
        errors.append("faltan slots")
        slots = {}
    task = case.get("task")
    if not isinstance(task, dict) or not all(
        isinstance(task.get(k), str) and task[k].strip() for k in ("goal", "opening_message", "success_criteria")
    ) or not isinstance(task.get("required_slots"), list):
        errors.append("task incompleta (goal, opening_message, required_slots, success_criteria)")
    else:
        # La tarea alimenta al usuario simulado (RR/STR): debe pedir una acción que necesite
        # datos ya dados, no repetir la pregunta del caso ni revelar esos datos.
        unknown = [s for s in task["required_slots"] if s not in slots]
        if unknown:
            errors.append(f"task.required_slots no está en slots: {unknown}")
        if len(task["required_slots"]) < 2:
            errors.append("task.required_slots necesita al menos dos slots")
        if fold(task["opening_message"]) == fold(question):
            errors.append("task.opening_message repite la pregunta del caso")
        revealed = [s for s in task["required_slots"]
                    if s in slots and contains_term(task["opening_message"], str(slots[s]))]
        if revealed:
            errors.append(f"task.opening_message revela slots ya dados: {revealed}")
    return errors


def validate_distractor(session: dict[str, Any], min_turns: int = 10) -> list[str]:
    if not _turns_ok(session):
        return ["turnos mal formados"]
    if len(session["turns"]) < min_turns:
        return [f"{len(session['turns'])} turnos (< {min_turns})"]
    return []


# ---------------------------------------------------------------------------
# Instanciación por longitud de historial
# ---------------------------------------------------------------------------

def _conflicts(session: dict[str, Any], case: dict[str, Any]) -> bool:
    """Un distractor no puede mencionar la respuesta ni los valores prohibidos del caso.

    No se usan los slots: muchos son palabras genéricas ("documentación", "tarjeta") y
    descartarían la mitad del pool sin que haya riesgo real de confusión.
    """
    terms = list(case.get("answer_keywords") or []) + list(case.get("forbidden_keywords") or [])
    text = session_text(session)
    return any(contains_term(text, term) for term in terms)


def instantiate(
    case: dict[str, Any],
    distractor_pool: list[dict[str, Any]],
    total_sessions: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Devuelve las sesiones previas del caso, en orden, para `total_sessions` sesiones.

    - Las sesiones de evidencia conservan su orden relativo (temporal y actualización).
    - Los distractores se eligen con una permutación fija por caso, así que el conjunto
      de 5 sesiones está contenido en el de 20 y este en el de 50: al alargar el historial
      solo se añaden distractores.
    - Las posiciones de la evidencia se sortean con una semilla por caso y longitud.
    Cada sesión devuelta lleva "kind" ("evidence" o "distractor") y "index".
    """
    evidence = case["evidence_sessions"]
    n_distractors = total_sessions - len(evidence)
    if n_distractors < 0:
        raise ValueError(f"{case['id']}: {len(evidence)} sesiones de evidencia > {total_sessions}")

    usable = [s for s in distractor_pool if not _conflicts(s, case)]
    order_rng = random.Random(f"{seed}:{case['id']}:distractors")
    order_rng.shuffle(usable)
    if len(usable) < n_distractors:
        raise ValueError(
            f"{case['id']}: solo {len(usable)} distractores compatibles para {total_sessions} sesiones"
        )
    chosen = usable[:n_distractors]

    position_rng = random.Random(f"{seed}:{case['id']}:{total_sessions}:positions")
    evidence_positions = sorted(position_rng.sample(range(total_sessions), len(evidence)))

    sessions: list[dict[str, Any]] = []
    evidence_iter = iter(evidence)
    distractor_iter = iter(chosen)
    for index in range(total_sessions):
        if index in evidence_positions:
            source, kind = next(evidence_iter), "evidence"
        else:
            source, kind = next(distractor_iter), "distractor"
        sessions.append(
            {
                "index": index,
                "kind": kind,
                "label": source.get("label") or source.get("id") or kind,
                "turns": source["turns"],
            }
        )
    return sessions


def evidence_indices(sessions: list[dict[str, Any]]) -> list[int]:
    return [s["index"] for s in sessions if s["kind"] == "evidence"]
