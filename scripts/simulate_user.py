"""
Usuario simulado: tasa de repetición (RR) y pasos para la resolución (STR).

Para cada tarea, condición y corrida:
  1. Construye el historial previo del caso (eval_dataset.instantiate) y lo pone a
     disposición de la condición igual que el runner de exactitud: C3 lo siembra en una
     entidad nueva de Memori; C1 y C4 lo reciben como contexto; C2 recupera fragmentos
     del historial crudo en cada turno; C0 no recibe nada.
  2. Abre una sesión nueva con task.opening_message y conversa con el asistente:
       - el simulador (LLM distinto del evaluado, temperatura 0) conoce sus datos (slots)
         pero solo da los que el asistente le pide; si le pide uno ya dado, lo repite;
       - el detector decide qué datos pide cada turno del asistente: una regla descarta
         los turnos que no piden nada (sin "?" ni verbos de petición) y un juez LLM
         resuelve los demás, con la lista de slots del caso.
  3. Termina cuando el simulador da la tarea por completada o tras --max-turns turnos.

Métricas por conversación:
  RR  = turnos del asistente que piden al menos un dato ya entregado en sesiones previas
        / turnos del asistente de la sesión de prueba.
  STR = turnos del usuario hasta completar la tarea (si no se completa: max_turns y
        completed=false, para tratarlo como censurado en el análisis).

Uso (con los backends levantados como en run_eval_conditions.sh):
  python scripts/simulate_user.py --baseline-url http://127.0.0.1:8000 \\
      --memori-url http://127.0.0.1:8001 --output-dir artifacts/eval_runs/final_sim

Variables: SIMULATOR_MODEL (por defecto gpt-6.1-terra), DETECTOR_MODEL (por defecto
gpt-6.1-terra), *_TEMPERATURE (por defecto 0) y *_REASONING_EFFORT (por
defecto none), con prefijos SIMULATOR_ y DETECTOR_.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import CASE_TYPES, append_jsonl, contains_term, fold, instantiate, load_jsonl  # noqa: E402
from run_conditions import (  # noqa: E402
    TurnRetriever,
    _git_sha,
    _history_block,
    _post,
    _rag_block,
    _seed_memori,
)

load_dotenv()

SIM_CONDITIONS = ("C0", "C1", "C2", "C3", "C4")

_REQUEST_MARKERS = (
    "indica", "indicame", "indicanos", "dime", "dinos", "facilita", "facilitame", "facilitanos",
    "confirma", "confirmame", "necesito", "necesitaria", "podrias", "puedes decirme", "comparte",
    "compartir", "proporciona", "proporcioname", "envia", "enviame", "cual es", "cuales son",
    "cuantos", "cuantas", "cuando", "que fecha", "que destino", "que presupuesto",
)

_SIMULATOR_SYSTEM = (
    "Eres un cliente de una agencia de viajes que conversa con su asistente de atención al cliente. "
    "Escribes en español de España, con mensajes breves y naturales."
)

_SIMULATOR_RULES = """Tus datos (solo los conoces tú):
{slots}

Tu objetivo en esta conversación: {goal}
La tarea está completada cuando: {success}

Reglas:
- Solo das un dato si el asistente te lo pide explícitamente en su ÚLTIMO mensaje. Si pide varios, dalos todos.
- No adelantas datos que no te han pedido.
- Si te pide un dato que ya diste en conversaciones anteriores, lo vuelves a dar sin quejarte.
- Si te pide algo que no está en tus datos, dices que no lo sabes o que no tienes preferencia.
- Si el asistente te propone algo, aceptas lo razonable y sigues hacia tu objetivo.
- El asistente no tiene tarifas ni acceso a sistemas externos: no le exijas precios, disponibilidad ni
  acciones ejecutadas. Considera la tarea "completada" cuando haya hecho todo lo que puede con tus datos
  correctos: darte la información que buscabas, o dejar tu solicitud preparada, registrada o derivada a la
  agencia con los datos correctos.
- Si el asistente dice que no puede darte un precio o hacer la gestión él mismo, pídele que deje tu solicitud
  preparada o derivada a la agencia con tus datos, antes de dar la tarea por perdida.
- Mientras el asistente siga pidiéndote datos o avanzando, la tarea está "en_curso": responde lo que sepas.
- Considera la tarea "sin_resolver" solo si la conversación ya no avanza: el asistente repite lo mismo que en
  su mensaje anterior, se niega a preparar o derivar la solicitud, o se despide sin haberla completado.
- En otro caso, la tarea está "en_curso" y escribes tu siguiente mensaje.

Conversación de hoy hasta ahora:
{transcript}

Devuelve solo JSON: {{"estado": "en_curso" | "completada" | "sin_resolver", "mensaje": "tu siguiente mensaje, o vacío si no está en curso"}}"""

# Criterio de éxito común a todas las tareas (v2). Los criterios de cada tarea, escritos por el
# generador, exigían en 29/60 tareas un precio o una estimación que el asistente no puede dar por
# diseño (no tiene tarifas y deriva a la agencia), así que STR medía eso y no la memoria.
UNIFORM_SUCCESS = ("La solicitud queda preparada, registrada o derivada a la agencia con los datos correctos "
                   "del cliente, o el cliente recibe la información que pedía. No se exige un precio ni una "
                   "acción en sistemas externos.")

_DETECTOR_PROMPT = """Analiza el último mensaje de un asistente de atención al cliente.
Indica qué datos del cliente le PIDE el asistente que proporcione, de esta lista:
{slots}

Cuenta un dato solo si el asistente pide al cliente que lo diga o lo facilite. No cuentes:
- confirmaciones en las que el asistente ya dice el valor (por ejemplo, "¿confirmas que es Lisboa?");
- menciones del dato sin pedirlo;
- peticiones de datos que no están en la lista.

Mensaje del asistente:
{reply}

Devuelve solo JSON: {{"datos_pedidos": ["clave", ...]}}"""


class JsonLLM:
    def __init__(self, prefix: str, default_model: str, concurrency: int) -> None:
        from openai import AsyncOpenAI

        self.model = os.getenv(f"{prefix}_MODEL") or default_model
        self.client = AsyncOpenAI(
            api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv(f"{prefix}_BASE_URL") or None,
        )
        self.extra: dict[str, Any] = {}
        temperature = os.getenv(f"{prefix}_TEMPERATURE", "0").strip()
        if temperature:
            self.extra["temperature"] = float(temperature)
        effort = os.getenv(f"{prefix}_REASONING_EFFORT", "none").strip()
        if effort:
            self.extra["reasoning_effort"] = effort
        self.semaphore = asyncio.Semaphore(concurrency)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    async def ask(self, prompt: str, system: str | None = None, retries: int = 3) -> dict[str, Any]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        last: Exception | None = None
        for attempt in range(retries):
            try:
                async with self.semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model, messages=messages,
                        response_format={"type": "json_object"}, **self.extra,
                    )
                self.usage["calls"] += 1
                if resp.usage is not None:
                    self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
                    self.usage["output_tokens"] += resp.usage.completion_tokens or 0
                payload = json.loads(resp.choices[0].message.content or "{}")
                if isinstance(payload, dict):
                    return payload
                raise ValueError("respuesta no es un objeto JSON")
            except Exception as exc:
                last = exc
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"{self.model}: {last}")


def _slots_text(slots: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in slots.items())


def _transcript(turns: list[dict[str, str]]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"Tú: {turn['user']}")
        lines.append(f"Asistente: {turn['assistant']}")
    return "\n".join(lines)


def looks_like_request(reply: str) -> bool:
    """Regla: un turno sin '?' y sin verbos de petición no pide ningún dato."""
    if "?" in reply:
        return True
    return any(contains_term(reply, marker) for marker in _REQUEST_MARKERS)


async def detect_requests(detector: JsonLLM, reply: str, slots: dict[str, Any]) -> tuple[list[str], str]:
    if not looks_like_request(reply):
        return [], "regla"
    payload = await detector.ask(_DETECTOR_PROMPT.format(slots=_slots_text(slots), reply=reply))
    requested = payload.get("datos_pedidos") or []
    keys = {fold(k): k for k in slots}
    found = [keys[fold(str(r))] for r in requested if fold(str(r)) in keys]
    # Regla de confirmación: si el turno ya contiene el valor del slot, el asistente lo
    # tiene y no lo está pidiendo (p. ej. "he anotado la reserva BRS-6402"). Corrige los
    # falsos positivos del juez observados en el piloto.
    return [s for s in found if not contains_term(reply, str(slots[s]))], "juez"


def _select_tasks(cases: list[dict[str, Any]], n_tasks: int, seed: int) -> list[dict[str, Any]]:
    """Muestra estratificada por categoría (n_tasks / 5 por categoría)."""
    rng = random.Random(seed)
    per_type = max(1, n_tasks // len(CASE_TYPES))
    chosen: list[dict[str, Any]] = []
    for case_type in CASE_TYPES:
        group = sorted((c for c in cases if c["type"] == case_type), key=lambda c: c["id"])
        chosen.extend(rng.sample(group, min(per_type, len(group))))
    return sorted(chosen, key=lambda c: c["id"])


async def _conversation(
    client: httpx.AsyncClient, case: dict[str, Any], pool: list[dict[str, Any]], condition: str,
    run: int, simulator: JsonLLM, detector: JsonLLM, retriever: TurnRetriever, args: argparse.Namespace,
) -> dict[str, Any]:
    sessions = instantiate(case, pool, args.history_length, seed=args.seed)
    evidence = [s for s in sessions if s["kind"] == "evidence"]
    task = case["task"]
    slots = case["slots"]
    row: dict[str, Any] = {
        "case_id": case["id"], "type": case["type"], "condition": condition,
        "history_length": args.history_length, "run": run, "goal": task["goal"],
        "success_criterion": args.success_criterion,
    }
    entity_id = f"{args.entity_prefix}-{case['id']}-{condition}-L{args.history_length}-r{run}-{uuid4().hex[:6]}"[:100]
    base_url = args.memori_url if condition == "C3" else args.baseline_url
    fixed_context: str | None = None
    if condition == "C3":
        row["seeding"] = await _seed_memori(client, args.memori_url, entity_id, sessions, args)
    elif condition == "C1":
        fixed_context, row["truncated_sessions"] = _history_block(sessions, args.max_context_chars)
    elif condition == "C4":
        fixed_context, row["truncated_sessions"] = _history_block(evidence, args.max_context_chars)

    session_id = f"{entity_id}-sim"
    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    user_message = task["opening_message"]
    completed = False
    outcome = "max_turnos"
    usage = {"input_tokens": 0, "output_tokens": 0}

    for user_turn in range(1, args.max_turns + 1):
        eval_context = fixed_context
        if condition == "C2":
            eval_context = _rag_block(retriever.retrieve(sessions, user_message, args.k))
        data = await _post(client, f"{base_url}/chat", {
            "message": user_message, "session_id": session_id, "entity_id": entity_id,
            "history": history, "eval_context": eval_context,
        }, args.retries, args.timeout)
        reply = str(data.get("reply", ""))
        debug = data.get("debug") or {}
        for key in usage:
            usage[key] += int((debug.get("usage") or {}).get(key, 0) or 0)
        requested, method = await detect_requests(detector, reply, slots)
        record = {"user": user_message, "assistant": reply, "requested_slots": requested,
                  "repeated": bool(requested), "detection": method}
        history += [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        turns.append(record)

        sim = await simulator.ask(
            _SIMULATOR_RULES.format(
                slots=_slots_text(slots), goal=task["goal"], success=(UNIFORM_SUCCESS if args.success_criterion == "uniform"
                                                      else task["success_criteria"]),
                transcript=_transcript(turns),
            ),
            system=_SIMULATOR_SYSTEM,
        )
        state = fold(str(sim.get("estado") or "en_curso"))
        if state == "completada":
            completed = True
            break
        if state == "sin_resolver":
            outcome = "sin_resolver"
            break
        user_message = str(sim.get("mensaje") or "").strip() or "¿Puedes continuar, por favor?"

    assistant_turns = len(turns)
    repeated_turns = sum(t["repeated"] for t in turns)
    row.update(
        turns=turns,
        assistant_turns=assistant_turns,
        repeated_turns=repeated_turns,
        rr=repeated_turns / assistant_turns if assistant_turns else 0.0,
        repeated_slots=sorted({s for t in turns for s in t["requested_slots"]}),
        completed=completed,
        outcome="completada" if completed else outcome,
        # STR: turnos del usuario hasta completar; si no se completa, censurada en el máximo.
        str_turns=len(turns) if completed else args.max_turns,
        usage=usage,
    )
    return row


async def run(args: argparse.Namespace) -> Path:
    cases = [c for c in load_jsonl(args.dataset_dir / "cases.jsonl") if c["split"] == args.split]
    tasks = _select_tasks(cases, args.tasks, args.seed)
    pool = load_jsonl(args.dataset_dir / "distractors.jsonl")
    conditions = [c for c in SIM_CONDITIONS if c in set(args.conditions)]
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "conversations.jsonl"
    done = {(r["condition"], r["run"], r["case_id"]) for r in load_jsonl(results_path) if not r.get("error")}

    simulator = JsonLLM("SIMULATOR", "gpt-6.1-terra", args.concurrency * 2)
    # Detector independiente del juez: gpt-6.1-terra es el que se usó en la ejecución final.
    detector = JsonLLM("DETECTOR", "gpt-6.1-terra", args.concurrency * 2)
    retriever = TurnRetriever()
    append_jsonl(out_dir / "manifests.jsonl", {
        "started_at": datetime.now(timezone.utc).isoformat(), "git": _git_sha(),
        "tasks": [c["id"] for c in tasks], "conditions": conditions, "runs": args.runs,
        "history_length": args.history_length, "max_turns": args.max_turns, "k": args.k,
        "success_criterion": args.success_criterion,
        "simulator_model": simulator.model, "simulator_params": simulator.extra,
        "detector_model": detector.model, "detector_params": detector.extra,
    })

    semaphore = asyncio.Semaphore(args.concurrency)
    counts = {"done": 0, "errors": 0}
    async with httpx.AsyncClient() as client:
        async def job(case: dict[str, Any], condition: str, run_index: int) -> None:
            if (condition, run_index, case["id"]) in done:
                return
            async with semaphore:
                try:
                    row = await _conversation(client, case, pool, condition, run_index,
                                              simulator, detector, retriever, args)
                except Exception as exc:
                    row = {"case_id": case["id"], "type": case["type"], "condition": condition,
                           "history_length": args.history_length, "run": run_index, "error": str(exc)}
                    counts["errors"] += 1
                append_jsonl(results_path, row)
                counts["done"] += 1
                if "rr" in row:
                    print(f"  {condition} r{run_index} {case['id']}: RR {row['rr']:.2f} "
                          f"STR {row['str_turns']} completada={row['completed']}", flush=True)
                else:
                    print(f"  {condition} r{run_index} {case['id']}: ERROR {row['error'][:120]}", flush=True)

        for run_index in range(1, args.runs + 1):
            await asyncio.gather(*(job(c, cond, run_index) for c in tasks for cond in conditions))

    print(f"Conversaciones nuevas: {counts['done']} (errores: {counts['errors']}) -> {results_path}")
    print(f"Simulador: {simulator.usage}  detector: {detector.usage}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Usuario simulado: RR y STR.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("app/data/eval_dataset"))
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--conditions", nargs="+", choices=SIM_CONDITIONS, default=["C0", "C1", "C2", "C3"])
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--success-criterion", choices=["uniform", "task"], default="uniform",
                        help="uniform: criterio común (v2); task: el criterio de cada tarea (v1, corrida descartada)")
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--memori-url", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval_runs/sim"))
    parser.add_argument("--entity-prefix", default="sim")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=400_000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.baseline_url = args.baseline_url.rstrip("/")
    args.memori_url = args.memori_url.rstrip("/")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
