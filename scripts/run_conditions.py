"""
Runner de condiciones experimentales C0-C4.

Para cada longitud de historial, corrida y caso del dataset v2:
  - instancia las sesiones previas (eval_dataset.instantiate);
  - C3 (Memori): siembra esas sesiones en una entidad NUEVA con /eval/ingest (memoria vacía
    en cada corrida) y formula la pregunta al backend con Memori;
  - C0 (stateless), C1 (contexto completo), C2 (RAG simple sobre el historial crudo) y
    C4 (oracle: solo sesiones de evidencia) formulan la pregunta al backend sin memoria,
    con el contexto previo en `eval_context`;
  - registra respuesta, acierto automático auxiliar, recall@k, memorias recuperadas,
    tokens y tiempos por etapa.

El veredicto principal lo da el juez (fase 3); aquí `auto_correct` es una comprobación
auxiliar por palabras clave. Los resultados se escriben en JSONL y la ejecución es
reanudable: las filas ya completadas sin error no se repiten.

Uso típico (con scripts/run_eval_conditions.sh, que levanta los dos backends):
  python scripts/run_conditions.py --baseline-url http://127.0.0.1:8000 \\
      --memori-url http://127.0.0.1:8001 --history-lengths 5 20 --runs 3
  python scripts/run_conditions.py ... --limit 5 --runs 1 --history-lengths 5   # piloto
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import (  # noqa: E402
    append_jsonl,
    contains_all,
    contains_any,
    contains_term,
    evidence_indices,
    instantiate,
    load_jsonl,
)

CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
CONDITION_LABELS = {
    "C0": "Stateless",
    "C1": "Contexto completo",
    "C2": "RAG simple",
    "C3": "Memori",
    "C4": "Oracle",
}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # el mismo que usa Memori en C3, para que la comparación sea justa

_ABSTENTION_MARKERS = (
    "no consta", "no me consta", "no tengo", "no dispongo", "no figura", "no aparece",
    "no me has", "no me indicaste", "no nos indicaste", "no nos has", "no lo indicaste",
    "no se menciona", "no mencionaste", "no tengo constancia", "no hay constancia",
    "no tengo registrado", "no tengo informacion", "no lo tengo", "no lo encuentro",
    "no me facilitaste", "no facilitaste", "no me diste", "no indicaste", "no me has dado",
    "no me has facilitado", "no me has indicado", "no has indicado", "no has facilitado",
)


# ---------------------------------------------------------------------------
# Contexto para C1, C2 y C4
# ---------------------------------------------------------------------------

def _format_session(session: dict[str, Any]) -> str:
    date = f" ({session['date']})" if session.get("date") else ""
    lines = [f"### Sesión {session['index'] + 1}{date}"]
    for turn in session["turns"]:
        lines.append(f"Usuario: {turn['user']}")
        lines.append(f"Asistente: {turn['assistant']}")
    return "\n".join(lines)


def _history_block(sessions: list[dict[str, Any]], max_chars: int) -> tuple[str, int]:
    """Concatena sesiones de la más antigua a la más reciente.

    Si no caben en max_chars se descartan las más antiguas (límite de contexto del modelo).
    Devuelve (texto, número de sesiones descartadas).
    """
    header = (
        "<historial_previo>Conversaciones anteriores con este mismo usuario, de la más antigua a "
        "la más reciente. Úsalas solo si son relevantes para la consulta.\n"
    )
    footer = "\n</historial_previo>"
    blocks = [_format_session(s) for s in sessions]
    dropped = 0
    while blocks and len(header) + len("\n\n".join(blocks)) + len(footer) > max_chars:
        blocks.pop(0)
        dropped += 1
    return header + "\n\n".join(blocks) + footer, dropped


class TurnRetriever:
    """C2: recuperación top-k sobre el historial crudo, un fragmento por turno."""

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        from memori import embed_texts

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for start in range(0, len(missing), 256):
            batch = missing[start:start + 256]
            vectors = np.asarray(embed_texts(batch, EMBEDDING_MODEL), dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            for text, vector in zip(batch, vectors):
                self._cache[text] = vector
        return np.stack([self._cache[t] for t in texts])

    def retrieve(self, sessions: list[dict[str, Any]], query: str, k: int) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for session in sessions:
            for turn_index, turn in enumerate(session["turns"]):
                chunks.append({
                    "session": session["index"] + 1,
                    "turn": turn_index + 1,
                    "content": f"Usuario: {turn['user']}\nAsistente: {turn['assistant']}",
                })
        if not chunks:
            return []
        matrix = self._embed([c["content"] for c in chunks])
        query_vector = self._embed([query])[0]
        scores = matrix @ query_vector
        top = np.argsort(-scores)[:k]
        return [{**chunks[i], "score": float(scores[i])} for i in top]


def _rag_block(chunks: list[dict[str, Any]]) -> str:
    ordered = sorted(chunks, key=lambda c: (c["session"], c["turn"]))
    body = "\n\n".join(f"[Sesión {c['session']}, turno {c['turn']}]\n{c['content']}" for c in ordered)
    return (
        "<historial_recuperado>Fragmentos de conversaciones anteriores con este mismo usuario, "
        "recuperados por similitud con la consulta. Úsalos solo si son relevantes.\n"
        + body + "\n</historial_recuperado>"
    )


# ---------------------------------------------------------------------------
# Métricas auxiliares
# ---------------------------------------------------------------------------

def recall_at_k(case: dict[str, Any], retrieved_texts: list[str]) -> bool | None:
    """True si TODOS los gold_facts están en lo recuperado (cada uno con todas sus keywords).

    None en abstención: no existe una memoria correcta que recuperar.
    """
    if case["type"] == "abstention":
        return None
    gold = case.get("gold_facts") or []
    if not gold:
        return None
    return all(
        any(contains_all(text, fact["keywords"]) for text in retrieved_texts) for fact in gold
    )


def auto_score(case: dict[str, Any], reply: str) -> dict[str, Any]:
    """Comprobación auxiliar por palabras clave (el veredicto principal es del juez).

    Mencionar un valor prohibido no invalida la respuesta: las respuestas correctas suelen
    citarlo ("Osaka sustituyó a Kioto", "no me diste el pasaporte, sí el DNI 1234"). Se
    registra en has_forbidden para revisarlo, pero no cuenta como fallo.
    """
    has_forbidden = contains_any(reply, case.get("forbidden_keywords") or [])
    if case["type"] == "abstention":
        abstained = any(contains_term(reply, marker) for marker in _ABSTENTION_MARKERS)
        return {"auto_correct": abstained, "abstained": abstained, "has_forbidden": has_forbidden}
    keywords = case.get("answer_keywords") or []
    if not keywords:  # p. ej. LongMemEval: respuesta libre, solo la califica el juez
        return {"auto_correct": None, "has_answer": None, "has_forbidden": has_forbidden}
    has_answer = contains_all(reply, keywords)
    return {"auto_correct": has_answer, "has_answer": has_answer, "has_forbidden": has_forbidden}


# ---------------------------------------------------------------------------
# Llamadas al backend
# ---------------------------------------------------------------------------

async def _post(client: httpx.AsyncClient, url: str, payload: dict[str, Any], retries: int,
                timeout: float) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"{url}: {last_exc}")


async def _seed_memori(client: httpx.AsyncClient, base_url: str, entity_id: str,
                       sessions: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    stats = {"sessions": 0, "turns": 0, "facts": 0, "extraction_calls": 0, "cached_extractions": 0,
             "input_tokens": 0, "output_tokens": 0, "write_ms": 0.0}
    for session in sessions:
        data = await _post(client, f"{base_url}/eval/ingest", {
            "entity_id": entity_id,
            "session_id": f"{entity_id}-s{session['index'] + 1}",
            "turns": session["turns"],
        }, args.retries, args.timeout)
        stats["sessions"] += 1
        stats["turns"] += data["turns"]
        stats["facts"] += len(data["facts"])
        stats["extraction_calls"] += data["extraction_calls"]
        stats["cached_extractions"] += data["cached_extractions"]
        stats["input_tokens"] += data["usage"].get("input_tokens", 0)
        stats["output_tokens"] += data["usage"].get("output_tokens", 0)
        stats["write_ms"] += data["write_ms"]
    return stats


async def _ask(client: httpx.AsyncClient, base_url: str, question: str, entity_id: str,
               eval_context: str | None, args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    data = await _post(client, f"{base_url}/chat", {
        "message": question,
        "session_id": f"{entity_id}-test-{uuid4().hex[:6]}",
        "entity_id": entity_id,
        "history": [],
        "eval_context": eval_context,
    }, args.retries, args.timeout)
    return data, round((time.perf_counter() - start) * 1000, 2)


def _row_key(condition: str, length: int, run: int, case_id: str) -> str:
    return f"{condition}|{length}|{run}|{case_id}"


async def _run_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    pool: list[dict[str, Any]],
    length: int,
    run: int,
    conditions: list[str],
    retriever: TurnRetriever,
    args: argparse.Namespace,
    write_row: Callable[[dict[str, Any]], None],
) -> None:
    # LongMemEval trae su propio historial (fixed_sessions); el dataset propio se instancia
    # con distractores según la longitud pedida.
    sessions = case.get("fixed_sessions") or instantiate(case, pool, length, seed=args.seed)
    evidence = [s for s in sessions if s["kind"] == "evidence"]
    base = {
        "case_id": case["id"], "type": case["type"], "split": case["split"], "intent": case.get("intent"),
        "question_type": case.get("question_type"), "sessions_count": len(sessions),
        "history_length": length, "run": run, "question": case["question"],
        "reference_answer": case["reference_answer"], "evidence_sessions": evidence_indices(sessions),
    }

    for condition in conditions:
        row: dict[str, Any] = {**base, "condition": condition}
        try:
            eval_context: str | None = None
            retrieved_texts: list[str] | None = None
            if condition == "C3":
                # Entidad nueva por corrida y por intento: memoria vacía.
                entity_id = f"{args.entity_prefix}-{case['id']}-L{length}-r{run}-{uuid4().hex[:6]}"[:100]
                row["seeding"] = await _seed_memori(client, args.memori_url, entity_id, sessions, args)
                data, latency_ms = await _ask(client, args.memori_url, case["question"], entity_id, None, args)
                retrieved = (data.get("debug") or {}).get("retrieved_memories") or []
                retrieved_texts = [m["content"] for m in retrieved[: args.k]]
                row["retrieved"] = retrieved
            else:
                entity_id = f"{args.entity_prefix}-{case['id']}-{condition}-{uuid4().hex[:6]}"[:100]
                if condition == "C1":
                    eval_context, dropped = _history_block(sessions, args.max_context_chars)
                    row["truncated_sessions"] = dropped
                elif condition == "C4":
                    eval_context, dropped = _history_block(evidence, args.max_context_chars)
                    row["truncated_sessions"] = dropped
                elif condition == "C2":
                    chunks = retriever.retrieve(sessions, case["question"], args.k)
                    eval_context = _rag_block(chunks)
                    retrieved_texts = [c["content"] for c in chunks]
                    row["retrieved"] = chunks
                data, latency_ms = await _ask(client, args.baseline_url, case["question"], entity_id,
                                              eval_context, args)

            debug = data.get("debug") or {}
            reply = str(data.get("reply", ""))
            row.update(
                reply=reply,
                route=debug.get("route"),
                memory_source=debug.get("memory_source"),
                context_chars=debug.get("memory_context_chars"),
                usage=debug.get("usage"),
                extraction_usage=debug.get("extraction_usage"),
                timings_ms=debug.get("timings_ms"),
                latency_ms=latency_ms,
                recall_at_k=recall_at_k(case, retrieved_texts) if retrieved_texts is not None else None,
                **auto_score(case, reply),
            )
            if debug.get("route") != "llm":
                row["error"] = f"la petición no pasó por el LLM (route={debug.get('route')})"
        except Exception as exc:  # se registra y se sigue; la fila se reintenta al reanudar
            row["error"] = str(exc)
        write_row(row)


def _git_sha() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


async def run(args: argparse.Namespace, client_factory: Callable[[], httpx.AsyncClient] | None = None) -> Path:
    cases = sorted(
        (c for c in load_jsonl(args.dataset_dir / "cases.jsonl") if c["split"] == args.split),
        key=lambda c: c["id"],
    )
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        # Piloto equilibrado: los primeros N casos de cada categoría.
        by_type: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            by_type.setdefault(case["type"], []).append(case)
        cases = [c for group in by_type.values() for c in group[: args.limit]]
    pool = load_jsonl(args.dataset_dir / "distractors.jsonl")
    conditions = [c for c in CONDITIONS if c in set(args.conditions)]
    if "C3" in conditions and not args.memori_url:
        raise SystemExit("C3 necesita --memori-url")
    if any(c != "C3" for c in conditions) and not args.baseline_url:
        raise SystemExit("C0, C1, C2 y C4 necesitan --baseline-url")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = {
        _row_key(r["condition"], r["history_length"], r["run"], r["case_id"])
        for r in load_jsonl(results_path) if not r.get("error")
    }

    factory = client_factory or (lambda: httpx.AsyncClient())
    async with factory() as client:
        health = {}
        for name, url in (("baseline", args.baseline_url), ("memori", args.memori_url)):
            if url:
                response = await client.get(f"{url}/health", timeout=60)
                response.raise_for_status()
                health[name] = response.json()
                if "eval" not in health[name]:
                    raise SystemExit(f"{url} no está en EVAL_MODE=true")
        if "memori" in health and not health["memori"]["eval"]["memori_enabled"]:
            raise SystemExit("--memori-url apunta a un backend sin MEMORI_ENABLED=true")
        if "baseline" in health and health["baseline"]["eval"]["memori_enabled"]:
            raise SystemExit("--baseline-url apunta a un backend con Memori activado")

        manifest = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "git": _git_sha(),
            "dataset_dir": str(args.dataset_dir),
            "cases_sha256": _file_sha(args.dataset_dir / "cases.jsonl"),
            "distractors_sha256": _file_sha(args.dataset_dir / "distractors.jsonl"),
            "split": args.split,
            "cases": len(cases),
            "conditions": conditions,
            "history_lengths": args.history_lengths,
            "runs": args.runs,
            "k": args.k,
            "c2_embedding_model": EMBEDDING_MODEL,
            "max_context_chars": args.max_context_chars,
            "seed": args.seed,
            "health": health,
        }
        manifests = out_dir / "manifests.jsonl"  # uno por ejecución (las reanudaciones se acumulan)
        append_jsonl(manifests, manifest)

        retriever = TurnRetriever()
        semaphore = asyncio.Semaphore(args.concurrency)
        progress = {"done": 0, "errors": 0}

        def write_row(row: dict[str, Any]) -> None:
            append_jsonl(results_path, row)
            progress["done"] += 1
            progress["errors"] += int(bool(row.get("error")))

        async def task(case: dict[str, Any], length: int, run_index: int) -> None:
            pending = [c for c in conditions if _row_key(c, length, run_index, case["id"]) not in done]
            if not pending:
                return
            async with semaphore:
                await _run_case(client, case, pool, length, run_index, pending, retriever, args, write_row)
            print(f"  L={length} run={run_index} {case['id']} ({progress['done']} filas, "
                  f"{progress['errors']} errores)", flush=True)

        for length in args.history_lengths:
            for run_index in range(1, args.runs + 1):
                print(f"Longitud {length}, corrida {run_index}: {len(cases)} casos x {len(conditions)} condiciones")
                await asyncio.gather(*(task(c, length, run_index) for c in cases))

    print(f"Resultados: {results_path} ({progress['done']} filas nuevas, {progress['errors']} con error)")
    if progress["errors"]:
        print("Hay filas con error: vuelve a ejecutar el mismo comando para reintentarlas.")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta las condiciones C0-C4 sobre el dataset v2.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("app/data/eval_dataset"))
    parser.add_argument("--split", choices=["test", "dev", "lme"], default="test",
                        help="lme: casos de import_longmemeval.py (historial fijo; usa --history-lengths 0).")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--history-lengths", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--baseline-url", default=None, help="Backend con MEMORI_ENABLED=false (C0, C1, C2, C4).")
    parser.add_argument("--memori-url", default=None, help="Backend con MEMORI_ENABLED=true (C3).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval_runs/default"))
    parser.add_argument("--entity-prefix", default="eval")
    parser.add_argument("--k", type=int, default=5, help="Top-k de C2; debe coincidir con el de Memori (5).")
    parser.add_argument("--max-context-chars", type=int, default=400_000,
                        help="Límite de C1/C4 en caracteres; se descartan las sesiones más antiguas.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None, help="Solo los primeros N casos de cada categoría.")
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.baseline_url = args.baseline_url.rstrip("/") if args.baseline_url else None
    args.memori_url = args.memori_url.rstrip("/") if args.memori_url else None
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
