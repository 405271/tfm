"""
Prueba de latencia y concurrencia del backend con Memori (H4).

1. Siembra --entities casos de test (historial de --history-length sesiones) en el backend.
2. Calentamiento (--warmup peticiones, no se miden): la primera recuperación carga el
   modelo ONNX de embeddings y tarda varios cientos de ms.
3. Para cada nivel de concurrencia (1, 5, 10, 20) lanza --requests peticiones /chat con
   las preguntas de esos casos, cada una contra su entidad, y registra por petición los
   tiempos por etapa que devuelve EVAL_MODE (recuperación, generación, escritura de memoria)
   y la latencia total vista por el cliente.
4. Calcula p50 y p95 por etapa y nivel, y si la recuperación cumple p95 < --threshold-ms.

Se ejecuta con el backend en exclusiva (sin otras evaluaciones en marcha), dos veces:
con MEMORI_OFFLOAD_SYNC=false (comportamiento original, las llamadas síncronas de Memori
bloquean el event loop) y =true (se llevan a un hilo). Ver scripts/run_load_test.sh.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import instantiate, load_jsonl  # noqa: E402
from run_conditions import _git_sha, _post, _seed_memori  # noqa: E402

STAGES = ("retrieval", "generation", "memory_write", "total")


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    array = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
            "mean": float(array.mean()), "n": int(array.size)}


async def _one(client: httpx.AsyncClient, url: str, target: dict[str, Any], timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        response = await client.post(f"{url}/chat", json={
            "message": target["question"], "session_id": f"{target['entity_id']}-load-{uuid4().hex[:6]}",
            "entity_id": target["entity_id"], "history": [],
        }, timeout=timeout)
        response.raise_for_status()
        timings = (response.json().get("debug") or {}).get("timings_ms") or {}
        return {**{k: timings.get(k) for k in STAGES[:-1]},
                "total": (time.perf_counter() - start) * 1000, "error": None}
    except Exception as exc:
        return {"total": (time.perf_counter() - start) * 1000, "error": str(exc)}


async def _level(client: httpx.AsyncClient, url: str, targets: list[dict[str, Any]], concurrency: int,
                 n_requests: int, timeout: float) -> list[dict[str, Any]]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n_requests):
        queue.put_nowait(i)
    results: list[dict[str, Any]] = []

    async def worker() -> None:
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await _one(client, url, targets[i % len(targets)], timeout))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


async def _memory_failures(client: httpx.AsyncClient, url: str) -> int | None:
    """Fallos de memoria acumulados según /health; None si el servidor no responde (bloqueado)."""
    try:
        health = (await client.get(f"{url}/health", timeout=60)).json()
    except httpx.HTTPError:
        return None
    return sum((health.get("eval") or {}).get("memory_failures", {}).values())


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


async def run(args: argparse.Namespace) -> None:
    cases = sorted((c for c in load_jsonl(args.dataset_dir / "cases.jsonl") if c["split"] == "test"),
                   key=lambda c: c["id"])
    pool = load_jsonl(args.dataset_dir / "distractors.jsonl")
    chosen = cases[:: max(1, len(cases) // args.entities)][: args.entities]
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{args.url}/health", timeout=60)).json()
        if not health.get("eval", {}).get("memori_enabled"):
            raise SystemExit(f"{args.url} debe tener EVAL_MODE=true y MEMORI_ENABLED=true")
        offload = health["eval"].get("memori_offload_sync")
        label = args.label or ("offload" if offload else "sync")

        print(f"Sembrando {len(chosen)} entidades con {args.history_length} sesiones...")
        targets: list[dict[str, Any]] = []
        for case in chosen:
            entity_id = f"load-{label}-{case['id']}-{uuid4().hex[:6]}"
            sessions = instantiate(case, pool, args.history_length, seed=args.seed)
            await _seed_memori(client, args.url, entity_id, sessions, args)
            targets.append({"entity_id": entity_id, "question": case["question"], "case_id": case["id"]})

        print(f"Calentamiento: {args.warmup} peticiones")
        await _level(client, args.url, targets, 1, args.warmup, args.timeout)

        summary: dict[str, Any] = {
            "label": label, "offload_sync": offload, "started_at": datetime.now(timezone.utc).isoformat(),
            "git": _git_sha(), "entities": len(targets), "history_length": args.history_length,
            "requests_per_level": args.requests, "threshold_ms": args.threshold_ms,
            "health": health, "levels": {},
        }
        raw_path = out_dir / f"load_{label}_requests.csv"
        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["concurrency", *STAGES, "error"])
            for concurrency in args.concurrency_levels:
                print(f"Concurrencia {concurrency}: {args.requests} peticiones")
                failures_before = await _memory_failures(client, args.url)
                started = time.perf_counter()
                results = await _level(client, args.url, targets, concurrency, args.requests, args.timeout)
                wall = time.perf_counter() - started
                for r in results:
                    writer.writerow([concurrency, *[r.get(s) for s in STAGES], r.get("error") or ""])
                ok = [r for r in results if not r.get("error")]
                stats = {s: _percentiles([r[s] for r in ok if r.get(s) is not None]) for s in STAGES}
                # Un fallo de memoria (p. ej. pool agotado) no devuelve error HTTP: la petición sigue
                # sin contexto de memoria. Se cuenta aparte y el nivel no cumple si hay alguno.
                failures_after = await _memory_failures(client, args.url)
                blocked = failures_after is None or failures_before is None
                memory_failures = None if blocked else failures_after - failures_before
                retrieval_p95 = stats["retrieval"]["p95"]
                summary["levels"][str(concurrency)] = {
                    **stats, "errors": len(results) - len(ok), "memory_failures": memory_failures,
                    "server_blocked": blocked,
                    "throughput_rps": len(ok) / wall if wall else None,
                    "retrieval_p95_ok": (retrieval_p95 is not None and retrieval_p95 < args.threshold_ms
                                         and not blocked and memory_failures == 0 and len(ok) == len(results)),
                }
                print(f"  recuperación p50 {_fmt(stats['retrieval']['p50'])} ms  p95 {_fmt(retrieval_p95)} ms  "
                      f"| total p95 {_fmt(stats['total']['p95'])} ms | errores {len(results) - len(ok)}  "
                      f"| fallos de memoria {'servidor bloqueado' if blocked else memory_failures}")
                if blocked:
                    print("  El servidor no responde: se detiene la prueba de esta variante.")
                    break

    out = out_dir / f"load_{label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resumen: {out}\nPeticiones: {raw_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de latencia y concurrencia (H4).")
    parser.add_argument("--url", required=True, help="Backend con EVAL_MODE=true y MEMORI_ENABLED=true.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("app/data/eval_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval_runs/load"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--entities", type=int, default=10)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--threshold-ms", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
