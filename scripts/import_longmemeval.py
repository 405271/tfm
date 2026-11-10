"""
Importa un subconjunto estratificado de LongMemEval (Wu et al., 2025) para la validación
externa.

Convierte cada pregunta al formato de casos v2 (eval_dataset.py) con un historial FIJO:
  - fixed_sessions: las sesiones del haystack de la pregunta, en orden, con su fecha; las
    de answer_session_ids se marcan como evidencia (sirven para C4, oracle);
  - cada mensaje de usuario lleva delante "[fecha de la sesión]", y la pregunta empieza por
    "Current date: <question_date>", para que todas las condiciones reciban la misma
    información temporal (Memori guarda la fecha de ingesta, no la de la sesión original);
  - type: categoría del dataset propio (extraction, multi_session, temporal, knowledge_update,
    abstention); question_type conserva el tipo original para el juez original.
No hay answer_keywords ni gold_facts: la respuesta es texto libre y la califica el juez con
los prompts originales de LongMemEval (judge_eval.py, prompt set "longmemeval-original").

Uso:
  zstd -d longmemeval_s_cleaned.json.zst -o <dir>/longmemeval_s_cleaned.json
  python scripts/import_longmemeval.py --input <dir>/longmemeval_s_cleaned.json \\
      --output-dir artifacts/eval_lme --per-category 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import CASE_TYPES, write_jsonl  # noqa: E402

_TYPE_MAP = {
    "single-session-user": "extraction",
    "single-session-assistant": "extraction",
    "single-session-preference": "extraction",
    "multi-session": "multi_session",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
}


def category(entry: dict[str, Any]) -> str:
    if str(entry["question_id"]).endswith("_abs"):
        return "abstention"
    return _TYPE_MAP[entry["question_type"]]


def pair_turns(messages: list[dict[str, Any]], date: str) -> list[dict[str, str]]:
    """Agrupa los mensajes en turnos usuario/asistente; rellena huecos para no perder texto."""
    turns: list[dict[str, str]] = []
    pending_user: str | None = None
    for message in messages:
        content = str(message.get("content") or "").strip()
        if message.get("role") == "user":
            if pending_user is not None:
                turns.append({"user": pending_user, "assistant": "(sin respuesta)"})
            pending_user = f"[{date}] {content or '(mensaje vacío)'}"
        else:
            turns.append({"user": pending_user or f"[{date}] (sin mensaje)", "assistant": content or "(sin respuesta)"})
            pending_user = None
    if pending_user is not None:
        turns.append({"user": pending_user, "assistant": "(sin respuesta)"})
    return turns


def convert(entry: dict[str, Any]) -> dict[str, Any]:
    evidence_ids = set(entry.get("answer_session_ids") or [])
    sessions = []
    for index, (session_id, date, messages) in enumerate(
        zip(entry["haystack_session_ids"], entry["haystack_dates"], entry["haystack_sessions"])
    ):
        sessions.append({
            "index": index,
            "kind": "evidence" if session_id in evidence_ids else "distractor",
            "label": session_id,
            "date": date,
            "turns": pair_turns(messages, date),
        })
    return {
        "id": f"lme-{entry['question_id']}",
        "split": "lme",
        "type": category(entry),
        "question_type": entry["question_type"],
        "domain": "longmemeval_s",
        "question": f"Current date: {entry['question_date']}\n{entry['question']}",
        "reference_answer": str(entry["answer"]),
        "answer_keywords": [],
        "forbidden_keywords": [],
        "gold_facts": [],
        "fixed_sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Importa un subconjunto estratificado de LongMemEval.")
    parser.add_argument("--input", type=Path, required=True, help="longmemeval_s_cleaned.json (descomprimido).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval_lme"))
    parser.add_argument("--per-category", type=int, default=12, help="Preguntas por categoría (5 categorías).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    entries = json.loads(raw)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_category.setdefault(category(entry), []).append(entry)

    rng = random.Random(args.seed)
    chosen: list[dict[str, Any]] = []
    for case_type in CASE_TYPES:
        group = sorted(by_category.get(case_type, []), key=lambda e: e["question_id"])
        chosen.extend(rng.sample(group, min(args.per_category, len(group))))
    cases = [convert(e) for e in chosen]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "cases.jsonl", cases)
    manifest = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": args.seed,
        "per_category": args.per_category,
        "question_ids": [e["question_id"] for e in chosen],
        "by_type": dict(Counter(c["type"] for c in cases)),
        "by_question_type": dict(Counter(c["question_type"] for c in cases)),
        "sessions_mean": sum(len(c["fixed_sessions"]) for c in cases) / len(cases),
        "turns_total": sum(len(s["turns"]) for c in cases for s in c["fixed_sessions"]),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "question_ids"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
