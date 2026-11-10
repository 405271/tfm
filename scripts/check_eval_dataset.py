"""
Control de calidad del dataset de evaluación de memoria.

Comprueba:
  1. Esquema y reglas de cada caso (eval_dataset.validate_case) y de cada distractor.
  2. Reparto por split, categoría e intención.
  3. Diversidad: similitud coseno entre escenarios (umbral < 0.4) y su distribución.
     Representación por defecto: TF-IDF sin acentos y sin palabras vacías. Con
     --embeddings openai se usan embeddings densos (EMBEDDINGS_MODEL o
     text-embedding-3-small); hay que indicar cuál se usó al dar
     los resultados, porque los embeddings densos dan similitudes mucho más altas que TF-IDF.
  4. Que cada caso se pueda instanciar con 5, 20 y 50 sesiones con el pool de distractores.
  5. Muestra estratificada del 20 % para revisarla a mano (CSV).

Escribe dataset_report.json y review_sample.csv junto al dataset y termina con código 1
si hay casos inválidos o pares por encima del umbral; los ids a regenerar se imprimen
listos para `generate_eval_dataset.py --regenerate`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import (  # noqa: E402
    CASE_TYPES,
    HISTORY_LENGTHS,
    instantiate,
    load_jsonl,
    scenario_text,
    validate_case,
    validate_distractor,
)

load_dotenv()

_SPANISH_STOPWORDS = """
a al algo algun alguna algunas alguno algunos ante antes aqui asi aun bien cada como con contra cual cuales
cuando de del desde donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba
estan estar este esto estos fue fueron ha habia han hasta hay la las le les lo los mas me mi mis mucho muy
nada ni no nos nosotros o os otra otras otro otros para pero poco por porque que quien se sea segun ser si
sin sobre son su sus tambien tan te tener tengo ti tiene tienen todo todos tu tus un una unas uno unos usted
vosotros y ya yo hola gracias buenas buenos dias perfecto vale gustaria quisiera querria favor puedo puede
""".split()


def _similarity_matrix(texts: list[str], method: str) -> np.ndarray:
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=_SPANISH_STOPWORDS)
        matrix = vectorizer.fit_transform(texts)
        return (matrix @ matrix.T).toarray()

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("EMBEDDINGS_BASE_URL") or None)
    model = os.getenv("EMBEDDINGS_MODEL") or "text-embedding-3-small"
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        resp = client.embeddings.create(model=model, input=texts[start:start + 64])
        vectors.extend(item.embedding for item in resp.data)
    array = np.asarray(vectors, dtype=np.float32)
    array /= np.linalg.norm(array, axis=1, keepdims=True)
    return array @ array.T


def _to_regenerate(pairs: list[tuple[str, str, float]]) -> list[str]:
    """Cobertura voraz: regenera el caso que aparece en más pares por encima del umbral."""
    remaining = list(pairs)
    chosen: list[str] = []
    while remaining:
        counts = Counter([a for a, _, _ in remaining] + [b for _, b, _ in remaining])
        worst = counts.most_common(1)[0][0]
        chosen.append(worst)
        remaining = [p for p in remaining if worst not in (p[0], p[1])]
    return sorted(chosen)


def _review_sample(cases: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for case_type in CASE_TYPES:
        group = [c for c in cases if c["type"] == case_type]
        size = max(1, round(len(group) * fraction)) if group else 0
        sample.extend(rng.sample(group, size))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Control de calidad del dataset de evaluación.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("app/data/eval_dataset"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--embeddings", choices=["tfidf", "openai"], default="tfidf")
    parser.add_argument("--min-turns", type=int, default=10)
    parser.add_argument("--review-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = load_jsonl(args.dataset_dir / "cases.jsonl")
    distractors = load_jsonl(args.dataset_dir / "distractors.jsonl")
    if not cases:
        raise SystemExit(f"No hay casos en {args.dataset_dir / 'cases.jsonl'}")

    # 1. Validez
    invalid = {c["id"]: errs for c in cases if (errs := validate_case(c, min_turns=args.min_turns))}
    invalid_distractors = {
        d["id"]: errs for d in distractors if (errs := validate_distractor(d, min_turns=args.min_turns))
    }
    duplicated_ids = [i for i, n in Counter(c["id"] for c in cases).items() if n > 1]

    # 2. Reparto
    by_split_type = Counter(f"{c['split']}/{c['type']}" for c in cases)
    by_intent = Counter(c.get("intent", "?") for c in cases)

    # 3. Diversidad
    sims = _similarity_matrix([scenario_text(c) for c in cases], args.embeddings)
    upper = sims[np.triu_indices(len(cases), k=1)]
    over = [
        (cases[i]["id"], cases[j]["id"], float(sims[i, j]))
        for i in range(len(cases)) for j in range(i + 1, len(cases))
        if sims[i, j] >= args.threshold
    ]
    over.sort(key=lambda p: p[2], reverse=True)
    distribution = {
        "pairs": int(upper.size),
        "mean": float(upper.mean()) if upper.size else 0.0,
        "p50": float(np.percentile(upper, 50)) if upper.size else 0.0,
        "p90": float(np.percentile(upper, 90)) if upper.size else 0.0,
        "p95": float(np.percentile(upper, 95)) if upper.size else 0.0,
        "max": float(upper.max()) if upper.size else 0.0,
        "over_threshold": len(over),
    }

    # 4. Instanciación
    instantiation_errors: dict[str, str] = {}
    for case in cases:
        if case["id"] in invalid:
            continue
        for length in HISTORY_LENGTHS:
            try:
                instantiate(case, distractors, length, seed=args.seed)
            except ValueError as exc:
                instantiation_errors[f"{case['id']}@{length}"] = str(exc)

    # 5. Muestra para revisión manual
    sample = _review_sample(cases, args.review_fraction, args.seed)
    review_path = args.dataset_dir / "review_sample.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "type", "intent", "question", "reference_answer", "answer_keywords",
                         "forbidden_keywords", "revisor_1_ok", "revisor_2_ok", "comentarios"])
        for case in sample:
            writer.writerow([case["id"], case["type"], case.get("intent", ""), case["question"],
                             case["reference_answer"], " | ".join(case.get("answer_keywords") or []),
                             " | ".join(case.get("forbidden_keywords") or []), "", "", ""])

    regenerate = sorted(set(_to_regenerate(over)) | set(invalid))
    report = {
        "cases": len(cases),
        "distractors": len(distractors),
        "by_split_type": dict(sorted(by_split_type.items())),
        "by_intent": dict(sorted(by_intent.items())),
        "invalid_cases": invalid,
        "invalid_distractors": invalid_distractors,
        "duplicated_ids": duplicated_ids,
        "similarity": {"method": args.embeddings, "threshold": args.threshold, **distribution,
                       "top_pairs": over[:20]},
        "instantiation_errors": instantiation_errors,
        "review_sample": {"path": str(review_path), "size": len(sample)},
        "regenerate": regenerate,
    }
    (args.dataset_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Casos: {len(cases)}  distractores: {len(distractors)}")
    for key, value in sorted(by_split_type.items()):
        print(f"  {key}: {value}")
    print(f"Intenciones cubiertas: {len(by_intent)}")
    print(f"Inválidos: {len(invalid)}  distractores inválidos: {len(invalid_distractors)}  "
          f"ids duplicados: {len(duplicated_ids)}")
    print(f"Similitud ({args.embeddings}): media {distribution['mean']:.3f}  p50 {distribution['p50']:.3f}  "
          f"p95 {distribution['p95']:.3f}  máx {distribution['max']:.3f}  "
          f"pares >= {args.threshold}: {len(over)}/{distribution['pairs']}")
    print(f"Errores de instanciación: {len(instantiation_errors)}")
    print(f"Muestra de revisión: {len(sample)} casos -> {review_path}")
    if regenerate:
        print("Regenerar con:")
        print("  python scripts/generate_eval_dataset.py --regenerate " + " ".join(regenerate))

    failed = bool(invalid or invalid_distractors or duplicated_ids or over or instantiation_errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
