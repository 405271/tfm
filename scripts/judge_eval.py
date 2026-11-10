"""
Juez LLM y su validación frente a humanos.

Subcomandos:

  judge      Califica las filas de results.jsonl con uno o varios jueces. Prompts de
             LongMemEval (Wu et al., 2025) traducidos y adaptados al dominio: uno general,
             uno temporal (tolera errores de ±1 en recuentos de días), uno de actualización
             (es correcta si da el valor vigente aunque cite el antiguo) y uno de abstención
             (es correcta si declara que el dato no consta). El juez NO ve la condición
             experimental; el veredicto es binario. Escribe judgments.jsonl (reanudable) y
             verdicts.jsonl con el veredicto final por fila:
               - un juez (por defecto deepseek-v4-pro, como LongMemEval con un único juez
                 validado frente a humanos): su veredicto;
               - varios jueces: el veredicto común, o "pendiente" si discrepan, salvo que
                 --resolutions aporte la resolución manual (CSV row_key,veredicto).

  sample     Muestra estratificada (categoría x condición) de N respuestas juzgadas para
             etiquetado humano doble. El CSV es anónimo: no muestra condición ni veredicto
             del juez. La correspondencia se guarda aparte (human_sample_key.json).

  agreement  Lee el CSV etiquetado (humano_1 y, si hay segundo anotador, humano_2, con si/no) y calcula el
             acuerdo y el kappa de Cohen entre humanos, y de cada juez frente al consenso
             humano. Si algún juez queda por debajo de --min-agreement (90 %), termina con
             código 1: hay que ajustar el prompt y volver a validar antes de usar el juez.

Uso:
  python scripts/judge_eval.py judge --run-dir artifacts/eval_runs/final
  python scripts/judge_eval.py sample --run-dir artifacts/eval_runs/final --size 50
  python scripts/judge_eval.py agreement --run-dir artifacts/eval_runs/final

Variables: JUDGE_MODELS (lista separada por comas; por defecto deepseek:deepseek-v4-pro),
JUDGE_API_KEY (o OPENAI_API_KEY), JUDGE_BASE_URL, JUDGE_TEMPERATURE (por defecto 0; vacía
= no se envía) y JUDGE_REASONING_EFFORT (vacía = no se envía).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import append_jsonl, fold, load_jsonl, write_jsonl  # noqa: E402

load_dotenv()

PROMPT_VERSION = "longmemeval-es-v1"

_BASE = (
    "Te daré una pregunta, la respuesta correcta y la respuesta de un asistente. Responde \"si\" "
    "si la respuesta del asistente contiene la respuesta correcta; en caso contrario, responde \"no\". "
    "Si la respuesta del asistente es equivalente a la respuesta correcta, o contiene todos los pasos "
    "intermedios para llegar a ella, también debes responder \"si\". Si solo contiene una parte de la "
    "información que exige la respuesta correcta, responde \"no\"."
)
_TEMPORAL = (
    " Además, no penalices los errores de uno en el número de días: si la pregunta pide un número de "
    "días, semanas o meses y el asistente se equivoca en uno (por ejemplo, 19 días cuando son 18), la "
    "respuesta sigue siendo correcta."
)
_UPDATE = (
    " Si la respuesta contiene información anterior junto con la respuesta actualizada, debe "
    "considerarse correcta siempre que la respuesta actualizada sea la que se pide."
)
_ABSTENTION = (
    "Te daré una pregunta que no tiene respuesta, una explicación y la respuesta de un asistente. "
    "Responde \"si\" si el asistente identifica correctamente que la pregunta no se puede responder "
    "con la información disponible. El asistente puede decir que la información está incompleta, o "
    "que se dio otra información pero no la que se pregunta. Responde \"no\" si el asistente da como "
    "respuesta un dato concreto como si fuera el pedido."
)
_FORMAT = (
    '\n\nDevuelve solo JSON: {{"veredicto": "si" | "no", "motivo": "una frase"}}.\n\n'
    "Pregunta: {question}\n\n{answer_label}: {reference}\n\nRespuesta del asistente: {reply}"
)


# Prompts originales de LongMemEval (inglés, respuesta yes/no), para la validación externa
# (validación externa con su juez original). Se aplican a las filas del split "lme".
PROMPT_VERSION_ORIGINAL = "longmemeval-original-v1"
_ORIG_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
    "the response only contains a subset of the information required by the answer, answer no."
)
_ORIG_TEMPORAL = (
    " In addition, do not penalize off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct."
)
_ORIG_UPDATE = (
    " If the response contains some previous information along with an updated answer, the response should "
    "be considered as correct as long as the updated answer is the required answer."
)
_ORIG_TAIL = "\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only."
_ORIG_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response from a model. "
    "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
    "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
    "utilizes the user's personal information correctly.\n\nQuestion: {q}\n\nRubric: {a}\n\nModel Response: {r}"
    "\n\nIs the model response correct? Answer yes or no only."
)
_ORIG_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please answer "
    "yes if the model correctly identifies the question as unanswerable. The model could say that the "
    "information is incomplete, or some other information is given but the asked information is not."
    "\n\nQuestion: {q}\n\nExplanation: {a}\n\nModel Response: {r}\n\nDoes the model correctly identify the "
    "question as unanswerable? Answer yes or no only."
)


def is_original_set(row: dict[str, Any]) -> bool:
    return row.get("split") == "lme"


def prompt_version(row: dict[str, Any]) -> str:
    return PROMPT_VERSION_ORIGINAL if is_original_set(row) else PROMPT_VERSION


def original_prompt(row: dict[str, Any]) -> str:
    q, a, r = row["question"], row["reference_answer"], row["reply"]
    if row["type"] == "abstention":
        return _ORIG_ABSTENTION.format(q=q, a=a, r=r)
    question_type = row.get("question_type") or ""
    if question_type == "single-session-preference":
        return _ORIG_PREFERENCE.format(q=q, a=a, r=r)
    body = _ORIG_DEFAULT
    if question_type == "temporal-reasoning":
        body += _ORIG_TEMPORAL
    elif question_type == "knowledge-update":
        body += _ORIG_UPDATE
    return body + _ORIG_TAIL.format(q=q, a=a, r=r)


def judge_prompt(row: dict[str, Any]) -> str:
    if is_original_set(row):
        return original_prompt(row)
    case_type = row["type"]
    if case_type == "abstention":
        instructions, answer_label = _ABSTENTION, "Explicación"
    else:
        instructions, answer_label = _BASE, "Respuesta correcta"
        if case_type == "temporal":
            instructions += _TEMPORAL
        elif case_type == "knowledge_update":
            instructions += _UPDATE
    return instructions + _FORMAT.format(
        question=row["question"], answer_label=answer_label,
        reference=row["reference_answer"], reply=row["reply"],
    )


def row_key(row: dict[str, Any]) -> str:
    return f"{row['condition']}|{row['history_length']}|{row['run']}|{row['case_id']}"


def _parse_verdict(text: str) -> bool:
    try:
        payload = json.loads(text)
        value = fold(str(payload.get("veredicto", "")))
    except (json.JSONDecodeError, AttributeError):
        value = fold(text)[:10]
    if value.startswith("si") or value.startswith("yes"):
        return True
    if value.startswith("no"):
        return False
    raise ValueError(f"veredicto no reconocido: {text[:80]!r}")


def _judge_models(cli: list[str] | None) -> list[str]:
    if cli:
        return cli
    raw = os.getenv("JUDGE_MODELS", "deepseek:deepseek-v4-pro")
    return [m.strip() for m in raw.split(",") if m.strip()]


_PROVIDERS = {
    # proveedor: (variable de la clave, variable de la URL, URL por defecto)
    "openai": ("JUDGE_API_KEY", "JUDGE_BASE_URL", None),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
}


def _client_for(spec: str, client_cls: Any) -> tuple[Any, str]:
    """"proveedor:modelo" (sin prefijo = openai) -> (cliente compatible con OpenAI, modelo)."""
    provider, _, model = spec.partition(":") if ":" in spec else ("openai", "", spec)
    if provider not in _PROVIDERS:
        raise SystemExit(f"Proveedor de juez desconocido: {provider} (usa {', '.join(_PROVIDERS)})")
    key_var, url_var, default_url = _PROVIDERS[provider]
    api_key = os.getenv(key_var) or (os.getenv("OPENAI_API_KEY") if provider == "openai" else None)
    if not api_key:
        raise SystemExit(f"Falta {key_var} para el juez {spec}")
    return client_cls(api_key=api_key, base_url=os.getenv(url_var) or default_url), model


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------

async def _judge_all(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    results = [r for r in load_jsonl(args.run_dir / "results.jsonl") if not r.get("error")]
    # Si una fila se reintentó, vale la última versión.
    latest: dict[str, dict[str, Any]] = {}
    for r in results:
        latest[row_key(r)] = r
    rows = list(latest.values())
    random.Random(args.seed).shuffle(rows)  # orden aleatorio: el juez no ve condiciones agrupadas

    judgments_path = args.run_dir / "judgments.jsonl"
    expected = {row_key(r): prompt_version(r) for r in rows}
    done = {(j["row_key"], j["judge_model"]) for j in load_jsonl(judgments_path)
            if "verdict" in j and j.get("prompt_version") == expected.get(j["row_key"])}

    extra: dict[str, Any] = {}
    temperature = os.getenv("JUDGE_TEMPERATURE", "0").strip()
    if temperature:
        extra["temperature"] = float(temperature)
    # Los modelos de razonamiento (gpt-6.1-terra, deepseek-v4-pro) solo admiten temperatura 0
    # con reasoning_effort=none (o razonan por defecto); por eso es el valor por defecto.
    effort = os.getenv("JUDGE_REASONING_EFFORT", "none").strip()
    if effort:
        extra["reasoning_effort"] = effort

    models = _judge_models(args.judge_models)
    from openai import BadRequestError

    clients = {spec: _client_for(spec, AsyncOpenAI) for spec in models}
    # Comprobación previa por juez: un error de configuración (400) aborta antes de lanzar todo.
    for spec, (client_, model_name) in clients.items():
        try:
            await client_.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": 'Responde solo este json: {"ok": true}'}],
                response_format={"type": "json_object"}, **extra,
            )
        except BadRequestError as exc:
            raise SystemExit(f"Configuración del juez rechazada por {spec} ({extra}): {exc}") from exc
    semaphore = asyncio.Semaphore(args.concurrency)
    stats = Counter()

    async def judge_one(row: dict[str, Any], model: str) -> None:
        prompt = judge_prompt(row)
        # Los prompts originales piden "yes"/"no" en texto plano: sin modo JSON.
        json_mode = {} if is_original_set(row) else {"response_format": {"type": "json_object"}}
        last_error = ""
        for attempt in range(args.retries):
            try:
                client_, model_name = clients[model]
                async with semaphore:
                    resp = await client_.chat.completions.create(
                        model=model_name, messages=[{"role": "user", "content": prompt}],
                        **json_mode, **extra,
                    )
                text = resp.choices[0].message.content or ""
                verdict = _parse_verdict(text)
                try:
                    reason = json.loads(text).get("motivo", "")
                except (json.JSONDecodeError, AttributeError):
                    reason = ""
                append_jsonl(judgments_path, {
                    "row_key": row_key(row), "judge_model": model, "prompt_version": prompt_version(row),
                    "verdict": verdict, "reason": reason,
                    "usage": {"input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                              "output_tokens": resp.usage.completion_tokens if resp.usage else 0},
                })
                stats["ok"] += 1
                return
            except BadRequestError as exc:  # no se arregla reintentando
                last_error = str(exc)
                break
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep(2 ** attempt)
        stats["errors"] += 1
        print(f"  error juzgando {row_key(row)} con {model}: {last_error}", flush=True)

    tasks = [judge_one(r, m) for r in rows for m in models if (row_key(r), m) not in done]
    print(f"Jueces: {models}  filas: {len(rows)}  juicios pendientes: {len(tasks)}")
    await asyncio.gather(*tasks)
    print(f"Juicios nuevos: {stats['ok']}  errores: {stats['errors']}")
    _write_verdicts(args.run_dir, rows, models, args.resolutions)


def _write_verdicts(run_dir: Path, rows: list[dict[str, Any]], models: list[str],
                    resolutions_path: Path | None) -> None:
    by_key: dict[str, dict[str, bool]] = defaultdict(dict)
    reasons: dict[str, dict[str, str]] = defaultdict(dict)
    expected = {row_key(r): prompt_version(r) for r in rows}
    for j in load_jsonl(run_dir / "judgments.jsonl"):
        if j["judge_model"] in models and j.get("prompt_version") == expected.get(j["row_key"]):
            by_key[j["row_key"]][j["judge_model"]] = j["verdict"]
            reasons[j["row_key"]][j["judge_model"]] = j.get("reason", "")
    resolutions: dict[str, bool] = {}
    if resolutions_path is not None and resolutions_path.exists():
        key_path = resolutions_path.with_name(resolutions_path.stem + "_key.json")
        item_map = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else {}
        with resolutions_path.open(encoding="utf-8") as handle:
            for rec in csv.DictReader(handle):
                label = fold(rec.get("veredicto") or "")
                if not label:
                    continue
                key = rec.get("row_key") or item_map.get(rec.get("item", ""))
                if key:
                    resolutions[key] = label.startswith("si")

    verdicts: list[dict[str, Any]] = []
    status = Counter()
    for row in sorted(rows, key=row_key):
        key = row_key(row)
        votes = by_key.get(key, {})
        if len(votes) < len(models):
            final, state = None, "sin_juzgar"
        elif len(set(votes.values())) == 1:
            final, state = next(iter(votes.values())), "acuerdo" if len(models) > 1 else "juez_unico"
        elif key in resolutions:
            final, state = resolutions[key], "resuelto_manual"
        else:
            final, state = None, "pendiente"
        status[state] += 1
        verdicts.append({
            "row_key": key, "condition": row["condition"], "history_length": row["history_length"],
            "run": row["run"], "case_id": row["case_id"], "type": row["type"],
            "correct": final, "status": state, "votes": votes, "auto_correct": row.get("auto_correct"),
            "recall_at_k": row.get("recall_at_k"),
        })
    write_jsonl(run_dir / "verdicts.jsonl", verdicts)
    print(f"Veredictos: {dict(status)} -> {run_dir / 'verdicts.jsonl'}")
    if len(models) > 1:
        disagreements = status["pendiente"] + status["resuelto_manual"]
        total = sum(status.values()) - status["sin_juzgar"]
        print(f"Desacuerdo entre jueces: {disagreements}/{total}")
        if status["pendiente"]:
            # CSV ciego (sin condición); la correspondencia item -> fila va en un fichero aparte.
            pending = [v for v in verdicts if v["status"] == "pendiente"]
            by_row = {row_key(r): r for r in rows}
            path = run_dir / "judge_disagreements.csv"
            if resolutions_path is not None and resolutions_path.resolve() == path.resolve():
                # No sobrescribir el fichero que el usuario está rellenando (perdería sus respuestas).
                print(f"Quedan {status['pendiente']} desacuerdos sin veredicto en {path}; complétalos y repite.")
                return
            item_map: dict[str, str] = {}
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                header = ["item", "tipo", "pregunta", "respuesta_correcta_o_explicacion", "respuesta_del_asistente"]
                for i in range(1, len(models) + 1):
                    header += [f"juez_{i}", f"motivo_juez_{i}"]
                writer.writerow(header + ["veredicto"])
                for index, v in enumerate(pending, start=1):
                    item = f"desacuerdo-{index:03d}"
                    item_map[item] = v["row_key"]
                    row = by_row[v["row_key"]]
                    cells = [item, row["type"], row["question"], row["reference_answer"], row["reply"]]
                    for model in models:
                        cells += ["si" if v["votes"].get(model) else "no", reasons[v["row_key"]].get(model, "")]
                    writer.writerow(cells + [""])
            path.with_name(path.stem + "_key.json").write_text(
                json.dumps(item_map, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Resuelve los pendientes en {path} (columna veredicto: si/no) y pásalo con --resolutions")


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def _sample(args: argparse.Namespace) -> None:
    results = {row_key(r): r for r in load_jsonl(args.run_dir / "results.jsonl") if not r.get("error")}
    verdicts = [v for v in load_jsonl(args.run_dir / "verdicts.jsonl") if v["row_key"] in results]
    if not verdicts:
        raise SystemExit("No hay veredictos; ejecuta primero el subcomando judge.")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for v in verdicts:
        strata[(v["type"], v["condition"])].append(v)
    rng = random.Random(args.seed)
    keys = sorted(strata)
    for key in keys:
        rng.shuffle(strata[key])
    # Reparto por turnos entre estratos hasta llegar al tamaño pedido.
    chosen: list[dict[str, Any]] = []
    while len(chosen) < args.size and any(strata[k] for k in keys):
        for key in keys:
            if strata[key] and len(chosen) < args.size:
                chosen.append(strata[key].pop())
    rng.shuffle(chosen)

    sample_path = args.run_dir / "human_sample.csv"
    key_map: dict[str, str] = {}
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "tipo", "pregunta", "respuesta_correcta_o_explicacion",
                         "respuesta_del_asistente", "humano_1", "humano_2", "comentarios"])
        for index, verdict in enumerate(chosen, start=1):
            row = results[verdict["row_key"]]
            item = f"item-{index:03d}"
            key_map[item] = verdict["row_key"]
            writer.writerow([item, row["type"], row["question"], row["reference_answer"], row["reply"], "", "", ""])
    (args.run_dir / "human_sample_key.json").write_text(
        json.dumps(key_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    by_type = Counter(results[v["row_key"]]["type"] for v in chosen)
    by_condition = Counter(results[v["row_key"]]["condition"] for v in chosen)
    print(f"Muestra de {len(chosen)} respuestas -> {sample_path}")
    print(f"  por categoría: {dict(by_type)}")
    print(f"  por condición: {dict(by_condition)}")
    print("Cada autor rellena su columna (si/no) sin ver la del otro.")


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------

def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def _agreement_stats(a: list[bool], b: list[bool]) -> dict[str, Any]:
    n = len(a)
    return {"n": n, "agreement": (sum(x == y for x, y in zip(a, b)) / n) if n else None,
            "kappa": cohen_kappa(a, b)}


def _parse_label(value: str) -> bool | None:
    folded = fold(value or "")
    if folded.startswith("si") or folded in {"1", "true", "correcto", "yes"}:
        return True
    if folded.startswith("no") or folded in {"0", "false", "incorrecto"}:
        return False
    return None


def _agreement(args: argparse.Namespace) -> None:
    sample_path = args.labels or (args.run_dir / "human_sample.csv")
    key_map = json.loads((args.run_dir / "human_sample_key.json").read_text(encoding="utf-8"))
    judgments: dict[tuple[str, str], bool] = {}
    for j in load_jsonl(args.run_dir / "judgments.jsonl"):
        if j.get("prompt_version") == PROMPT_VERSION:
            judgments[(j["row_key"], j["judge_model"])] = j["verdict"]

    # Con dos anotadores, la referencia es el consenso (los desacuerdos humanos se reportan aparte).
    # Con uno solo (humano_2 vacío en todas las filas), la referencia es humano_1.
    labelled: list[tuple[str, bool, bool | None]] = []
    incomplete = 0
    with sample_path.open(encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            h1, h2 = _parse_label(rec.get("humano_1", "")), _parse_label(rec.get("humano_2", ""))
            if h1 is None:
                incomplete += 1
                continue
            labelled.append((key_map[rec["item"]], h1, h2))
    if not labelled:
        raise SystemExit(f"No hay filas con humano_1 rellena en {sample_path}")
    two_annotators = any(h2 is not None for _, _, h2 in labelled)
    if two_annotators:
        missing = [key for key, _, h2 in labelled if h2 is None]
        if missing:
            raise SystemExit(f"humano_2 está rellena solo en parte ({len(missing)} filas vacías): complétala o vacíala.")

    report: dict[str, Any] = {
        "labels_file": str(sample_path), "labelled": len(labelled), "incomplete": incomplete,
        "annotators": 2 if two_annotators else 1,
        "humans": (_agreement_stats([h1 for _, h1, _ in labelled], [h2 for _, _, h2 in labelled])
                   if two_annotators else None),
    }
    if two_annotators:
        consensus = [(key, h1) for key, h1, h2 in labelled if h1 == h2]
        report["human_disagreements"] = [key for key, h1, h2 in labelled if h1 != h2]
    else:
        consensus = [(key, h1) for key, h1, _ in labelled]
    report["consensus_items"] = len(consensus)

    # Solo los jueces activos (JUDGE_MODELS o --judge-models), aunque judgments.jsonl tenga más.
    active = set(_judge_models(args.judge_models))
    models = sorted({m for (_, m) in judgments if m in active})
    report["judges"] = {}
    failed = False
    for model in models:
        pairs = [(label, judgments[(key, model)]) for key, label in consensus if (key, model) in judgments]
        stats = _agreement_stats([p[0] for p in pairs], [p[1] for p in pairs])
        stats["passes"] = stats["agreement"] is not None and stats["agreement"] >= args.min_agreement
        failed |= not stats["passes"]
        report["judges"][model] = stats

    out = args.run_dir / "judge_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    humans = report["humans"]
    print(f"Etiquetadas: {len(labelled)} (incompletas: {incomplete}); anotadores: {report['annotators']}")
    if humans:
        print(f"Humano-humano: acuerdo {humans['agreement']:.1%}  kappa {humans['kappa']}")
    for model, stats in report["judges"].items():
        mark = "OK" if stats["passes"] else f"NO SUPERA {args.min_agreement:.0%}"
        print(f"{model} vs {'consenso' if humans else 'humano_1'}: acuerdo {stats['agreement']:.1%}  kappa {stats['kappa']}  (n={stats['n']}) {mark}")
    print(f"Informe: {out}")
    if failed:
        print("Ajusta el prompt del juez (y PROMPT_VERSION) y vuelve a validar antes de usarlo en resultados.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Juez LLM y validación frente a humanos.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_judge = sub.add_parser("judge", help="Califica results.jsonl con el juez.")
    p_judge.add_argument("--run-dir", type=Path, required=True)
    p_judge.add_argument("--judge-models", nargs="+", default=None)
    p_judge.add_argument("--concurrency", type=int, default=8)
    p_judge.add_argument("--retries", type=int, default=3)
    p_judge.add_argument("--resolutions", type=Path, default=None)
    p_judge.add_argument("--seed", type=int, default=42)

    p_sample = sub.add_parser("sample", help="Muestra para etiquetado humano.")
    p_sample.add_argument("--run-dir", type=Path, required=True)
    p_sample.add_argument("--size", type=int, default=50)
    p_sample.add_argument("--seed", type=int, default=42)

    p_agree = sub.add_parser("agreement", help="Acuerdo juez-humano y kappa.")
    p_agree.add_argument("--run-dir", type=Path, required=True)
    p_agree.add_argument("--labels", type=Path, default=None)
    p_agree.add_argument("--judge-models", nargs="+", default=None)
    p_agree.add_argument("--min-agreement", type=float, default=0.9)

    args = parser.parse_args()
    if args.command == "judge":
        asyncio.run(_judge_all(args))
    elif args.command == "sample":
        _sample(args)
    else:
        _agreement(args)


if __name__ == "__main__":
    main()
