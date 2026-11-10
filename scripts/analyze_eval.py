"""
Análisis estadístico de la evaluación y generación del informe.

Entradas (las que falten se reportan como "no evaluado"):
  --run-dir   results.jsonl + verdicts.jsonl (juez) de run_conditions.py / judge_eval.py
  --sim-dir   conversations.jsonl de simulate_user.py (RR y STR)
  --load-dir  load_*.json de load_test.py (latencia, H4)
  --dataset-dir dataset_report.json (diversidad de escenarios)
  judge_validation.json en --run-dir (acuerdo juez-humano)

Unidad de análisis: el caso. Con varias corridas, el acierto de un caso en una condición es
el voto mayoritario de sus corridas (empate = fallo); la media y la desviación entre
corridas se reportan aparte. Pruebas:
  - McNemar exacto (binomial sobre discordantes) y bootstrap pareado (10 000) para la
    diferencia de proporciones, con Holm sobre las comparaciones de C3;
  - bootstrap pareado de la reducción relativa de RR;
  - H2 sobre multi-sesión y actualización;
  - H3: diferencia C3-C1 por longitud con margen de equivalencia declarado.

Escribe en <run-dir>/report/: report.md, summary.json y tablas CSV.
--source auto usa la comprobación por palabras clave en lugar del juez (solo pilotos).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import CASE_TYPE_LABELS, CASE_TYPES, load_jsonl  # noqa: E402

CONDITION_LABELS = {"C0": "C0 Stateless", "C1": "C1 Contexto completo", "C2": "C2 RAG simple",
                    "C3": "C3 Memori", "C4": "C4 Oracle"}
CONDITIONS = tuple(CONDITION_LABELS)
BOOTSTRAP = 10_000


# ---------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------

def mcnemar_exact(a: list[bool], b: list[bool]) -> float:
    """p bilateral exacto: binomial sobre los pares discordantes."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    return float(stats.binomtest(only_a, n, 0.5).pvalue)


def bootstrap_diff(a: Iterable[float], b: Iterable[float], seed: int = 42) -> tuple[float, float]:
    """IC 95 % percentil de mean(a) - mean(b) remuestreando pares."""
    diffs = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    if diffs.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(BOOTSTRAP, diffs.size), replace=True).mean(axis=1)
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def bootstrap_relative_reduction(base: list[float], cand: list[float], seed: int = 42) -> tuple[float, float]:
    """IC 95 % de 1 - mean(cand)/mean(base), remuestreando unidades pareadas."""
    base_arr, cand_arr = np.asarray(base, dtype=float), np.asarray(cand, dtype=float)
    if base_arr.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, base_arr.size, size=(BOOTSTRAP, base_arr.size))
    base_means = base_arr[idx].mean(axis=1)
    cand_means = cand_arr[idx].mean(axis=1)
    valid = base_means > 0
    reductions = 1 - cand_means[valid] / base_means[valid]
    if reductions.size == 0:
        return (math.nan, math.nan)
    return (float(np.percentile(reductions, 2.5)), float(np.percentile(reductions, 97.5)))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def length_label(length: int) -> str:
    """0 = historial fijo de LongMemEval-S (cada pregunta trae el suyo, ~48 sesiones)."""
    return "LongMemEval-S" if length == 0 else str(length)


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}"


def fmt_p(p: float | None) -> str:
    if p is None or math.isnan(p):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _row_key(r: dict[str, Any]) -> str:
    return f"{r['condition']}|{r['history_length']}|{r['run']}|{r['case_id']}"


def load_accuracy(run_dir: Path, source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = {}
    for r in load_jsonl(run_dir / "results.jsonl"):
        if not r.get("error"):
            results[_row_key(r)] = r  # la última versión de cada fila
    info: dict[str, Any] = {"source": source, "rows": len(results)}
    if source == "judge":
        verdicts = {v["row_key"]: v for v in load_jsonl(run_dir / "verdicts.jsonl")}
        if not verdicts:
            raise SystemExit("No hay verdicts.jsonl: ejecuta judge_eval.py judge o usa --source auto (piloto).")
        pending = sum(1 for k in results if verdicts.get(k, {}).get("correct") is None)
        info["unjudged_or_pending"] = pending
        rows = []
        for key, r in results.items():
            correct = verdicts.get(key, {}).get("correct")
            if correct is not None:
                rows.append({**r, "correct": bool(correct)})
    else:
        rows = [{**r, "correct": bool(r.get("auto_correct"))} for r in results.values()]
    return rows, info


def per_case(rows: list[dict[str, Any]], field: str = "correct") -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    """{(condición, longitud): {case_id: {"value": voto mayoritario, "runs": [...], "type": ...}}}"""
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get(field) is None:
            continue
        slot = grouped[(r["condition"], r["history_length"])].setdefault(
            r["case_id"], {"runs": {}, "type": r["type"]}
        )
        slot["runs"][r["run"]] = bool(r[field])
    for cases in grouped.values():
        for slot in cases.values():
            values = list(slot["runs"].values())
            slot["value"] = sum(values) > len(values) / 2
    return grouped


# ---------------------------------------------------------------------------
# Secciones del informe
# ---------------------------------------------------------------------------

def accuracy_tables(rows: list[dict[str, Any]], main_length: int, lengths: list[int],
                    conditions: list[str], margin_pp: float, alpha: float) -> tuple[list[str], dict[str, Any]]:
    md: list[str] = []
    out: dict[str, Any] = {}
    cases = per_case(rows)
    runs = sorted({r["run"] for r in rows})

    # Exactitud por categoría
    md.append(f"## Exactitud de recuerdo\n")
    caption = (f"Exactitud (%) por categoría con historial de {main_length} sesiones, "
               if main_length else
               "**Validación externa.** Exactitud (%) por categoría con el historial completo de LongMemEval-S, ")
    md.append(caption + f"voto mayoritario de {len(runs)} corrida(s) por caso.\n")
    md.append("| Categoría | " + " | ".join(CONDITION_LABELS[c] for c in conditions) + " |")
    md.append("| --- | " + " | ".join("---" for _ in conditions) + " |")
    table: dict[str, dict[str, Any]] = {}
    for case_type in CASE_TYPES:
        cells = []
        for c in conditions:
            values = [s["value"] for s in cases.get((c, main_length), {}).values() if s["type"] == case_type]
            acc = sum(values) / len(values) if values else None
            table.setdefault(case_type, {})[c] = {"accuracy": acc, "n": len(values)}
            cells.append(f"{pct(acc, 0)} (n={len(values)})" if values else "—")
        md.append(f"| {CASE_TYPE_LABELS[case_type]} | " + " | ".join(cells) + " |")
    global_cells, global_stats = [], {}
    for c in conditions:
        values = [s["value"] for s in cases.get((c, main_length), {}).values()]
        k, n = sum(values), len(values)
        global_stats[c] = {"accuracy": k / n if n else None, "n": n}
        global_cells.append(f"**{pct(k / n if n else None)}**")
    md.append(f"| **Global** | " + " | ".join(global_cells) + " |\n")
    if any(table[t][c]["n"] < 20 for t in table for c in conditions if table[t][c]["n"]):
        md.append("Con menos de 20 casos por categoría, los resultados por categoría son descriptivos.\n")

    # Variación entre corridas
    if len(runs) > 1:
        md.append("**Variación entre corridas** (exactitud global por corrida, media ± DE):\n")
        md.append("| Condición | " + " | ".join(f"Corrida {r}" for r in runs) + " | Media ± DE |")
        md.append("| --- | " + " | ".join("---" for _ in runs) + " | --- |")
        variation = {}
        for c in conditions:
            per_run = []
            for run in runs:
                vals = [r["correct"] for r in rows if r["condition"] == c and r["history_length"] == main_length and r["run"] == run]
                per_run.append(sum(vals) / len(vals) if vals else math.nan)
            arr = np.asarray(per_run)
            variation[c] = {"per_run": per_run, "mean": float(np.nanmean(arr)), "sd": float(np.nanstd(arr, ddof=1))}
            md.append(f"| {c} | " + " | ".join(pct(v) for v in per_run) +
                      f" | {pct(variation[c]['mean'])} ± {pct(variation[c]['sd'])} |")
        md.append("")
        out["run_variation"] = variation

    # Comparaciones pareadas de C3
    def paired(c_a: str, c_b: str, length: int, types: set[str] | None = None) -> dict[str, Any] | None:
        a_cases, b_cases = cases.get((c_a, length), {}), cases.get((c_b, length), {})
        common = sorted(k for k in a_cases if k in b_cases and (types is None or a_cases[k]["type"] in types))
        if not common:
            return None
        a = [a_cases[k]["value"] for k in common]
        b = [b_cases[k]["value"] for k in common]
        lo, hi = bootstrap_diff([float(x) for x in a], [float(x) for x in b])
        return {"n": len(common), "diff": sum(a) / len(a) - sum(b) / len(b), "ci": [lo, hi],
                "p": mcnemar_exact(a, b), "acc_a": sum(a) / len(a), "acc_b": sum(b) / len(b)}

    comparisons = [c for c in ("C0", "C2", "C1") if c in conditions and "C3" in conditions]
    results = {c: paired("C3", c, main_length) for c in comparisons}
    valid = [c for c in comparisons if results[c]]
    adjusted = holm([results[c]["p"] for c in valid])
    caption = (f"Comparaciones pareadas de C3 con {main_length} sesiones "
               if main_length else "**Validación externa.** Comparaciones pareadas de C3 en LongMemEval-S ")
    md.append(caption + "(McNemar exacto, bootstrap pareado, Holm).\n")
    md.append("| Comparación | n | Diferencia (pp) | IC 95 % | p | p ajustado (Holm) |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for c, p_adj in zip(valid, adjusted):
        r = results[c]
        r["p_holm"] = p_adj
        md.append(f"| C3 contra {c} | {r['n']} | {100 * r['diff']:+.1f} | [{100 * r['ci'][0]:+.1f}, "
                  f"{100 * r['ci'][1]:+.1f}] | {fmt_p(r['p'])} | {fmt_p(p_adj)} |")
    md.append("")
    out["comparisons"] = results

    # H2
    h2 = paired("C3", "C0", main_length, {"multi_session", "knowledge_update"}) if "C0" in conditions else None
    if h2:
        md.append(f"**H2** (multi-sesión y actualización, C3 contra C0): {100 * h2['diff']:+.1f} pp, n = {h2['n']}.\n")
    out["h2"] = {k: h2[k] for k in ("diff", "n", "acc_a", "acc_b")} if h2 else None

    # H3: C3 contra C1 por longitud
    h3 = {}
    if "C1" in conditions and "C3" in conditions:
        md.append(f"**H3** (C3 contra C1 por longitud; margen de equivalencia ±{margin_pp:.0f} pp):\n")
        md.append("| Sesiones | Diferencia (pp) | IC 95 % | p | Lectura |")
        md.append("| --- | --- | --- | --- | --- |")
        for length in lengths:
            r = paired("C3", "C1", length)
            if not r:
                continue
            lo, hi = 100 * r["ci"][0], 100 * r["ci"][1]
            if r["p"] < alpha and r["diff"] > 0:
                reading = "Ventaja de C3"
            elif r["p"] < alpha and r["diff"] < 0:
                reading = "Ventaja de C1"
            elif -margin_pp <= lo and hi <= margin_pp:
                reading = "Equivalentes (IC dentro del margen)"
            else:
                reading = "Sin diferencia detectada (no concluyente)"
            r["reading"] = reading
            h3[str(length)] = r
            md.append(f"| {length} | {100 * r['diff']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {fmt_p(r['p'])} | {reading} |")
        md.append("")
    out.update(table=table, global_=global_stats, h3=h3)
    return md, out


def recall_and_tokens(rows: list[dict[str, Any]], lengths: list[int], conditions: list[str],
                      k: int) -> tuple[list[str], dict[str, Any]]:
    md: list[str] = []
    out: dict[str, Any] = {"recall": {}, "tokens": {}}
    recall_cases = per_case([r for r in rows if r.get("recall_at_k") is not None], "recall_at_k")
    md.append(f"**Recall@{k} de memoria** (casos no de abstención, voto mayoritario):\n")
    md.append("| Sesiones | " + " | ".join(c for c in ("C2", "C3") if c in conditions) + " |")
    md.append("| --- | " + " | ".join("---" for c in ("C2", "C3") if c in conditions) + " |")
    for length in lengths:
        cells = []
        for c in ("C2", "C3"):
            if c not in conditions:
                continue
            values = [s["value"] for s in recall_cases.get((c, length), {}).values()]
            acc = sum(values) / len(values) if values else None
            out["recall"].setdefault(c, {})[str(length)] = {"recall": acc, "n": len(values)}
            cells.append(f"{pct(acc)} (n={len(values)})" if values else "—")
        md.append(f"| {length_label(length)} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("**Tokens de entrada por pregunta** (media; en C3, coste de escritura de memoria por turno "
              "sembrado aparte):\n")
    md.append("| Sesiones | " + " | ".join(conditions) + " | C3 escritura/turno |")
    md.append("| --- | " + " | ".join("---" for _ in conditions) + " | --- |")
    for length in lengths:
        cells = []
        for c in conditions:
            us = [r["usage"] for r in rows if r["condition"] == c and r["history_length"] == length and r.get("usage")]
            if not us:
                cells.append("—")
                continue
            tin = float(np.mean([u["input_tokens"] for u in us]))
            out["tokens"].setdefault(c, {})[str(length)] = {"input": tin}
            cells.append(f"{tin:,.0f}".replace(",", " "))
        seeding = [r["seeding"] for r in rows if r["condition"] == "C3" and r["history_length"] == length and r.get("seeding")]
        write = (sum(s["input_tokens"] + s["output_tokens"] for s in seeding) / max(1, sum(s["turns"] for s in seeding))) if seeding else None
        out["tokens"].setdefault("C3_write_per_turn", {})[str(length)] = write
        md.append(f"| {length_label(length)} | " + " | ".join(cells) + f" | {write:,.0f} |".replace(",", " ") if write else
                  f"| {length_label(length)} | " + " | ".join(cells) + " | — |")
    md.append("")
    return md, out


def simulation_section(sim_dir: Path | None) -> tuple[list[str], dict[str, Any] | None]:
    if sim_dir is None or not (sim_dir / "conversations.jsonl").exists():
        return ["## Repetición y pasos para la resolución\n", "No evaluado (falta conversations.jsonl).\n"], None
    convs = [c for c in load_jsonl(sim_dir / "conversations.jsonl") if not c.get("error")]
    latest = {(c["condition"], c["run"], c["case_id"]): c for c in convs}
    convs = list(latest.values())
    conditions = [c for c in CONDITIONS if any(x["condition"] == c for x in convs)]
    tasks = sorted({c["case_id"] for c in convs})
    by_task: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for c in convs:
        slot = by_task[c["case_id"]][c["condition"]]
        slot["rr"].append(c["rr"])
        slot["str"].append(c["str_turns"])
        slot["completed"].append(float(c["completed"]))
        slot["repeated"].append(c["repeated_turns"])
        slot["turns"].append(c["assistant_turns"])
    md = ["## Repetición y pasos para la resolución\n",
          f"Usuario simulado, {len(tasks)} tareas, historial de {convs[0]['history_length']} sesiones, "
          f"{len({c['run'] for c in convs})} corrida(s).\n",
          "| Métrica | " + " | ".join(conditions) + " | Reducción de C3 frente a C0 |",
          "| --- | " + " | ".join("---" for _ in conditions) + " | --- |"]
    out: dict[str, Any] = {"tasks": len(tasks), "conditions": {}}
    for c in conditions:
        rel = [c2 for c2 in convs if c2["condition"] == c]
        out["conditions"][c] = {
            "rr_pooled": sum(x["repeated_turns"] for x in rel) / max(1, sum(x["assistant_turns"] for x in rel)),
            "rr_mean": float(np.mean([x["rr"] for x in rel])),
            "str_mean": float(np.mean([x["str_turns"] for x in rel])),
            "completion": float(np.mean([x["completed"] for x in rel])),
        }

    def task_means(cond: str, metric: str) -> tuple[list[float], list[float]]:
        common = [t for t in tasks if by_task[t]["C0"][metric] and by_task[t][cond][metric]]
        return ([float(np.mean(by_task[t]["C0"][metric])) for t in common],
                [float(np.mean(by_task[t][cond][metric])) for t in common])

    reductions: dict[str, Any] = {}
    if "C0" in conditions and "C3" in conditions:
        base, cand = task_means("C3", "rr")
        rr_red = 1 - (np.mean(cand) / np.mean(base)) if np.mean(base) > 0 else math.nan
        reductions["rr"] = {"relative": float(rr_red), "ci": bootstrap_relative_reduction(base, cand)}
        base_s, cand_s = task_means("C3", "str")
        str_red = 1 - np.mean(cand_s) / np.mean(base_s)
        reductions["str"] = {"relative": float(str_red)}
    rr_cells = [pct(out["conditions"][c]["rr_pooled"]) for c in conditions]
    str_cells = [f"{out['conditions'][c]['str_mean']:.1f}" for c in conditions]
    comp_cells = [pct(out["conditions"][c]["completion"], 0) for c in conditions]
    rr_red_txt = str_red_txt = "—"
    if reductions:
        r = reductions["rr"]
        rr_red_txt = f"{pct(r['relative'])} % relativa, IC 95 % [{pct(r['ci'][0], 0)}, {pct(r['ci'][1], 0)}]"
        s = reductions["str"]
        str_red_txt = f"{pct(s['relative'])} % relativa"
    md.append("| Tasa de repetición (RR, %) | " + " | ".join(rr_cells) + f" | {rr_red_txt} |")
    md.append("| Pasos para la resolución (STR, media) | " + " | ".join(str_cells) + f" | {str_red_txt} |")
    md.append("| Tareas completadas (%) | " + " | ".join(comp_cells) + " | |\n")
    md.append("Las conversaciones no completadas cuentan con el máximo de turnos (censuradas).\n")
    out["reductions"] = reductions
    return md, out


def latency_section(load_dir: Path | None, threshold_ms: float) -> tuple[list[str], dict[str, Any] | None]:
    files = sorted(load_dir.glob("load_*.json")) if load_dir and load_dir.exists() else []
    files = [f for f in files if not f.name.endswith("_requests.json")]
    if not files:
        return ["## Latencia y costo\n", "Latencia no evaluada (faltan load_*.json).\n"], None
    md = ["## Latencia y costo\n"]
    out: dict[str, Any] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        label = data["label"]
        out[label] = data
        md.append(f"**Variante {label}.** Latencia por etapa con Memori, {data['requests_per_level']} peticiones "
                  f"por nivel, {data['entities']} usuarios con {data['history_length']} sesiones "
                  f"(MEMORI_OFFLOAD_SYNC={data['offload_sync']}).\n")
        md.append("| Concurrencia | Recuperación p95 | Generación p95 | Escritura p95 | "
                  f"Total p95 | Errores | Fallos de memoria | ¿Recuperación p95 < {threshold_ms:.0f} ms? |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for level, s in data["levels"].items():
            f = lambda st, q: "—" if s[st][q] is None else f"{s[st][q]:.0f}"  # noqa: E731
            md.append(f"| {level} | {f('retrieval', 'p95')} | {f('generation', 'p95')} | "
                      f"{f('memory_write', 'p95')} | {f('total', 'p95')} | {s['errors']} | "
                      f"{'servidor bloqueado' if s.get('server_blocked') else s.get('memory_failures', '—')} | "
                      f"{'Sí' if s['retrieval_p95_ok'] else 'No'} |")
        md.append("")
    return md, out


def lme_section(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Validación externa con LongMemEval-S: exactitud global y diferencia C3 frente a C1."""
    cases = per_case(rows)
    conditions = [c for c in CONDITIONS if any(cond == c for (cond, _) in cases)]
    by_cond = {c: [v for (cond, _), cs in cases.items() if cond == c for v in cs.values()] for c in conditions}
    md = ["## Validación externa (LongMemEval-S)\n",
          f"Exactitud (%) en el subconjunto de LongMemEval-S, una corrida, "
          f"n = {max(len(v) for v in by_cond.values())} preguntas.\n",
          "| Medida | " + " | ".join(CONDITION_LABELS[c] for c in conditions) + " |",
          "| --- | " + " | ".join("---" for _ in conditions) + " |"]
    out: dict[str, Any] = {"global": {}}
    cells_g = []
    for c in conditions:
        vals = [v["value"] for v in by_cond[c]]
        k, n = sum(vals), len(vals)
        out["global"][c] = {"accuracy": k / n if n else None, "n": n}
        cells_g.append(pct(k / n if n else None))
    md.append("| Exactitud (%) | " + " | ".join(cells_g) + " |\n")
    if "C3" in by_cond and "C1" in by_cond:
        c3, c1 = cases.get(("C3", 0), {}), cases.get(("C1", 0), {})
        common = sorted(set(c3) & set(c1))
        if common:
            x, y = [c3[k]["value"] for k in common], [c1[k]["value"] for k in common]
            out["c3_vs_c1"] = {"n": len(common), "diff": (sum(x) - sum(y)) / len(common)}
            md.append(f"C3 contra C1: {100 * out['c3_vs_c1']['diff']:+.1f} pp, n = {len(common)}.\n")
    return md, out


def objectives_section(acc: dict[str, Any] | None, rec: dict[str, Any] | None, sim: dict[str, Any] | None,
                       load: dict[str, Any] | None, dataset_report: dict[str, Any] | None,
                       judge_validation: dict[str, Any] | None, th: dict[str, Any], main_length: int,
                       lengths: list[int], lme: dict[str, Any] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    alpha = th["alpha"]
    rows: list[dict[str, str]] = []

    def add(objective: str, result: str, status: str) -> None:
        rows.append({"objetivo": objective, "resultado": result, "estado": status})

    comp = (acc or {}).get("comparisons", {})
    c0 = comp.get("C0")
    if c0:
        status = "Cumplido" if c0["p_holm"] < alpha and c0["diff"] > 0 else "No cumplido"
        add("Exactitud mayor que stateless (H2 global)",
            f"{100 * c0['diff']:+.1f} pp (p Holm {fmt_p(c0['p_holm'])})", status)
    else:
        add("Exactitud mayor que stateless (H2 global)", "Sin datos", "No evaluado")
    h3 = (acc or {}).get("h3", {})
    if h3:
        wins = [L for L, r in h3.items() if r["reading"] == "Ventaja de C3"]
        detail = "; ".join(f"{L} sesiones: {100 * r['diff']:+.1f} pp ({r['reading'][0].lower() + r['reading'][1:]})" for L, r in h3.items())
        status = "Cumplido" if wins and len(wins) == len(h3) else ("Parcial" if wins else "No cumplido")
        add("Ventaja frente a contexto completo (H3)", detail, status)
    else:
        add("Ventaja frente a contexto completo (H3)", "Sin datos", "No evaluado")
    reds = (sim or {}).get("reductions") or {}
    if reds:
        rr = reds["rr"]["relative"]
        lo_e, hi_e = th["rr_reduction_expected"]
        if rr >= lo_e:
            status = "Cumplido (dentro o por encima del valor esperado)"
        elif rr >= th["rr_reduction_min"]:
            status = f"Cumplido el mínimo; el valor esperado de {lo_e * 100:.0f}-{hi_e * 100:.0f} % no se alcanzó"
        else:
            status = "No cumplido"
        add(f"Reducción de RR ≥ {th['rr_reduction_min'] * 100:.0f} % (H1)", f"{pct(rr)} %", status)
        st_red = reds["str"]["relative"]
        add(f"Reducción de STR ≥ {th['str_reduction_min'] * 100:.0f} %", f"{pct(st_red)} %",
            "Cumplido" if st_red >= th["str_reduction_min"] else "No cumplido")
    else:
        add(f"Reducción de RR ≥ {th['rr_reduction_min'] * 100:.0f} % (H1)", "Sin datos", "No evaluado")
        add(f"Reducción de STR ≥ {th['str_reduction_min'] * 100:.0f} %", "Sin datos", "No evaluado")
    recall = ((rec or {}).get("recall") or {}).get("C3", {}).get(str(main_length))
    if recall and recall["recall"] is not None:
        add(f"Recall@{th['recall_k']} ≥ {th['recall_at_k_min'] * 100:.0f} %",
            f"{pct(recall['recall'])} % ({main_length} sesiones)",
            "Cumplido" if recall["recall"] >= th["recall_at_k_min"] else "No cumplido")
    else:
        add(f"Recall@{th['recall_k']} ≥ {th['recall_at_k_min'] * 100:.0f} %", "Sin datos", "No evaluado")
    if load:
        parts, statuses = [], []
        for label, data in load.items():
            ok = [lv for lv, s in data["levels"].items() if s["retrieval_p95_ok"]]
            parts.append(f"{label}: cumple en concurrencia {', '.join(ok) if ok else 'ninguna'}")
            statuses.append("all" if len(ok) == len(data["levels"]) else ("some" if ok else "none"))
        status = "Cumplido" if "all" in statuses else ("Parcial" if "some" in statuses else "No cumplido")
        add(f"Latencia p95 de recuperación < {th['retrieval_p95_ms_max']} ms (H4)", "; ".join(parts), status)
    else:
        add(f"Latencia p95 de recuperación < {th['retrieval_p95_ms_max']} ms (H4)", "Sin datos", "No evaluado")
    if dataset_report:
        sim_info = dataset_report["similarity"]
        add(f"Similitud entre escenarios < {th['scenario_similarity_max']}",
            f"Máximo {sim_info['max']:.2f} ({sim_info['method']}); {sim_info['over_threshold']} pares por encima",
            "Cumplido" if sim_info["over_threshold"] == 0 else "No cumplido")
    if judge_validation and judge_validation.get("judges"):
        parts = [f"{m}: {pct(s['agreement'])} % (κ = {'—' if s['kappa'] is None else format(s['kappa'], '.2f')}, n = {s['n']})"
                 for m, s in judge_validation["judges"].items()]
        passed = all(s["passes"] for s in judge_validation["judges"].values())
        add(f"Acuerdo juez-humano ≥ {th['judge_human_agreement_min'] * 100:.0f} %", "; ".join(parts),
            "Cumplido" if passed else "No cumplido")
    else:
        add(f"Acuerdo juez-humano ≥ {th['judge_human_agreement_min'] * 100:.0f} %",
            "Etiquetado humano pendiente", "No evaluado")
    if lme:
        parts = [f"{c} {pct(v['accuracy'])} %" for c, v in lme["global"].items() if v["accuracy"] is not None]
        add(f"Validación externa, LongMemEval-S ({lme['n']} preguntas)", ", ".join(parts),
            "Orientativa: RMM reporta 70.4 % con otro recuperador y otro juez (Tan, Z. et al., 2025)")
    else:
        add("Validación externa, LongMemEval-S", "Pendiente", "No evaluado")

    md = ["## Estado de los objetivos\n", "Trazabilidad entre objetivos, resultados y estado.\n",
          "| Objetivo o criterio | Resultado | Estado |", "| --- | --- | --- |"]
    md += [f"| {r['objetivo']} | {r['resultado']} | {r['estado']} |" for r in rows]
    md.append("")
    return md, rows



def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis estadístico y tablas de la evaluación.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", choices=["judge", "auto"], default="judge")
    parser.add_argument("--sim-dir", type=Path, default=None)
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("app/data/eval_dataset"))
    parser.add_argument("--thresholds", type=Path, default=Path(__file__).resolve().parent / "eval_thresholds.json")
    parser.add_argument("--main-length", type=int, default=None, help="Longitud del historial para las tablas de exactitud y de comparaciones pareadas (por defecto 20 si existe).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lme-dir", type=Path, default=None,
                        help="Corrida de LongMemEval (run_conditions.py --split lme) ya juzgada.")
    args = parser.parse_args()

    th = json.loads(args.thresholds.read_text(encoding="utf-8"))
    rows, info = load_accuracy(args.run_dir, args.source)
    conditions = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    lengths = sorted({r["history_length"] for r in rows})
    main_length = args.main_length or (20 if 20 in lengths else lengths[-1])
    out_dir = args.output_dir or (args.run_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_names = ""
    verdicts_path = args.run_dir / "verdicts.jsonl"
    if args.source == "judge" and verdicts_path.exists():
        first = next(iter(load_jsonl(verdicts_path)), {})
        judge_names = " (" + ", ".join(m.split(":")[-1] for m in (first.get("votes") or {})) + ")"
    md = ["# Resultados de la evaluación\n",
          f"Fuente del veredicto: **{'juez LLM' + judge_names if args.source == 'judge' else 'palabras clave (auxiliar, solo piloto)'}**. "
          f"Filas: {info['rows']}" + (f"; sin veredicto o pendientes: {info.get('unjudged_or_pending', 0)}" if args.source == 'judge' else "") + ".\n"]
    acc_md, acc = accuracy_tables(rows, main_length, lengths, conditions, th["equivalence_margin_pp"], th["alpha"])
    rec_md, rec = recall_and_tokens(rows, lengths, conditions, th["recall_k"])
    sim_md, sim = simulation_section(args.sim_dir)
    lat_md, load = latency_section(args.load_dir, th["retrieval_p95_ms_max"])
    dataset_report_path = args.dataset_dir / "dataset_report.json"
    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8")) if dataset_report_path.exists() else None
    jv_path = args.run_dir / "judge_validation.json"
    judge_validation = json.loads(jv_path.read_text(encoding="utf-8")) if jv_path.exists() else None
    lme_summary, lme_md = None, []
    if args.lme_dir is not None:
        lme_rows, _ = load_accuracy(args.lme_dir, "judge")
        lme_md, lme_summary = lme_section(lme_rows)
        lme_summary["n"] = max((v["n"] for v in lme_summary["global"].values()), default=0)
    obj_md, objectives = objectives_section(acc, rec, sim, load, dataset_report, judge_validation, th,
                                            main_length, lengths, lme_summary)
    md += acc_md + rec_md + sim_md + lat_md + lme_md + obj_md

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    with (out_dir / "objectives.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["objetivo", "resultado", "estado"])
        writer.writeheader()
        writer.writerows(objectives)
    summary = {"info": info, "main_length": main_length, "lengths": lengths, "conditions": conditions,
               "accuracy": acc, "recall_tokens": rec, "simulation": sim, "longmemeval": lme_summary, "latency_files": list((load or {}).keys()),
               "objectives": objectives}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    print("\n".join(md))
    print(f"\nInforme: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
