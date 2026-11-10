"""
Generador del dataset de evaluación de memoria (v2) con un LLM.

Genera, con un modelo distinto del evaluado (GENERATOR_MODEL, por defecto gpt-6.1-terra):
  - cases.jsonl: casos de test (20 por categoría) y de ajuste (5 por categoría);
  - distractors.jsonl: pool compartido de sesiones distractoras sin datos personales;
  - manifest.json: modelo, parámetros y semilla usados.

Cada caso recibe una "semilla de escenario" determinista (intención, persona, destino,
mes, número de viajeros) para repartir las 10 intenciones y forzar la diversidad. La
salida se valida con eval_dataset.validate_case; si falla, se reintenta pasando los
errores al modelo. La generación es reanudable: los ids ya escritos no se regeneran.

Uso:
  python scripts/generate_eval_dataset.py                       # todo el dataset
  python scripts/generate_eval_dataset.py --test-per-type 2 --dev-per-type 0 --distractors 8 \\
      --output-dir artifacts/eval_dataset_pilot                 # piloto barato
  python scripts/generate_eval_dataset.py --regenerate test-temporal-004 test-extraction-011

Variables de entorno: GENERATOR_MODEL, GENERATOR_API_KEY (o OPENAI_API_KEY),
GENERATOR_BASE_URL, GENERATOR_TEMPERATURE (vacía = no se envía) y
GENERATOR_REASONING_EFFORT (vacía = no se envía).
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

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import (  # noqa: E402
    CASE_TYPES,
    INTENTS,
    MIN_EVIDENCE_SESSIONS,
    append_jsonl,
    load_jsonl,
    validate_case,
    validate_distractor,
    write_jsonl,
)

load_dotenv()

DEFAULT_OUTPUT_DIR = Path("app/data/eval_dataset")

_FIRST_NAMES = [
    "Lucía", "Mateo", "Valeria", "Hugo", "Martina", "Leo", "Daniela", "Álvaro", "Carmen", "Iker",
    "Noelia", "Gonzalo", "Irene", "Rubén", "Paula", "Tomás", "Alba", "Javier", "Nerea", "Óscar",
    "Rocío", "Adrián", "Sofía", "Marcos", "Claudia", "Bruno", "Inés", "Raúl", "Julia", "Víctor",
]
_LAST_NAMES = [
    "Aguilar", "Benítez", "Cabrera", "Delgado", "Escobar", "Fuentes", "Gallardo", "Herrera", "Iglesias",
    "Lozano", "Medina", "Núñez", "Ortiz", "Pastor", "Quintana", "Rivas", "Salazar", "Toledo", "Urrutia",
    "Vidal", "Zamora", "Bravo", "Cano", "Duarte", "Espinosa", "Ferrer", "Guerrero", "Lorenzo", "Marín", "Prieto",
]
_DESTINATIONS = [
    "Lisboa", "Oporto", "Roma", "Florencia", "Nápoles", "Atenas", "Santorini", "Estambul", "Praga", "Viena",
    "Budapest", "Cracovia", "Berlín", "Ámsterdam", "Brujas", "Dublín", "Edimburgo", "Copenhague", "Estocolmo",
    "Reikiavik", "Marrakech", "El Cairo", "Tokio", "Kioto", "Seúl", "Bangkok", "Bali", "Hanói", "Sídney",
    "Nueva York", "Cancún", "La Habana", "Cartagena de Indias", "Cusco", "Buenos Aires", "Río de Janeiro",
    "Ciudad de México", "Punta Cana", "Tenerife", "Mallorca", "Ibiza", "Granada", "Sevilla", "Bilbao",
    "Santiago de Compostela", "Menorca", "Dubrovnik", "Malta", "Zanzíbar", "Ciudad del Cabo",
]
_MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
_DISTRACTOR_TOPICS = [
    "política general de cancelación y reembolsos",
    "qué documentación se necesita para viajar dentro de la UE",
    "cómo funcionan los seguros de viaje en general",
    "horarios y canales de atención al cliente",
    "cómo se emite y descarga una factura",
    "qué incluye normalmente un paquete vacacional",
    "equipaje permitido en vuelos de bajo coste",
    "cómo funcionan los pagos fraccionados",
    "qué hacer ante una incidencia en destino",
    "diferencias entre régimen de media pensión y todo incluido",
    "cómo funcionan los programas de puntos de la agencia",
    "requisitos para viajar con mascotas",
    "consejos generales para viajar con niños pequeños",
    "cómo se gestionan los cambios de nombre en una reserva",
]

_SYSTEM_PROMPT = (
    "Eres un diseñador de datasets de evaluación para asistentes de atención al cliente de una agencia "
    "de viajes. Escribes en español, con diálogos realistas, variados y naturales. "
    "Todos los datos personales son ficticios. Respondes solo con un objeto JSON válido."
)

_CASE_SCHEMA = """{
  "evidence_sessions": [
    {"label": "texto breve", "turns": [{"user": "...", "assistant": "..."}]}
  ],
  "question": "pregunta del usuario en una sesión posterior nueva",
  "reference_answer": "respuesta correcta y breve",
  "answer_keywords": ["valores exactos que debe contener una respuesta correcta"],
  "forbidden_keywords": ["valores que harían incorrecta la respuesta"],
  "gold_facts": [{"text": "hecho del usuario necesario para responder", "keywords": ["valor exacto"]}],
  "slots": {"nombre_slot": "valor dado por el usuario"},
  "task": {
    "goal": "acción que el usuario quiere completar en una sesión futura (reservar, modificar, pedir presupuesto...)",
    "opening_message": "primer mensaje del usuario en esa sesión: pide la acción sin dar datos ni pedir que se los recuerden",
    "required_slots": ["claves de slots que el asistente necesita para completar la acción"],
    "success_criteria": "estado final verificable que indica la tarea resuelta"
  }
}"""

_TYPE_INSTRUCTIONS = {
    "extraction": (
        "Categoría EXTRACCIÓN. Un dato concreto del usuario se menciona una sola vez en una sesión "
        "de evidencia (por ejemplo presupuesto, fecha de salida, número de reserva, restricción alimentaria). "
        "La pregunta, en una sesión posterior, pide ese dato sin mencionarlo."
    ),
    "multi_session": (
        "Categoría MULTI-SESIÓN. La respuesta exige combinar datos que el usuario dio en AL MENOS DOS sesiones "
        "distintas (por ejemplo, el destino en una y el número de viajeros en otra, o sumar dos importes). "
        "Ninguna sesión por sí sola basta. answer_keywords incluye los valores de las distintas sesiones o el "
        "resultado combinado, y gold_facts tiene un hecho por sesión."
    ),
    "temporal": (
        "Categoría TEMPORAL. Dos o más sesiones de evidencia con eventos o consultas distintas, en orden. "
        "Las fechas se dicen de forma explícita dentro del diálogo. La pregunta pide el orden o la fecha "
        "relativa (qué fue primero, cuánto tiempo pasó entre dos eventos) SIN ofrecer las opciones en la "
        "pregunta: nada de '¿A o B?'. answer_keywords contiene el evento o la fecha correcta y "
        "forbidden_keywords el evento incorrecto con el que se podría confundir."
    ),
    "knowledge_update": (
        "Categoría ACTUALIZACIÓN DE CONOCIMIENTO. Un dato (fecha, destino, presupuesto, número de viajeros, "
        "contacto) se da en la primera sesión de evidencia y el usuario lo CAMBIA en la última. Puede haber "
        "otra sesión intermedia que no lo toque. La pregunta pide el valor vigente sin mencionar ninguno de "
        "los dos. answer_keywords es el valor nuevo y forbidden_keywords el antiguo."
    ),
    "abstention": (
        "Categoría ABSTENCIÓN. El historial contiene información CERCANA pero no equivalente a lo que se "
        "pregunta (por ejemplo, el usuario dio su nacionalidad pero nunca su número de pasaporte; dio la "
        "fecha de ida pero nunca la de vuelta). La pregunta pide el dato que nunca se dio. reference_answer "
        "dice que ese dato no consta. answer_keywords es una lista vacía y forbidden_keywords contiene los "
        "valores cercanos que una respuesta incorrecta podría dar como si fueran el dato pedido."
    ),
}


def _case_prompt(case_type: str, scenario: dict[str, Any], min_turns: int, max_turns: int,
                 feedback: list[str] | None, avoid: list[str] | None = None) -> str:
    min_sessions = MIN_EVIDENCE_SESSIONS[case_type]
    parts = [
        _TYPE_INSTRUCTIONS[case_type],
        "",
        "Semilla de escenario (úsala como punto de partida, con detalles propios):",
        json.dumps(scenario, ensure_ascii=False),
        "",
        "Reglas:",
        f"- Entre {min_sessions} y {max(min_sessions, 3)} sesiones de evidencia, cada una con entre "
        f"{min_turns} y {max_turns} turnos (un turno = un mensaje del usuario y la respuesta del asistente).",
        "- Las sesiones son conversaciones completas y verosímiles de atención al cliente; la información "
        "relevante aparece de forma natural, mezclada con otras consultas, no en el primer turno siempre.",
        "- El asistente de las sesiones de evidencia no repite ni resume los datos clave del usuario; "
        "responde con brevedad y sin inventar políticas extrañas.",
        "- La pregunta va en una sesión NUEVA posterior, como si el usuario volviera días después. "
        "No debe contener la respuesta.",
        "- answer_keywords y las keywords de gold_facts son valores exactos que aparecen literalmente en el "
        "diálogo (nombres propios, cifras, códigos, fechas, ciudades), no paráfrasis. Deben ser cortos.",
        f"- El dato clave aparece hacia el {scenario['posicion_dato']} de su sesión.",
        "- gold_facts contiene SOLO los hechos necesarios para responder la pregunta (en abstención, los "
        "datos cercanos con los que se podría confundir), no todos los datos del usuario.",
        "- slots recoge TODOS los datos personales que el usuario dio en las sesiones de evidencia.",
        "- task es una ACCIÓN futura del mismo usuario (reservar, modificar, pedir presupuesto, reclamar...) "
        "que el asistente solo puede completar con al menos dos de esos slots. No es la pregunta del caso ni "
        "una petición de recordar datos. required_slots usa las mismas claves que slots.",
        f"- La persona se llama {scenario['persona']}.",
        "",
        "Devuelve SOLO un objeto JSON con esta forma:",
        _CASE_SCHEMA,
    ]
    if avoid:
        parts += [
            "",
            "Ya existen casos de esta categoría e intención con estas preguntas. Elige otro dato, otro tipo "
            "de evento y otra formulación; el escenario debe ser claramente distinto:",
            *[f"- {q}" for q in avoid],
        ]
    if feedback:
        parts += ["", "Tu respuesta anterior no era válida. Corrige estos errores:", *[f"- {e}" for e in feedback]]
    return "\n".join(parts)


def _distractor_prompt(topic: str, min_turns: int, max_turns: int, feedback: list[str] | None) -> str:
    parts = [
        "Escribe una conversación de atención al cliente de una agencia de viajes sobre este tema general:",
        topic,
        "",
        "Reglas:",
        f"- Entre {min_turns} y {max_turns} turnos (mensaje del usuario y respuesta del asistente).",
        "- El usuario NO da ningún dato personal: ni nombre, ni ciudades o países concretos, ni fechas, ni "
        "importes, ni números de reserva, ni preferencias propias. Solo pregunta dudas generales.",
        "- El asistente responde de forma general y breve, sin citar destinos concretos.",
        "",
        'Devuelve SOLO JSON: {"label": "tema breve", "turns": [{"user": "...", "assistant": "..."}]}',
    ]
    if feedback:
        parts += ["", "Corrige estos errores de tu respuesta anterior:", *[f"- {e}" for e in feedback]]
    return "\n".join(parts)


class Generator:
    def __init__(self, concurrency: int) -> None:
        from openai import AsyncOpenAI

        self.model = os.getenv("GENERATOR_MODEL", "gpt-6.1-terra")
        self.client = AsyncOpenAI(
            api_key=os.getenv("GENERATOR_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("GENERATOR_BASE_URL") or None,
        )
        self.extra: dict[str, Any] = {}
        temperature = os.getenv("GENERATOR_TEMPERATURE", "").strip()
        if temperature:
            self.extra["temperature"] = float(temperature)
        effort = os.getenv("GENERATOR_REASONING_EFFORT", "").strip()
        if effort:
            self.extra["reasoning_effort"] = effort
        self.semaphore = asyncio.Semaphore(concurrency)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    async def ask_json(self, prompt: str) -> dict[str, Any]:
        async with self.semaphore:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                **self.extra,
            )
        self.usage["calls"] += 1
        if resp.usage is not None:
            self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
            self.usage["output_tokens"] += resp.usage.completion_tokens or 0
        payload = json.loads(resp.choices[0].message.content or "{}")
        if not isinstance(payload, dict):
            raise ValueError("la respuesta no es un objeto JSON")
        return payload


def _plan_cases(test_per_type: int, dev_per_type: int, seed: int) -> list[dict[str, Any]]:
    """Asigna id, split, tipo y semilla de escenario a cada caso, de forma determinista."""
    rng = random.Random(seed)
    names = [f"{first} {last}" for first in _FIRST_NAMES for last in _LAST_NAMES]
    rng.shuffle(names)
    destinations = list(_DESTINATIONS)
    plan: list[dict[str, Any]] = []
    counter = 0
    for split, per_type in (("test", test_per_type), ("dev", dev_per_type)):
        for type_index, case_type in enumerate(CASE_TYPES):
            for n in range(per_type):
                # Cada categoría recorre las 10 intenciones con un desfase distinto.
                intent = INTENTS[(n + 2 * type_index + (0 if split == "test" else 5)) % len(INTENTS)]
                scenario = {
                    "intencion": intent,
                    "persona": names[counter % len(names)],
                    "destino_inicial": destinations[(counter * 7 + type_index) % len(destinations)],
                    "mes_del_viaje": _MONTHS[(counter * 5) % len(_MONTHS)],
                    "viajeros": 1 + (counter * 3) % 5,
                    "posicion_dato": ("principio", "medio", "final")[counter % 3],
                }
                plan.append(
                    {
                        "id": f"{split}-{case_type.replace('_', '-')}-{n + 1:03d}",
                        "split": split,
                        "type": case_type,
                        "intent": intent,
                        "scenario": scenario,
                    }
                )
                counter += 1
    return plan


async def _generate_case(gen: Generator, spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    feedback: list[str] | None = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            payload = await gen.ask_json(
                _case_prompt(spec["type"], spec["scenario"], args.min_turns, args.max_turns, feedback,
                             spec.get("avoid"))
            )
        except Exception as exc:  # error de red o JSON inválido: se reintenta
            feedback = [f"respuesta no utilizable: {exc}"]
            continue
        case = {
            "id": spec["id"],
            "split": spec["split"],
            "type": spec["type"],
            "intent": spec["intent"],
            "domain": "travel_agency",
            "persona": {"name": spec["scenario"]["persona"]},
            "scenario_seed": spec["scenario"],
            **{k: payload.get(k) for k in (
                "evidence_sessions", "question", "reference_answer", "answer_keywords",
                "forbidden_keywords", "gold_facts", "slots", "task",
            )},
        }
        case["answer_keywords"] = case["answer_keywords"] or []
        case["forbidden_keywords"] = case["forbidden_keywords"] or []
        errors = validate_case(case, min_turns=args.min_turns)
        if not errors:
            case["generation"] = {"model": gen.model, "attempts": attempt}
            return case
        feedback = errors
        print(f"  {spec['id']}: intento {attempt} inválido: {'; '.join(errors[:3])}", flush=True)
    print(f"  {spec['id']}: DESCARTADO tras {args.max_attempts} intentos", flush=True)
    return None


async def _generate_distractor(gen: Generator, index: int, args: argparse.Namespace) -> dict[str, Any] | None:
    topic = _DISTRACTOR_TOPICS[index % len(_DISTRACTOR_TOPICS)]
    feedback: list[str] | None = None
    for _ in range(args.max_attempts):
        try:
            payload = await gen.ask_json(_distractor_prompt(topic, args.min_turns, args.max_turns, feedback))
        except Exception as exc:
            feedback = [f"respuesta no utilizable: {exc}"]
            continue
        session = {"id": f"distractor-{index + 1:03d}", "label": payload.get("label") or topic,
                   "topic": topic, "turns": payload.get("turns")}
        errors = validate_distractor(session, min_turns=args.min_turns)
        if not errors:
            return session
        feedback = errors
    print(f"  distractor-{index + 1:03d}: DESCARTADO", flush=True)
    return None


async def _run(args: argparse.Namespace) -> None:
    out_dir: Path = args.output_dir
    cases_path = out_dir / "cases.jsonl"
    distractors_path = out_dir / "distractors.jsonl"

    existing_cases = load_jsonl(cases_path)
    if args.regenerate:
        drop = set(args.regenerate)
        existing_cases = [c for c in existing_cases if c["id"] not in drop]
        write_jsonl(cases_path, existing_cases)
    done_ids = {c["id"] for c in existing_cases}
    plan = [spec for spec in _plan_cases(args.test_per_type, args.dev_per_type, args.seed)
            if spec["id"] not in done_ids]
    if args.regenerate:
        # Un caso se regenera por parecerse demasiado a otro de su misma categoría e
        # intención: se le muestran esas preguntas para que cambie de enfoque.
        for spec in plan:
            spec["avoid"] = [
                c["question"] for c in existing_cases
                if c["type"] == spec["type"] and c.get("intent") == spec["intent"]
            ]

    existing_distractors = load_jsonl(distractors_path)
    done_distractors = {d["id"] for d in existing_distractors}
    distractor_todo = [i for i in range(args.distractors) if f"distractor-{i + 1:03d}" not in done_distractors]

    gen = Generator(args.concurrency)
    print(f"Modelo generador: {gen.model}")
    print(f"Casos pendientes: {len(plan)} (ya hechos {len(done_ids)}); "
          f"distractores pendientes: {len(distractor_todo)}")

    async def case_task(spec: dict[str, Any]) -> None:
        case = await _generate_case(gen, spec, args)
        if case is not None:
            append_jsonl(cases_path, case)
            print(f"  {case['id']} ok", flush=True)

    async def distractor_task(index: int) -> None:
        session = await _generate_distractor(gen, index, args)
        if session is not None:
            append_jsonl(distractors_path, session)

    await asyncio.gather(*(distractor_task(i) for i in distractor_todo), *(case_task(s) for s in plan))

    cases = load_jsonl(cases_path)
    distractors = load_jsonl(distractors_path)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": gen.model,
        "generator_params": gen.extra,
        "seed": args.seed,
        "min_turns": args.min_turns,
        "max_turns": args.max_turns,
        "planned": {"test_per_type": args.test_per_type, "dev_per_type": args.dev_per_type,
                    "distractors": args.distractors},
        "counts": {
            "cases": len(cases),
            "by_split_type": {
                f"{split}/{t}": sum(1 for c in cases if c["split"] == split and c["type"] == t)
                for split in ("test", "dev") for t in CASE_TYPES
            },
            "distractors": len(distractors),
        },
        "usage_this_run": gen.usage,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Uso en esta ejecución: {gen.usage}")
    planned_ids = {s["id"] for s in _plan_cases(args.test_per_type, args.dev_per_type, args.seed)}
    missing = len(planned_ids - {c["id"] for c in cases})
    if missing:
        print(f"AVISO: faltan {missing} casos; vuelve a ejecutar para reintentarlos.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el dataset de evaluación de memoria (v2) con un LLM.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-per-type", type=int, default=20)
    parser.add_argument("--dev-per-type", type=int, default=5)
    parser.add_argument("--distractors", type=int, default=70,
                        help="Tamaño del pool de distractores (>= 50 para instanciar con 50 sesiones).")
    parser.add_argument("--min-turns", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate", nargs="+", default=None, metavar="ID",
                        help="Borra y vuelve a generar estos casos (p. ej. tras check_eval_dataset).")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
