"""
Generador de casos de evaluación para el asistente de soporte.

Produce 60 casos cubriendo los cinco tipos de LongMemEval usando un fichero
de configuración de dominio. Esto permite adaptar los casos al vocabulario
y terminología del asistente evaluado (soporte genérico, agencia de viajes, etc.).

Uso:
  # Dominio genérico (por defecto)
  python scripts/generate_eval_cases.py \\
      --output app/data/eval_memory_cases_custom.json

  # Agencia de viajes
  python scripts/generate_eval_cases.py \\
      --domain-config app/data/domain_configs/travel_agency.json \\
      --output app/data/eval_memory_cases_custom.json

  # Solo ciertos tipos
  python scripts/generate_eval_cases.py \\
      --domain-config app/data/domain_configs/travel_agency.json \\
      --types cross_session_recall knowledge_update \\
      --count 20

  # Combinar con LongMemEval importado
  python scripts/generate_eval_cases.py \\
      --domain-config app/data/domain_configs/travel_agency.json \\
      --output app/data/eval_memory_cases.json \\
      --merge app/data/eval_memory_cases_lme.json

Distribucion por defecto: 15 + 15 + 10 + 10 + 10 = 60 casos.

Para crear un dominio nuevo, copia app/data/domain_configs/generic.json
y edita los campos: terms, companies, roles, topics_pairs, abstractions.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Nombres de persona (constante: no suelen ser dominio-específicos)
# ---------------------------------------------------------------------------

_NAMES = [
    "Ana García", "Carlos López", "María Martínez", "Juan Rodríguez",
    "Laura Sánchez", "Pedro Fernández", "Isabel Jiménez", "David González",
    "Cristina Díaz", "Alejandro Moreno", "Elena Ruiz", "Sergio Torres",
    "Patricia Hernández", "Roberto Flores", "Marta Castillo", "Andrés Romero",
    "Lucía Navarro", "Miguel Ramos", "Beatriz Molina", "Francisco Ortega",
]

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "domain_configs" / "generic.json"


# ---------------------------------------------------------------------------
# Configuración de dominio
# ---------------------------------------------------------------------------

@dataclass
class DomainConfig:
    domain_name: str = "generic"
    # Términos clave usados en las plantillas de mensajes
    term_incident: str = "incidencia"
    term_request: str = "solicitud"
    term_ticket: str = "ticket"
    ref_incident: str = "INC"
    ref_request: str = "SOL"
    ref_ticket: str = "TKT"
    ref_incident_label: str = "incidencia"
    ref_request_label: str = "solicitud"
    ref_ticket_label: str = "ticket"
    # Pools de datos
    companies: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    topics_pairs: list[dict[str, Any]] = field(default_factory=list)
    abstractions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> "DomainConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        terms = data.get("terms", {})
        return cls(
            domain_name=data.get("domain_name", "generic"),
            term_incident=terms.get("incident", "incidencia"),
            term_request=terms.get("request", "solicitud"),
            term_ticket=terms.get("ticket", "ticket"),
            ref_incident=terms.get("ref_incident", "INC"),
            ref_request=terms.get("ref_request", "SOL"),
            ref_ticket=terms.get("ref_ticket", "TKT"),
            ref_incident_label=terms.get("ref_incident_label", "incidencia"),
            ref_request_label=terms.get("ref_request_label", "solicitud"),
            ref_ticket_label=terms.get("ref_ticket_label", "ticket"),
            companies=data.get("companies", []),
            roles=data.get("roles", []),
            topics_pairs=data.get("topics_pairs", []),
            abstractions=data.get("abstractions", []),
        )


# ---------------------------------------------------------------------------
# Generadores de referencias
# ---------------------------------------------------------------------------

def _ref(cfg: DomainConfig) -> str:
    year = random.randint(2024, 2025)
    num = random.randint(1000, 9999)
    return f"{cfg.ref_request}-{year}-{num:04d}"


def _ticket_ref(cfg: DomainConfig) -> str:
    date = random.randint(20240101, 20241231)
    num = random.randint(1000, 9999)
    return f"{cfg.ref_ticket}-{date}-{num:04d}"


def _incident_ref(cfg: DomainConfig) -> str:
    year = random.randint(2024, 2025)
    num = random.randint(1000, 9999)
    return f"{cfg.ref_incident}-{year}-{num:04d}"


def _pick(pool: list) -> Any:
    return random.choice(pool)


# ---------------------------------------------------------------------------
# single_session_recall
# ---------------------------------------------------------------------------

def _gen_single_session_recall(idx: int, cfg: DomainConfig) -> dict[str, Any]:
    templates = [_make_single_ref, _make_single_ticket, _make_single_company]
    return random.choice(templates)(cfg)


def _make_single_ref(cfg: DomainConfig) -> dict[str, Any]:
    ref = _ref(cfg)
    label = cfg.ref_request_label
    return {
        "type": "single_session_recall",
        "sessions": [{"label": "recall", "turns": [
            {"message": f"Hola, llamo en referencia a la {label} número {ref}. ¿Podéis indicarme su estado actual?"},
            {"message": f"¿Cuál era el número de {label} que te indiqué al inicio de esta conversación?",
             "expect_reply_contains": [ref],
             "expect_answer": ref,
             "marks_resolution": True},
        ]}],
    }


def _make_single_ticket(cfg: DomainConfig) -> dict[str, Any]:
    tk = _ticket_ref(cfg)
    name = _pick(_NAMES)
    label = cfg.ref_ticket_label
    return {
        "type": "single_session_recall",
        "sessions": [{"label": "recall", "turns": [
            {"message": f"Soy {name} y os escribo por el {label} {tk}. Quería saber si hay alguna actualización."},
            {"message": f"¿Puedes repetirme el número de {label} que te di?",
             "expect_reply_contains": [tk],
             "expect_answer": tk,
             "marks_resolution": True},
        ]}],
    }


def _make_single_company(cfg: DomainConfig) -> dict[str, Any]:
    name = _pick(_NAMES)
    company = _pick(cfg.companies)
    ref = _ref(cfg)
    label = cfg.ref_request_label
    keyword = company.lower().split()[0]
    return {
        "type": "single_session_recall",
        "sessions": [{"label": "recall", "turns": [
            {"message": f"Buenos días, soy {name} de {company}. Contacto por la {label} {ref}."},
            {"message": "¿De qué empresa soy según lo que te dije antes?",
             "expect_reply_contains_any": [keyword],
             "expect_answer": company,
             "marks_resolution": True},
        ]}],
    }


# ---------------------------------------------------------------------------
# cross_session_recall
# ---------------------------------------------------------------------------

def _gen_cross_session_recall(idx: int, cfg: DomainConfig) -> dict[str, Any]:
    templates = [
        _make_cross_company, _make_cross_ticket,
        _make_cross_role, _make_cross_ref, _make_cross_department,
    ]
    return random.choice(templates)(cfg)


def _make_cross_company(cfg: DomainConfig) -> dict[str, Any]:
    name = _pick(_NAMES)
    company = _pick(cfg.companies)
    label = cfg.ref_request_label
    keyword = company.lower().split()[0]
    return {
        "type": "cross_session_recall",
        "sessions": [
            {"label": "encoding", "turns": [
                {"message": f"Hola, soy {name} de {company}. Quería consultar sobre el proceso de {label}."},
                {"message": "Muchas gracias, ya entendí el proceso."},
            ]},
            {"label": "distractor", "turns": [
                {"message": f"¿Cuál es el horario de atención al cliente?"},
            ]},
            {"label": "recall", "turns": [
                {"message": "Hola, soy yo de nuevo. ¿Recuerdas a qué empresa pertenezco?",
                 "expect_reply_contains_any": [keyword],
                 "expect_reply_not_contains": ["no recuerdo", "no tengo información sobre tu empresa"],
                 "expect_answer": company,
                 "marks_resolution": True},
            ]},
        ],
    }


def _make_cross_ticket(cfg: DomainConfig) -> dict[str, Any]:
    tk = _ticket_ref(cfg)
    name = _pick(_NAMES)
    label = cfg.ref_ticket_label
    return {
        "type": "cross_session_recall",
        "sessions": [
            {"label": "encoding", "turns": [
                {"message": f"Hola, soy {name}. Os escribo en relación al {label} {tk} que tramitamos la semana pasada."},
                {"message": "Quedamos en esperar confirmación del equipo. Gracias."},
            ]},
            {"label": "recall", "turns": [
                {"message": f"¿Cuál era el número de {label} del que te hablé en mi contacto anterior?",
                 "expect_reply_contains": [tk],
                 "expect_answer": tk,
                 "marks_resolution": True},
            ]},
        ],
    }


def _make_cross_role(cfg: DomainConfig) -> dict[str, Any]:
    name = _pick(_NAMES)
    role = _pick(cfg.roles)
    company = _pick(cfg.companies)
    ref = _ref(cfg)
    label = cfg.ref_request_label
    keywords = [w for w in role.split() if len(w) > 4][:2]
    return {
        "type": "cross_session_recall",
        "sessions": [
            {"label": "encoding", "turns": [
                {"message": f"Buenos días, me llamo {name}, soy {role} en {company}. Consulto por la {label} {ref}."},
                {"message": "Perfecto, muchas gracias por la atención."},
            ]},
            {"label": "recall", "turns": [
                {"message": "Hola, volvemos a hablar. ¿Recuerdas cuál es mi cargo?",
                 "expect_reply_contains_any": keywords if keywords else [role.split()[0]],
                 "expect_answer": role,
                 "marks_resolution": True},
            ]},
        ],
    }


def _make_cross_ref(cfg: DomainConfig) -> dict[str, Any]:
    ref = _ref(cfg)
    name = _pick(_NAMES)
    label = cfg.ref_request_label
    return {
        "type": "cross_session_recall",
        "sessions": [
            {"label": "encoding", "turns": [
                {"message": f"Soy {name} y llamo por la {label} {ref}. El estado es pendiente de validación."},
                {"message": "Gracias, espero la actualización."},
            ]},
            {"label": "distractor", "turns": [
                {"message": "¿Podéis explicarme cómo funciona el proceso de escalado de incidencias?"},
            ]},
            {"label": "recall", "turns": [
                {"message": f"¿Cuál era el número de {label} que te mencioné en mi última sesión?",
                 "expect_reply_contains": [ref],
                 "expect_answer": ref,
                 "marks_resolution": True},
            ]},
        ],
    }


def _make_cross_department(cfg: DomainConfig) -> dict[str, Any]:
    name = _pick(_NAMES)
    role = _pick(cfg.roles)
    company = _pick(cfg.companies)
    tk = _ticket_ref(cfg)
    label = cfg.ref_ticket_label
    keywords = [w for w in role.split() if len(w) > 5][:2]
    return {
        "type": "cross_session_recall",
        "sessions": [
            {"label": "encoding", "turns": [
                {"message": f"Hola, soy {name} de {company}. Trabajo como {role}. Os contacto por el {label} {tk}."},
            ]},
            {"label": "recall", "turns": [
                {"message": "¿Sabes cuál es mi cargo según lo que te dije antes?",
                 "expect_reply_contains_any": keywords if keywords else [role.split()[0]],
                 "expect_answer": role,
                 "marks_resolution": True},
            ]},
        ],
    }


# ---------------------------------------------------------------------------
# knowledge_update
# ---------------------------------------------------------------------------

def _gen_knowledge_update(idx: int, cfg: DomainConfig) -> dict[str, Any]:
    templates = [_make_update_status, _make_update_contact, _make_update_ref_correction]
    return random.choice(templates)(cfg)


def _make_update_status(cfg: DomainConfig) -> dict[str, Any]:
    inc = _incident_ref(cfg)
    name = _pick(_NAMES)
    label = cfg.ref_incident_label
    resolutions = [
        ("arregló", "resuelta"),
        ("solucionó", "solucionada"),
        ("cerró", "cerrada"),
        ("resolvió", "resuelta"),
    ]
    verb, adj = _pick(resolutions)
    return {
        "type": "knowledge_update",
        "sessions": [
            {"label": "initial", "turns": [
                {"message": f"Hola, soy {name}. La {label} {inc} sigue sin resolverse. El acceso sigue fallando."},
            ]},
            {"label": "update", "turns": [
                {"message": f"Os informo de que la {label} {inc} ya quedó {adj} esta mañana. El equipo la {verb}."},
            ]},
            {"label": "recall_update", "turns": [
                {"message": f"¿En qué estado quedó la {label} {inc}?",
                 "expect_reply_contains_any": ["resuelta", "solucionada", "cerrada", "arreglada", verb, adj],
                 "expect_reply_not_contains": ["sin resolver", "pendiente", "sigue sin"],
                 "expect_answer": adj,
                 "marks_resolution": True},
            ]},
        ],
    }


def _make_update_contact(cfg: DomainConfig) -> dict[str, Any]:
    name = _pick(_NAMES)
    inc = _incident_ref(cfg)
    label = cfg.ref_incident_label
    old_contact = _pick(_NAMES)
    new_contact = _pick([n for n in _NAMES if n != old_contact])
    return {
        "type": "knowledge_update",
        "sessions": [
            {"label": "initial", "turns": [
                {"message": f"Soy {name}. Para la {label} {inc}, el contacto técnico es {old_contact}."},
            ]},
            {"label": "update", "turns": [
                {"message": f"Corrección: el contacto para {inc} ha cambiado. Ahora es {new_contact}, no {old_contact}."},
            ]},
            {"label": "recall_update", "turns": [
                {"message": f"¿Quién es el contacto actual para la {label} {inc}?",
                 "expect_reply_contains_any": [new_contact.split()[0].lower()],
                 "expect_reply_not_contains": [old_contact.split()[0].lower()],
                 "expect_answer": new_contact,
                 "marks_resolution": True},
            ]},
        ],
    }


def _make_update_ref_correction(cfg: DomainConfig) -> dict[str, Any]:
    name = _pick(_NAMES)
    wrong_ref = _ref(cfg)
    correct_ref = _ref(cfg)
    label = cfg.ref_request_label
    return {
        "type": "knowledge_update",
        "sessions": [
            {"label": "initial", "turns": [
                {"message": f"Hola soy {name}. Mi {label} es {wrong_ref}."},
            ]},
            {"label": "update", "turns": [
                {"message": f"Perdón, me equivoqué antes. El número correcto de {label} es {correct_ref}, no {wrong_ref}."},
            ]},
            {"label": "recall_update", "turns": [
                {"message": f"¿Cuál es el número correcto de mi {label}?",
                 "expect_reply_contains_any": [correct_ref],
                 "expect_reply_not_contains": [wrong_ref],
                 "expect_answer": correct_ref,
                 "marks_resolution": True},
            ]},
        ],
    }


# ---------------------------------------------------------------------------
# temporal_order
# ---------------------------------------------------------------------------

def _gen_temporal_order(idx: int, cfg: DomainConfig) -> dict[str, Any]:
    pair = _pick(cfg.topics_pairs)
    topic_a = pair["a"]
    topic_b = pair["b"]
    label_a = topic_a["label"]
    label_b = topic_b["label"]
    keywords_a = topic_a["keywords"]
    name = _pick(_NAMES)
    use_name = random.random() > 0.5

    question_variants = [
        f"¿Sobre qué tema te consulté primero: {label_a} o {label_b}?",
        f"¿Qué te pregunté antes, lo de {topic_a['key']} o lo de {topic_b['key']}?",
        f"Entre {label_a} y {label_b}, ¿cuál fue mi primera consulta?",
    ]

    return {
        "type": "temporal_order",
        "sessions": [
            {"label": "first_topic", "turns": [
                {"message": f"{'Hola, soy ' + name + '. ' if use_name else 'Hola. '}Quería preguntar sobre {label_a}."},
            ]},
            {"label": "second_topic", "turns": [
                {"message": f"Hola de nuevo. Esta vez quería consultar sobre {label_b}."},
            ]},
            {"label": "order_recall", "turns": [
                {"message": _pick(question_variants),
                 "expect_reply_contains_any": keywords_a,
                 "expect_answer": label_a,
                 "marks_resolution": True},
            ]},
        ],
    }


# ---------------------------------------------------------------------------
# abstention
# ---------------------------------------------------------------------------

def _gen_abstention(idx: int, cfg: DomainConfig) -> dict[str, Any]:
    abstention = _pick(cfg.abstractions)
    datum = abstention["datum"]
    forbidden = abstention["forbidden"]
    topic_pair = _pick(cfg.topics_pairs)
    unrelated_topic = topic_pair["a"]["label"]

    return {
        "type": "abstention",
        "sessions": [
            {"label": "unrelated", "turns": [
                {"message": f"Hola, quería preguntar sobre {unrelated_topic}."},
            ]},
            {"label": "abstention_test", "turns": [
                {"message": f"¿Cuál es mi {datum}?",
                 "expect_reply_not_contains": forbidden,
                 "marks_resolution": True},
            ]},
        ],
    }


# ---------------------------------------------------------------------------
# Distribución y punto de entrada
# ---------------------------------------------------------------------------

_GENERATORS = {
    "single_session_recall": (_gen_single_session_recall, 15),
    "cross_session_recall": (_gen_cross_session_recall, 15),
    "knowledge_update": (_gen_knowledge_update, 10),
    "temporal_order": (_gen_temporal_order, 10),
    "abstention": (_gen_abstention, 10),
}


def _generate_all(
    counts: dict[str, int],
    cfg: DomainConfig,
    seed: int | None,
) -> list[dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    cases: list[dict[str, Any]] = []
    for case_type, (generator, _) in _GENERATORS.items():
        n = counts.get(case_type, 0)
        for i in range(n):
            case = generator(i, cfg)
            case["id"] = f"custom-{case_type.replace('_', '-')}-{i + 1:03d}"
            case["source"] = "generated"
            case["domain"] = cfg.domain_name
            cases.append(case)

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera casos de evaluacion para el asistente de soporte."
    )
    parser.add_argument(
        "--domain-config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help=(
            "Fichero JSON de configuracion de dominio. "
            "Por defecto usa generic.json. "
            "Ejemplo: app/data/domain_configs/travel_agency.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/data/eval_memory_cases_custom.json"),
        help="Ruta de salida para el archivo de casos generados.",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help="Archivo de casos existente a combinar con los casos generados.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(_GENERATORS.keys()),
        default=None,
        help="Generar solo estos tipos de caso.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="Numero de casos por tipo (sobreescribe los valores por defecto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42).",
    )
    args = parser.parse_args()

    cfg = DomainConfig.from_json(args.domain_config)
    print(f"Dominio: {cfg.domain_name} ({args.domain_config})")

    if not cfg.topics_pairs:
        raise SystemExit("ERROR: el fichero de dominio no tiene 'topics_pairs'. Revisa la configuracion.")
    if not cfg.abstractions:
        raise SystemExit("ERROR: el fichero de dominio no tiene 'abstractions'. Revisa la configuracion.")
    if not cfg.companies:
        raise SystemExit("ERROR: el fichero de dominio no tiene 'companies'. Revisa la configuracion.")

    active_types = args.types or list(_GENERATORS.keys())
    counts = {t: (args.count or _GENERATORS[t][1]) for t in active_types}

    cases = _generate_all(counts, cfg, seed=args.seed)

    if args.merge and args.merge.exists():
        existing = json.loads(args.merge.read_text(encoding="utf-8"))
        existing_ids = {c["id"] for c in existing}
        new_cases = [c for c in cases if c["id"] not in existing_ids]
        cases = existing + new_cases
        print(f"Combinado con {args.merge}: {len(existing)} existentes + {len(new_cases)} nuevos.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    by_type: dict[str, int] = {}
    for c in cases:
        t = c.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"Casos escritos en {args.output}: {len(cases)} total")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
