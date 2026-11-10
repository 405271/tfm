# Asistente conversacional con memoria persistente para atención al cliente en agencias de viajes

Es un asistente de atención al cliente (FastAPI + LangChain) con FAQ, glosario, RAG sobre la documentación de la agencia y memoria persistente opcional con Memori. Incluye los scripts de evaluación que comparan el asistente con y sin memoria.

## Requisitos

- Python 3.10 o superior (la evaluación se ejecutó con Python 3.13.13).
- `OPENAI_API_KEY` (modelo evaluado, generador del dataset, usuario simulado y embeddings del RAG).
- `DEEPSEEK_API_KEY`, solo para la evaluación (juez `deepseek-v4-pro`).

Versiones principales (fijadas en `requirements.txt`): `fastapi==0.136.1`, `langchain==1.3.1`, `langchain-openai==1.2.1`, `memori==3.3.3`, `faiss-cpu==1.12.0`, `sqlalchemy==2.0.44`, `streamlit==1.57.0`. Se instalan como dependencias y la evaluación usó `openai==2.54.0` (de `langchain-openai`) y `sentence-transformers==6.3.0` (de `memori`).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # torch solo CPU; para GPU, quitar esas líneas
cp .env.example .env              # completar las claves; la app lee .env al arrancar
```

## Configuración

Las variables están descritas en `.env.example`. Las más importantes:

| Variable | Uso |
|---|---|
| `LLM_PROVIDER` | `openai`, `deepseek` u `ollama`. |
| `LLM_MODEL` | Modelo del chat. |
| `LLM_TEMPERATURE`, `LLM_REASONING_EFFORT` | Vacías no se envían. Los modelos de razonamiento solo admiten temperatura distinta de 1 con `LLM_REASONING_EFFORT=none`. |
| `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL` | Embeddings del RAG (`openai` u `ollama`). Por defecto, `ollama` si el LLM es Ollama y `openai` en otro caso. |
| `PRODUCT_NAME`, `SUPPORT_CONTACT` | Nombre de la agencia y contacto de soporte humano en el prompt. |
| `DOCS_SOURCE_DIR`, `DOCS_ROOT_URL` | Fuente del RAG: carpeta local y/o URL que se rastrea. |
| `MEMORI_ENABLED` | `true` activa la memoria persistente (SQLite en `MEMORI_STORAGE_URL`). |
| `MEMORI_OFFLOAD_SYNC` | `true` ejecuta las llamadas síncronas de Memori en un hilo; recomendado con varios usuarios a la vez. |
| `ISSUE_CONTEXT_ENABLED`, `ESCALATION_ENABLED`, `FAQ_PATH`, `GLOSSARY_PATH` | Activan o desactivan las respuestas predefinidas. |
| `EVAL_MODE` | Solo para la evaluación. |

Las variables se leen al arrancar: reinicia el servidor después de cambiarlas.

## Uso

```bash
python -m app.rag.ingest_docs               # construye el índice RAG (repetir si cambian los documentos o el modelo de embeddings)
uvicorn app.api.main:app --reload           # API en http://127.0.0.1:8000 (documentación en /docs)
streamlit run app/ui/streamlit_app.py       # interfaz web (en otra terminal)
python app/ui/cli_chat.py                   # interfaz de terminal
```

El corpus de ejemplo está en `docs/travel_agency/` (`DOCS_SOURCE_DIR=docs/travel_agency`).

**API.** `POST /chat` recibe `message` y, opcionalmente, `session_id`, `entity_id` e `history`. Devuelve `reply`, `session_id`, `escalated` y `sources`. `entity_id` identifica al cliente entre sesiones y es la clave de la memoria; sin él no hay memoria entre sesiones. `GET /health` informa del estado.

**Flujo de una petición.**
1. Si el mensaje describe una incidencia sin datos, pide contexto; si contiene una palabra de escalado, deriva a una persona; si pide una definición del glosario, la devuelve. Ninguno de estos casos llama al LLM.
2. En otro caso, el prompt se compone con los fragmentos del RAG, las FAQ más parecidas, los hechos que Memori recupera del cliente y el historial de la sesión.
3. Con Memori activo, antes de devolver la respuesta, un LLM extrae hechos breves del turno («El usuario…»), que se vectorizan en local (`all-MiniLM-L6-v2`) y se guardan en SQLite. La extracción se hace en local, sin usar el servicio en la nube de Memori. Como el código usa funciones internas de Memori, la versión está fijada en `memori==3.3.3`.

## Reproducir la evaluación

**Configuración usada** (`.env`): `OPENAI_MODEL=gpt-6-luna`, `LLM_REASONING_EFFORT=none`, `PRODUCT_NAME=Agencia de viajes` y el índice RAG construido con `DOCS_SOURCE_DIR=docs/travel_agency` y `text-embedding-3-small`. Los scripts ponen `EVAL_MODE=true`, temperatura 0 y desactivan FAQ, glosario, escalado y petición de contexto.

**Semillas.** Todos los scripts usan `--seed 42` por defecto. La semilla fija qué distractores entran en cada historial y en qué posición va la evidencia, la muestra de revisión, el orden en que el juez ve las respuestas, las tareas del usuario simulado, las preguntas de LongMemEval y el bootstrap del análisis. Las llamadas a los modelos no admiten semilla: se usa temperatura 0 y la extracción de hechos se guarda en caché (`artifacts/eval_runs/extraction_cache.jsonl`), así que repetir una ejecución puede dar respuestas algo distintas. Cada ejecución guarda en `manifests.jsonl` la semilla, los modelos y el commit.

**0. Índice RAG.** hay que construirlo antes de las condiciones:

```bash
python -m app.rag.ingest_docs     # con DOCS_SOURCE_DIR=docs/travel_agency en .env
```

**1. Dataset** (`app/data/eval_dataset/`): 100 casos de test, 25 de ajuste y 70 sesiones distractoras, generados con `gpt-6.1-terra`. Ya está incluido; regenerarlo es opcional. Los casos están en `cases.jsonl`, los distractores en `distractors.jsonl`, y `manifest.json` guarda el modelo y la semilla. `dataset_report.json` y `review_sample.csv` son la salida de la validación. Los prompts están en `tesis/generacion_dataset.md`.

```bash
python scripts/generate_eval_dataset.py     # genera (reanudable)
python scripts/check_eval_dataset.py        # valida, mide la similitud y extrae la muestra de revisión
```

**2. Condiciones C0-C4** (sin memoria, contexto completo, RAG sobre la conversación, Memori y oráculo), con 5 y 20 sesiones previas y 3 corridas. Levanta un backend sin memoria (:8000) y otro con Memori (:8001). Si se interrumpe, se repite el mismo comando.

```bash
RUN_NAME=final bash scripts/run_eval_conditions.sh --history-lengths 5 20 --runs 3
```

**3. Juez** (`deepseek-v4-pro`, ciego a la condición, prompts de LongMemEval traducidos):

```bash
python scripts/judge_eval.py judge --run-dir artifacts/eval_runs/final
python scripts/judge_eval.py sample --run-dir artifacts/eval_runs/final      # muestra de 50 para etiquetado humano
python scripts/judge_eval.py agreement --run-dir artifacts/eval_runs/final   # acuerdo con las etiquetas humanas (umbral 90 %)
```

Antes de `agreement` hay que rellenar a mano la columna `humano_1` (y `humano_2` si hay segundo anotador) de `human_sample.csv` con `si` o `no`.

**4. Usuario simulado** (tasa de repetición y pasos hasta la resolución):

```bash
RUN_NAME=final_sim RUNNER=scripts/simulate_user.py bash scripts/run_eval_conditions.sh --conditions C0 C3
```

**5. Prueba de carga** (200 peticiones con 1, 5, 10 y 20 usuarios simultáneos; con la máquina libre):

```bash
bash scripts/run_load_test.sh
```

**6. Validación externa con LongMemEval-S** (60 preguntas). Descargar `longmemeval_s_cleaned.json` del repositorio de LongMemEval (https://github.com/xiaowu0162/LongMemEval). Las preguntas importadas quedan en `artifacts/eval_lme/`:

```bash
python scripts/import_longmemeval.py --input <dir>/longmemeval_s_cleaned.json --output-dir artifacts/eval_lme --per-category 12
EVAL_PROMPT_PROFILE=neutral MEMORI_FACT_LANGUAGE=en RAG_INDEX_DIR=no_existe \
EVAL_EXTRACTION_CACHE=artifacts/eval_runs/extraction_cache_lme.jsonl BASELINE_PORT=8010 MEMORI_PORT=8011 \
RUN_NAME=lme bash scripts/run_eval_conditions.sh --split lme --dataset-dir artifacts/eval_lme \
  --history-lengths 0 --runs 1 --conditions C0 C1 C3 C4 --max-context-chars 2000000
python scripts/judge_eval.py judge --run-dir artifacts/eval_runs/lme
```

**7. Análisis:**

```bash
python scripts/analyze_eval.py --run-dir artifacts/eval_runs/final --sim-dir artifacts/eval_runs/final_sim \
  --load-dir artifacts/eval_runs/load --lme-dir artifacts/eval_runs/lme --output-dir artifacts/eval_runs/final/report
```

Genera `report.md`, `summary.json` y `objectives.csv` (umbrales en `scripts/eval_thresholds.json`).

### Resultados

Los resultados están en `resultados/`. Es una copia, sin cambios, de lo que salió en `artifacts/eval_runs/`:

- `final/report/`: el informe final (`report.md`, `summary.json`, `objectives.csv`).
- `final/verdicts.jsonl`: veredicto del juez para cada respuesta de C0-C4.
- `final/human_sample.csv`, `human_sample_key.json` y `judge_validation.json`: las 50 etiquetas humanas y el acuerdo con el juez.
- `lme/verdicts.jsonl`: veredictos de LongMemEval-S.
- `load/load_sync.json` y `load_offload.json`: latencias de la prueba de carga.
- `*/manifests.jsonl`: configuración de cada ejecución.

Las ejecuciones completas ocupan unos 200 MB y no están en el repositorio (`artifacts/` se ignora). Al repetir los pasos anteriores se crean en `artifacts/eval_runs/<RUN_NAME>/`, con `results.jsonl`, `manifests.jsonl`, `judgments.jsonl`, `verdicts.jsonl`, `memori.db` y los registros de los servidores. `analyze_eval.py` necesita esas ejecuciones completas; con lo que hay en `resultados/` no basta.

### Evaluador anterior (v1)

Se conserva el primer evaluador (`scripts/eval_memory.py`, `scripts/run_eval_memory.sh`), con 50 casos de plantilla y comprobación por subcadenas.

```bash
bash scripts/run_eval_memory.sh   # levanta las dos variantes y las compara; resultados en artifacts/eval_memory/
```
