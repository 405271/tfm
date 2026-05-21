# Asistente de soporte con LangChain, FastAPI, Streamlit y Gradio

Proyecto base orientado a soporte funcional de productos documentados con FAQ + RAG, con:

- `langchain==1.3.1`
- `fastapi==0.136.1`
- `streamlit==1.57.0`
- `gradio==6.14.0`
- `langchain-openai==1.2.1`
- `langchain-ollama==1.1.0`
- `memori==3.3.3` opcional para memoria persistente

Incluye:

- prompt de soporte funcional configurable por entorno
- FAQ local opcional en `app/data/faq.json`
- glosario local opcional en `app/data/glossary.json`
- RAG sobre documentacion web configurable con indice vectorial local
- sesiones simples por `session_id`
- memoria opcional con MemoriLabs
- escalacion basica a revision humana
- frontends en Streamlit, Gradio y CLI
- conocimiento inicial extraido desde la URL que configures en `DOCS_ROOT_URL`

## Requisitos

- Python 3.10 o superior
- `OPENAI_API_KEY` configurada

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

La app carga `.env` automaticamente al iniciar. No hace falta ejecutar `export`.

Variables opcionales para memoria:

- `MEMORI_ENABLED=true` para activar MemoriLabs
- `MEMORI_PROCESS_ID=support-api` para identificar el proceso/agente
- `MEMORI_STORAGE_URL=sqlite:///app/data/memori.db` para definir la base local
- `MEMORI_API_KEY=...` si ademas vas a usar Memori Cloud / augmentation con tu cuenta

Variables para proveedor de modelo:

- `LLM_PROVIDER=openai|deepseek|ollama`
- `LLM_MODEL` para indicar el modelo concreto del proveedor
- `LLM_BASE_URL` para sobreescribir el endpoint del proveedor
- `LLM_TEMPERATURE` para la temperatura del modelo
- `DEEPSEEK_API_KEY` para DeepSeek

Ejemplos:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
```

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=tu_api_key
```

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
LLM_BASE_URL=http://127.0.0.1:11434
```

Variables para embeddings:

- `EMBEDDINGS_PROVIDER=openai|ollama`
- `EMBEDDINGS_MODEL` para indicar el modelo de embeddings
- `EMBEDDINGS_BASE_URL` para sobreescribir el endpoint de embeddings

Si `EMBEDDINGS_PROVIDER` no se define:

- usa `ollama` automaticamente cuando `LLM_PROVIDER=ollama`
- usa `openai` en los demas casos

Ejemplos:

```env
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small
```

```env
EMBEDDINGS_PROVIDER=ollama
EMBEDDINGS_MODEL=nomic-embed-text
EMBEDDINGS_BASE_URL=http://127.0.0.1:11434
```

En este proyecto, DeepSeek queda habilitado para chat, pero no para embeddings. Si usas `LLM_PROVIDER=deepseek`, debes dejar embeddings en `openai` u `ollama`.

Variables para generalizar el dominio y el RAG:

- `PRODUCT_NAME` nombre del producto al que responde el asistente
- `SUPPORT_CONTACT` texto con canales de soporte humano
- `GLOSSARY_TITLE` y `GLOSSARY_URL` para la fuente del glosario
- `ISSUE_CONTEXT_REQUEST` texto usado cuando falta contexto en un caso o incidencia
- `ESCALATION_REQUEST` texto base para escalar a soporte humano
- `DOCS_ROOT_URL` URL raiz desde donde arranca el crawler web
- `DOCS_SOURCE_DIR` carpeta local con documentos `.txt`, `.md`, `.html` o `.pdf`
- `DOCS_ALLOWED_DOMAINS` lista separada por comas de dominios permitidos
- `DOCS_PATH_PREFIX` prefijo de path permitido para enlaces del crawler
- `DOCS_URL_SUFFIX` sufijo de URL permitido
- `DOCS_CONTENT_SELECTOR` selector CSS del contenedor principal del contenido
- `DOCS_MAX_PAGES` limite de paginas a recorrer
- `RAG_INDEX_DIR` carpeta bajo `app/data/` donde se guarda el indice
- `RAG_CHUNK_SIZE` y `RAG_CHUNK_OVERLAP` para troceo

## Construir la base vectorial

Antes de usar el modo RAG, genera el indice local:

```bash
source .venv/bin/activate
python -m app.rag.ingest_docs
```

Puedes usar cualquiera de estas opciones:

- `DOCS_ROOT_URL` para rastrear documentacion web
- `DOCS_SOURCE_DIR` para indexar documentos locales
- ambas a la vez, si quieres combinar fuentes

Esto procesa la documentacion configurada, la trocea, genera embeddings y guarda el indice en `app/data/<RAG_INDEX_DIR>/`.
Tambien limpia y reconstruye el cache local de consultas.

## Ejecutar el backend

```bash
source .venv/bin/activate
uvicorn app.api.main:app --reload
```

La API queda en `http://127.0.0.1:8000` y FastAPI expone OpenAPI en:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

Endpoint principal:

- `POST /chat`
  - acepta `message`, `session_id` opcional y `history`
  - devuelve `reply`, `session_id`, `escalated` y `sources`

## Ejecutar el frontend con Streamlit

En otra terminal:

```bash
source .venv/bin/activate
streamlit run app/ui/streamlit_app.py
```

## Ejecutar el frontend con Gradio

En otra terminal:

```bash
source .venv/bin/activate
python app/ui/gradio_app.py
```

## Ejecutar la interfaz CLI

En otra terminal:

```bash
source .venv/bin/activate
python app/ui/cli_chat.py
```

## Flujo

1. El frontend envia el mensaje y una `session_id`.
2. FastAPI recupera el historial de la sesion.
3. Si `MEMORI_ENABLED=true`, MemoriLabs envuelve `ChatOpenAI` y persiste/recupera memoria asociada al `session_id` en SQLite local.
   Esta integracion esta preparada para `openai` y `deepseek`. En `ollama`, Memori debe quedar desactivado.
4. El backend consulta el indice vectorial local de la documentacion configurada.
5. Si existen, el backend selecciona FAQs o definiciones de glosario y aplica reglas basicas de soporte.
6. Si falta contexto del incidente o el caso requiere humano, responde con handoff.
7. En los demas casos, LangChain llama a OpenAI usando `ChatOpenAI` con contexto RAG.

## Personalizacion rapida

- Edita [app/data/faq.json](/tmp/0/chat/app/data/faq.json) con preguntas reales de tu despliegue si quieres respuestas directas tipo FAQ.
- Crea `app/data/glossary.json` si quieres respuestas definicionales rápidas.
- Ajusta el prompt en [app/api/main.py](/tmp/0/chat/app/api/main.py).
- Cambia reglas de escalacion en `ESCALATION_KEYWORDS`.
- Cambia `MEMORI_STORAGE_URL` si quieres guardar la memoria en otra base.
- Configura `DOCS_ROOT_URL` o `DOCS_SOURCE_DIR` para la fuente documental que quieras usar.
- Ajusta `DOCS_ALLOWED_DOMAINS` y `DOCS_CONTENT_SELECTOR` si trabajas con documentacion web.
- Regenera el indice con `python -m app.rag.ingest_docs` cuando cambie la documentacion.
- Conecta luego usuarios, tickets, logs o permisos reales si quieres respuestas accionables.

`faq.json` y `glossary.json` no son obligatorios. Si no existen, estan vacios o no contienen datos validos, el backend sigue funcionando y responde con la informacion recuperada por RAG.

## Evaluar con y sin memoria

Puedes comparar dos instancias del backend, una con memoria y otra sin memoria, usando:

```bash
python scripts/eval_memory.py \
  --baseline-url http://127.0.0.1:8000 \
  --candidate-url http://127.0.0.1:8001
```

Si quieres levantar ambas variantes automaticamente y ejecutar la comparacion de punta a punta:

```bash
bash scripts/run_eval_memory.sh
```

El runner:

- levanta una instancia sin memoria en `127.0.0.1:8000`
- levanta una instancia con memoria en `127.0.0.1:8001`
- espera a que ambas respondan en `/health`
- corre `scripts/eval_memory.py`
- guarda logs en `artifacts/eval_memory/`

Variables utiles:

- `BASELINE_PORT=9000`
- `CANDIDATE_PORT=9001`
- `CASES_PATH=app/data/eval_memory_cases.json`

Tambien puedes evaluar una sola instancia:

```bash
python scripts/eval_memory.py \
  --api-url http://127.0.0.1:8000 \
  --variant without_memory
```

Casos base: [app/data/eval_memory_cases.json](/home/xaero/Documents/docs/dev/py/tfm/chat/app/data/eval_memory_cases.json)

Salidas:

- `artifacts/eval_memory/results.json`
- `artifacts/eval_memory/results.csv`

El evaluador mide por variante:

- tasa de casos aprobados
- tasa de turnos aprobados
- latencia promedio
- repeticiones exactas entre respuestas
- repeticiones por alta similitud
- turnos hasta resolucion cuando el caso la marca
