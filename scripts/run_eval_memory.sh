#!/usr/bin/env bash

set -euo pipefail

BASELINE_PORT="${BASELINE_PORT:-8000}"
CANDIDATE_PORT="${CANDIDATE_PORT:-8001}"
BASELINE_URL="http://127.0.0.1:${BASELINE_PORT}"
CANDIDATE_URL="http://127.0.0.1:${CANDIDATE_PORT}"
CASES_PATH="${CASES_PATH:-app/data/eval_memory_cases.json}"
ARTIFACTS_DIR="artifacts/eval_memory"

# Separate SQLite files per variante para evitar contaminacion cross-run.
CANDIDATE_DB="${ARTIFACTS_DIR}/memori_candidate.db"

cleanup() {
  if [[ -n "${BASELINE_PID:-}" ]] && kill -0 "$BASELINE_PID" 2>/dev/null; then
    kill "$BASELINE_PID" 2>/dev/null || true
    wait "$BASELINE_PID" 2>/dev/null || true
  fi
  if [[ -n "${CANDIDATE_PID:-}" ]] && kill -0 "$CANDIDATE_PID" 2>/dev/null; then
    kill "$CANDIDATE_PID" 2>/dev/null || true
    wait "$CANDIDATE_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

wait_for_health() {
  local url="$1/health"

  if ! command -v curl >/dev/null 2>&1; then
    return
  fi

  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "No respondio health en $url" >&2
  exit 1
}

mkdir -p "$ARTIFACTS_DIR"

# Limpiar BD del candidate antes de cada run para que los resultados
# no se contaminen con memorias de ejecuciones anteriores.
rm -f "$CANDIDATE_DB"
echo "BD del candidate limpiada: ${CANDIDATE_DB}"

# Durante la evaluación de memoria desactivamos FAQ y glosario para que los mensajes
# lleguen siempre al LLM. De lo contrario, el path rápido de FAQ responde antes que
# Memori y ambas variantes obtienen puntuaciones idénticas (el FAQ es estático).
# ISSUE_CONTEXT_ENABLED=false apaga también la respuesta predefinida que pide contexto
# (se disparaba con subcadenas como "con-firma-ción"). EVAL_MODE expone la traza
# (memorias, tokens y tiempos por etapa) y LLM_TEMPERATURE=0 para que las respuestas sean reproducibles.
FAQ_PATH=/dev/null
GLOSSARY_PATH=/dev/null
ESCALATION_ENABLED=false
ISSUE_CONTEXT_ENABLED=false
EVAL_MODE=true
LLM_TEMPERATURE="${EVAL_LLM_TEMPERATURE:-0}"
export FAQ_PATH GLOSSARY_PATH ESCALATION_ENABLED ISSUE_CONTEXT_ENABLED EVAL_MODE LLM_TEMPERATURE

echo "Levantando baseline sin memoria en ${BASELINE_URL}"
MEMORI_ENABLED=false \
uvicorn app.api.main:app --host 127.0.0.1 --port "$BASELINE_PORT" \
  > "${ARTIFACTS_DIR}/baseline.log" 2>&1 &
BASELINE_PID=$!

echo "Levantando candidate con memoria en ${CANDIDATE_URL}"
MEMORI_ENABLED=true \
MEMORI_STORAGE_URL="sqlite:///${CANDIDATE_DB}" \
MEMORI_WAIT_AUGMENTATION=true \
uvicorn app.api.main:app --host 127.0.0.1 --port "$CANDIDATE_PORT" \
  > "${ARTIFACTS_DIR}/candidate.log" 2>&1 &
CANDIDATE_PID=$!

wait_for_health "$BASELINE_URL"
wait_for_health "$CANDIDATE_URL"

python scripts/eval_memory.py \
  --cases "$CASES_PATH" \
  --baseline-url "$BASELINE_URL" \
  --candidate-url "$CANDIDATE_URL" \
  --output-json "${ARTIFACTS_DIR}/results.json" \
  --output-csv "${ARTIFACTS_DIR}/results.csv"
