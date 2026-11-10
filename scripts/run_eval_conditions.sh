#!/usr/bin/env bash
# Ejecuta las condiciones C0-C4 sobre el dataset v2.
#
# Levanta dos backends en modo evaluación:
#   - baseline (MEMORI_ENABLED=false) para C0, C1, C2 y C4;
#   - memori   (MEMORI_ENABLED=true)  para C3, con una BD propia por ejecución.
# Los argumentos extra se pasan a scripts/run_conditions.py, por ejemplo:
#   RUN_NAME=piloto bash scripts/run_eval_conditions.sh --limit 2 --runs 1 --history-lengths 5
#   RUN_NAME=final  bash scripts/run_eval_conditions.sh --history-lengths 5 20 --runs 3
# Repetir el mismo comando reanuda la ejecución (las filas completas no se repiten).
# RUNNER=scripts/simulate_user.py usa los mismos backends para el usuario simulado (RR/STR):
#   RUN_NAME=final_sim RUNNER=scripts/simulate_user.py bash scripts/run_eval_conditions.sh

set -euo pipefail

RUN_NAME="${RUN_NAME:-default}"
BASELINE_PORT="${BASELINE_PORT:-8000}"
MEMORI_PORT="${MEMORI_PORT:-8001}"
RUN_DIR="artifacts/eval_runs/${RUN_NAME}"
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-scripts/run_conditions.py}"

mkdir -p "$RUN_DIR"

cleanup() {
  for pid in "${BASELINE_PID:-}" "${MEMORI_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

wait_for_health() {
  local url="$1/health"
  for _ in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "No respondió health en $url" >&2
  exit 1
}

# Configuración común: mismo LLM, mismo prompt, temperatura 0 y sin atajos de la
# tubería, para que todas las preguntas lleguen al LLM.
export EVAL_MODE=true
export FAQ_PATH=/dev/null
export GLOSSARY_PATH=/dev/null
export ESCALATION_ENABLED=false
export ISSUE_CONTEXT_ENABLED=false
export LLM_TEMPERATURE="${EVAL_LLM_TEMPERATURE:-0}"
# La caché de extracción se comparte entre ejecuciones: la clave incluye modelo y prompt.
export EVAL_EXTRACTION_CACHE="${EVAL_EXTRACTION_CACHE:-artifacts/eval_runs/extraction_cache.jsonl}"

echo "Levantando baseline (sin memoria) en :${BASELINE_PORT}"
MEMORI_ENABLED=false \
  "$PYTHON" -m uvicorn app.api.main:app --host 127.0.0.1 --port "$BASELINE_PORT" \
  > "${RUN_DIR}/baseline.log" 2>&1 &
BASELINE_PID=$!

echo "Levantando memori (C3) en :${MEMORI_PORT}, BD ${RUN_DIR}/memori.db"
MEMORI_ENABLED=true \
MEMORI_STORAGE_URL="sqlite:///${RUN_DIR}/memori.db" \
  "$PYTHON" -m uvicorn app.api.main:app --host 127.0.0.1 --port "$MEMORI_PORT" \
  > "${RUN_DIR}/memori.log" 2>&1 &
MEMORI_PID=$!

wait_for_health "http://127.0.0.1:${BASELINE_PORT}"
wait_for_health "http://127.0.0.1:${MEMORI_PORT}"

"$PYTHON" "$RUNNER" \
  --baseline-url "http://127.0.0.1:${BASELINE_PORT}" \
  --memori-url "http://127.0.0.1:${MEMORI_PORT}" \
  --output-dir "$RUN_DIR" \
  --entity-prefix "$RUN_NAME" \
  "$@"
