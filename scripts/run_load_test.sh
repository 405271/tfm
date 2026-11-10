#!/usr/bin/env bash
# Prueba de latencia y concurrencia (H4) en las dos variantes del backend con Memori:
#   sync    -> MEMORI_OFFLOAD_SYNC=false (original: Memori bloquea el event loop)
#   offload -> MEMORI_OFFLOAD_SYNC=true  (llamadas síncronas de Memori en un hilo)
# Ejecutar con la máquina libre (sin otras evaluaciones), porque mide latencias.
# Argumentos extra -> scripts/load_test.py (p. ej. --requests 50 para un piloto).

set -euo pipefail

PORT="${LOAD_PORT:-8002}"
OUT_DIR="${LOAD_OUT_DIR:-artifacts/eval_runs/load}"
PYTHON="${PYTHON:-python}"
VARIANTS="${VARIANTS:-sync offload}"
mkdir -p "$OUT_DIR"

export EVAL_MODE=true
export FAQ_PATH=/dev/null
export GLOSSARY_PATH=/dev/null
export ESCALATION_ENABLED=false
export ISSUE_CONTEXT_ENABLED=false
export LLM_TEMPERATURE="${EVAL_LLM_TEMPERATURE:-0}"
export EVAL_EXTRACTION_CACHE="${EVAL_EXTRACTION_CACHE:-artifacts/eval_runs/extraction_cache.jsonl}"
export MEMORI_ENABLED=true

for variant in $VARIANTS; do
  offload=false
  [[ "$variant" == "offload" ]] && offload=true
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "El puerto ${PORT} ya está en uso (¿servidor de una prueba anterior?). Deténlo antes de seguir." >&2
    exit 1
  fi
  rm -f "${OUT_DIR}/memori_${variant}.db"
  echo "== Variante ${variant} (MEMORI_OFFLOAD_SYNC=${offload}) en :${PORT}"
  MEMORI_OFFLOAD_SYNC="$offload" \
  MEMORI_STORAGE_URL="sqlite:///${OUT_DIR}/memori_${variant}.db" \
    "$PYTHON" -m uvicorn app.api.main:app --host 127.0.0.1 --port "$PORT" \
    > "${OUT_DIR}/server_${variant}.log" 2>&1 &
  SERVER_PID=$!
  # Un servidor bloqueado (variante sync bajo carga) puede no atender SIGTERM: se fuerza con -9.
  trap 'kill $SERVER_PID 2>/dev/null; sleep 3; kill -9 $SERVER_PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 1
  done
  # Si una variante falla (p. ej. el servidor se bloquea), se sigue con la siguiente.
  "$PYTHON" scripts/load_test.py --url "http://127.0.0.1:${PORT}" --output-dir "$OUT_DIR" --label "$variant" "$@" \
    || echo "La variante ${variant} terminó con error; ver ${OUT_DIR}/server_${variant}.log" >&2
  kill "$SERVER_PID" 2>/dev/null || true
  sleep 3
  kill -9 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
done
