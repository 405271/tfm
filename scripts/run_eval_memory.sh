#!/usr/bin/env bash

set -euo pipefail

BASELINE_PORT="${BASELINE_PORT:-8000}"
CANDIDATE_PORT="${CANDIDATE_PORT:-8001}"
BASELINE_URL="http://127.0.0.1:${BASELINE_PORT}"
CANDIDATE_URL="http://127.0.0.1:${CANDIDATE_PORT}"
CASES_PATH="${CASES_PATH:-app/data/eval_memory_cases.json}"

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

echo "Levantando baseline sin memoria en ${BASELINE_URL}"
MEMORI_ENABLED=false \
uvicorn app.api.main:app --host 127.0.0.1 --port "$BASELINE_PORT" \
  > artifacts/eval_memory_baseline.log 2>&1 &
BASELINE_PID=$!

echo "Levantando candidate con memoria en ${CANDIDATE_URL}"
MEMORI_ENABLED=true \
uvicorn app.api.main:app --host 127.0.0.1 --port "$CANDIDATE_PORT" \
  > artifacts/eval_memory_candidate.log 2>&1 &
CANDIDATE_PID=$!

wait_for_health "$BASELINE_URL"
wait_for_health "$CANDIDATE_URL"

python scripts/eval_memory.py \
  --cases "$CASES_PATH" \
  --baseline-url "$BASELINE_URL" \
  --candidate-url "$CANDIDATE_URL"
