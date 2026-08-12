#!/usr/bin/env bash
# Run a scenario across arms x runs, one run per process.
#
#   scripts/run_matrix.sh incident 5 baseline treatment treatment-nosupersede treatment-subrecall
#
# One process per run, with Ollama restarted between them. Both matter:
#
#   - Ollama wedges under sustained load (ollama#15950). It accepts connections
#     and never answers while /api/version still returns 200, so a wedge looks
#     exactly like a slow generation until the turn deadline fires. Restarting
#     between runs is the only reliable prophylactic found so far.
#   - A crashed or deadlocked run takes only its own process with it. The
#     harness writes each run's JSON before the next starts, so a failure
#     halfway through a 20-run matrix costs one run, not the set.
#
# The run index is preserved in the filename so the judge still pairs arms
# correctly for a given index.
set -uo pipefail

SCENARIO="${1:?usage: run_matrix.sh SCENARIO RUNS ARM [ARM...]}"
RUNS="${2:?usage: run_matrix.sh SCENARIO RUNS ARM [ARM...]}"
shift 2
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || { echo "no arms given" >&2; exit 2; }

cd "$(dirname "$0")/.."
LOG="results/matrix-${SCENARIO}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p results
echo "scenario=$SCENARIO runs=$RUNS arms=${ARMS[*]}" | tee "$LOG"

wait_for_ollama() {
  # Poll /api/generate, NOT /api/version: the control endpoints answer 200 on a
  # wedged server, so only a real generation proves it can serve.
  for _ in $(seq 1 20); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 \
      -d '{"model":"qwen3:8b","prompt":"hi","stream":false,"options":{"num_predict":3}}' \
      http://localhost:11434/api/generate 2>/dev/null || true)
    [ "$code" = "200" ] && return 0
    sleep 5
  done
  return 1
}

failed=0
for i in $(seq 0 $((RUNS - 1))); do
  for arm in "${ARMS[@]}"; do
    echo "=== restarting ollama before ${arm}:${i} ===" | tee -a "$LOG"
    brew services restart ollama >/dev/null 2>&1 || true
    if ! wait_for_ollama; then
      echo "!!! ollama never became ready; skipping ${arm}:${i}" | tee -a "$LOG"
      failed=$((failed + 1))
      continue
    fi
    echo "=== ${arm} run ${i} ===" | tee -a "$LOG"
    caffeinate -dimsu uv run python -m demo.harness \
      --scenario "$SCENARIO" --only "${arm}:${i}" 2>&1 | tee -a "$LOG"
    # Deliberately not `set -e`: one bad run must not abandon the matrix.
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!!! ${arm}:${i} exited nonzero" | tee -a "$LOG"; failed=$((failed + 1)); }
  done
done

echo "=== matrix complete; ${failed} run(s) failed ===" | tee -a "$LOG"
echo "aggregate: uv run python -m demo.aggregate results/raw" | tee -a "$LOG"
