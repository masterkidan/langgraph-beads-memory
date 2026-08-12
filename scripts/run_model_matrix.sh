#!/usr/bin/env bash
# Model x arm x run matrix for one scenario.
#
#   scripts/run_model_matrix.sh incident 3 "qwen3:8b qwen3.5:9b gemma4:12b" "baseline treatment"
#
# The question this answers is per-model and paired: within a given model, does
# the memory harness change accuracy and token cost? Pooling ACROSS models would
# attribute the model's own strength to the harness, which is why
# demo/aggregate.py refuses to do it and demo/compare_models.py compares
# within-model deltas instead.
#
# Runs land in results/matrix/<model>/ so each model's runs stay separable.
set -uo pipefail

SCENARIO="${1:?usage: run_model_matrix.sh SCENARIO RUNS "MODELS" "ARMS"}"
RUNS="${2:?}"; MODELS="${3:?}"; ARMS="${4:?}"

cd "$(dirname "$0")/.."
mkdir -p results/matrix
LOG="results/matrix/run-$(date +%Y%m%d-%H%M%S).log"
echo "scenario=$SCENARIO runs=$RUNS models=$MODELS arms=$ARMS" | tee "$LOG"

wait_for_ollama() {  # probe generation, not /api/version — control endpoints lie
  for _ in $(seq 1 24); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 40 \
      -d "{\"model\":\"$1\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":3}}" \
      http://localhost:11434/api/generate 2>/dev/null || true)
    [ "$code" = "200" ] && return 0
    sleep 5
  done
  return 1
}

failed=0
for model in $MODELS; do
  dest="results/matrix/$(echo "$model" | tr ':/' '__')"
  mkdir -p "$dest"
  for i in $(seq 0 $((RUNS - 1))); do
    for arm in $ARMS; do
      echo "=== $model / $arm / run $i ===" | tee -a "$LOG"
      brew services restart ollama >/dev/null 2>&1 || true
      if ! wait_for_ollama "$model"; then
        echo "!!! ollama not ready; skipping $model/$arm/$i" | tee -a "$LOG"
        failed=$((failed + 1)); continue
      fi
      BEADS_DEMO_MODEL="$model" caffeinate -dimsu uv run python -m demo.harness \
        --scenario "$SCENARIO" --only "${arm}:${i}" 2>&1 | tee -a "$LOG"
      [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!!! ${model}/${arm}:${i} nonzero" | tee -a "$LOG"; failed=$((failed+1)); }
      # Move this run's output under the model's directory before the next run
      # overwrites results/raw.
      mv results/raw/*.json "$dest"/ 2>/dev/null || true
    done
  done
done

echo "=== matrix complete; ${failed} failure(s) ===" | tee -a "$LOG"
echo "report: uv run python -m demo.compare_models results/matrix" | tee -a "$LOG"
