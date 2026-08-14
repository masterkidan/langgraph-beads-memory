#!/usr/bin/env bash
# One-shot local setup for the benchmark: toolchain, Postgres, Ollama, models,
# then the pre-flight gate.
#
#   scripts/setup_local.sh                      # default model set
#   scripts/setup_local.sh gemma4:12b glm4:9b   # explicit set
#
# Everything here is idempotent — re-running it on a warm machine is a no-op
# plus a smoke test. Postgres is the container from docker-compose.yml, not a
# host install; nothing in this script installs a service onto the machine.
#
# The smoke test at the end is a GATE, not a formality. A model that cannot
# emit tool calls produces an empty treatment arm, which reads as "the memory
# layer lost" when it actually means "pick a better model". Measured here:
# glm4:9b advertises `tools` in /api/show and its template handles them, yet it
# returned zero tool calls on the gate prompt across two runs. Capability
# metadata is not evidence; the gate is.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(gemma4:12b qwen3.5:9b lfm2.5:8b)
EMBEDDER="nomic-embed-text"

say() { printf '\n=== %s ===\n' "$*"; }
fail() { printf '!!! %s\n' "$*" >&2; exit 1; }

# --- toolchain -------------------------------------------------------------
# uv installs to ~/.local/bin, which is not on a default macOS PATH. Add it
# here rather than making every caller remember to.
export PATH="$HOME/.local/bin:$PATH"

say "toolchain"
command -v uv >/dev/null || fail "uv not found. Install it (brew install uv, or
  https://astral.sh/uv) and re-run. The project needs Python >=3.13; system
  python on macOS is 3.9, so uv is what supplies the interpreter too."
echo "uv $(uv --version | awk '{print $2}')"
uv sync --extra demo || fail "uv sync failed"

# --- postgres --------------------------------------------------------------
say "postgres (docker)"
command -v docker >/dev/null || fail "docker not found; Postgres runs as a container"
docker info >/dev/null 2>&1 || fail "docker daemon is not running"
docker compose up -d || fail "docker compose up failed"

# Wait for readiness rather than assuming: the container reports Started well
# before Postgres accepts connections, and the first harness run would
# otherwise fail on connect.
for _ in $(seq 1 30); do
  docker compose exec -T postgres pg_isready -U beads -d beads >/dev/null 2>&1 && break
  sleep 2
done
docker compose exec -T postgres pg_isready -U beads -d beads >/dev/null 2>&1 \
  || fail "postgres never became ready"
echo "postgres ready on localhost:5433"

# --- ollama ----------------------------------------------------------------
say "ollama"
command -v ollama >/dev/null || fail "ollama not found (brew install ollama)"
# Prefer the brew service: scripts/run_model_matrix.sh restarts Ollama between
# runs via `brew services restart`, and that only works if brew owns it.
if ! curl -sf --max-time 5 http://localhost:11434/api/version >/dev/null 2>&1; then
  brew services start ollama >/dev/null 2>&1 || nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf --max-time 5 http://localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -sf --max-time 5 http://localhost:11434/api/version >/dev/null 2>&1 \
  || fail "ollama did not come up"
brew services list 2>/dev/null | grep -q '^ollama.*started' \
  || echo "note: ollama is not brew-managed, so run_model_matrix.sh cannot restart it
  between runs. It wedges under sustained load; expect to babysit long matrices."

# --- models ----------------------------------------------------------------
say "models"
have() { ollama list | awk 'NR>1 {print $1}' | grep -qx "$1"; }
for m in "$EMBEDDER" "${MODELS[@]}"; do
  # `ollama list` prints the embedder as nomic-embed-text:latest
  probe="$m"; [[ "$m" == *:* ]] || probe="$m:latest"
  if have "$probe"; then
    echo "have $probe"
  else
    echo "pulling $m"
    ollama pull "$m" || fail "pull failed: $m"
  fi
done

# --- clean slate -----------------------------------------------------------
say "database reset"
./scripts/reset_db.sh || fail "reset_db.sh failed"

# --- gate ------------------------------------------------------------------
# Deliberately not `set -e` around this loop: report every model's verdict,
# then exit nonzero once. Knowing which of four models failed beats aborting
# on the first.
gate_failed=0
for m in "${MODELS[@]}"; do
  say "smoke test: $m"
  if BEADS_DEMO_MODEL="$m" uv run python -m demo.smoke_test; then
    echo "GATE PASS: $m"
  else
    echo "GATE FAIL: $m — do not benchmark this model; its arms are not comparable"
    gate_failed=$((gate_failed + 1))
  fi
done

say "setup complete"
if [ "$gate_failed" -gt 0 ]; then
  echo "$gate_failed model(s) failed the gate. Drop them from the set before running."
  exit 1
fi
echo "next: scripts/run_model_matrix.sh incident 3 \"${MODELS[*]}\" \"baseline treatment\""
