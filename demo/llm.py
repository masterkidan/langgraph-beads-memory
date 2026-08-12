"""Model configuration for the demo. One model for agents AND judge."""

import contextlib
import os

from langchain_ollama import ChatOllama

from demo.resilient import CHAT_TIMEOUT_S, ensure_healthy

# qwen3:8b. 4b was tried and reverted: despite reasoning=False it writes long
# chain-of-thought into its responses ("Okay, let me figure out what's going on
# here..."), producing ~12,000 output tokens per run against 8b's ~1,400. Turns
# took 200s+ and hit the 900s deadline. An earlier microbenchmark suggested 4b
# was ~1.8x faster, but that capped num_predict at 250, which hid the verbosity
# in exactly the open-ended agent turns where it matters.
MODEL = os.environ.get("BEADS_DEMO_MODEL", "qwen3:8b")


# CORRECTION: an earlier comment here claimed client-side timeouts do not work.
# That was wrong — the test behind it was faulty. Verified against a socket that
# accepts and never replies, client_kwargs={"timeout": 5} raises ReadTimeout in
# 5.0s exactly.
#
# It is still not sufficient on its own. httpx's is a *per-read* timeout, so a
# response that trickles bytes resets it indefinitely — which is how a real run
# hung for 8.5 minutes with this set to 300s. The harness therefore also applies
# a hard per-turn deadline; see demo/harness.py.
REQUEST_TIMEOUT_S = float(os.environ.get("BEADS_DEMO_TIMEOUT", "300"))


class ResilientChatOllama(ChatOllama):
    """ChatOllama that repairs a wedged server instead of blocking on it.

    Ollama accepts a connection and then never processes the request; the client
    sits in `sock_recv` while `/api/version` still answers 200. There is no retry
    policy on ChatOllama (it exposes no retry fields), so before this a wedge
    burned the 900s turn deadline and was never repaired.

    A bare retry does not help — it dispatches onto the same wedged server, often
    onto the same pooled httpx connection. So on failure this probes
    `/api/generate` (the control endpoints lie), restarts the daemon if it is
    genuinely wedged, rebuilds the client so no dead pooled connection is reused,
    and retries once. A second failure is raised: the harness records an errored
    turn and moves on, which is better than looping against a broken server.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as first:  # noqa: BLE001 - any transport failure is a candidate
            if not ensure_healthy(self.model):
                raise
            # Drop pooled connections: the old ones may point at a dead runner.
            for attr in ("_client", "_async_client"):
                if hasattr(self, attr):
                    with contextlib.suppress(Exception):  # best effort only
                        object.__setattr__(self, attr, None)
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as second:
                raise second from first


def make_llm(temperature: float = 0.0, reasoning: bool | None = False) -> ChatOllama:
    """Build the demo LLM.

    `reasoning` defaults to False. qwen3 is a reasoning model, and with its
    default thinking mode ON it was measured to silently drop facts during
    extraction-style prompts (it returned only 1 of 2 facts, reproducibly, at
    temperature 0 across 3 runs; 773 reasoning tokens spent, second fact lost).
    That matters for fairness, not just quality: the LangMem baseline depends on
    extraction, so leaving thinking on would hobble the baseline for a reason
    that has nothing to do with the memory architecture under test — an
    accidental strawman. Disabling it fixed the same prompt in 18 tokens.

    Pass `reasoning=True` to opt back in for a specific call site.
    """
    return ResilientChatOllama(
        model=MODEL,
        temperature=temperature,
        reasoning=reasoning,
        # 120s, not 300s: healthy calls are 6-40s, so a longer bound only delays
        # discovering a wedge. Recovery, not patience, is what fixes this.
        client_kwargs={"timeout": CHAT_TIMEOUT_S},
    )
