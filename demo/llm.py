"""Model configuration for the demo. One model for agents AND judge."""

import contextlib
import os
import threading

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


# Guards client rebuilds. The LLM is shared across ToolNode's worker threads
# (see make_llm), so a rebuild must not race another thread's rebuild.
_REBUILD_LOCK = threading.Lock()

# One client per (model, temperature, reasoning) instead of one per turn and per
# sub-agent invocation. See make_llm for why this mattered.
_LLM_CACHE: dict[tuple, "ResilientChatOllama"] = {}
_CACHE_LOCK = threading.Lock()


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

    CORRECTION (2026-08-11): the rebuild used to set `_client = None` and retry.
    That never worked. langchain_ollama's `_create_chat_stream` raises
    `RuntimeError("Ollama sync client is not initialized")` on a null client
    rather than reconstructing it, so every retry failed instantly and the
    wrapper only ever converted a transport error into a RuntimeError. Verified
    directly: nulling `_client` and calling `_generate` raises that RuntimeError
    before any network I/O. Any earlier claim that this policy produced a clean
    run was wrong — those runs were clean because the server did not wedge.

    The rebuild now re-runs `_set_clients()`, the `model_validator(mode="after")`
    that built the clients in the first place, which is the only construction
    path that actually produces working ones.
    """

    def _rebuild_clients(self) -> None:
        """Reconstruct the underlying ollama/httpx clients, dropping pooled
        connections that may point at a dead runner.

        The old httpx client is closed so its sockets are released now rather
        than at GC. That can disturb a request in flight on another thread — but
        this only runs after `ensure_healthy` has found the server wedged, at
        which point any in-flight request is already lost.
        """
        old = getattr(self, "_client", None)
        self._set_clients()
        inner = getattr(old, "_client", None)
        if inner is not None:
            with contextlib.suppress(Exception):  # best effort only
                inner.close()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as first:  # noqa: BLE001 - any transport failure is a candidate
            if not ensure_healthy(self.model):
                raise
            with _REBUILD_LOCK:
                self._rebuild_clients()
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

    Instances are CACHED per (model, temperature, reasoning). Every call site
    used to build a fresh one, and each carries its own `ollama.Client` wrapping
    its own `httpx.Client` connection pool. That was per turn and per sub-agent
    invocation, not per run: `build_baseline`'s `invoke()` constructs the root
    agent plus all three researcher agents on every turn, so a 6-turn run built
    ~24 pools, three quarters of them for researchers that were never called on
    that turn. None were ever closed, so their sockets lingered — which is why a
    single idle run showed 11 ESTABLISHED connections to Ollama.

    That churn is a plausible contributor to the wedging that has dogged this
    demo, and it is pure waste regardless: the client is stateless configuration.
    Sharing is safe here — `httpx.Client` is documented thread-safe, and
    ToolNode's workers only ever issue requests through it.
    """
    key = (MODEL, temperature, reasoning)
    with _CACHE_LOCK:
        llm = _LLM_CACHE.get(key)
        if llm is None:
            llm = ResilientChatOllama(
                model=MODEL,
                temperature=temperature,
                reasoning=reasoning,
                # 120s, not 300s: healthy calls are 6-40s, so a longer bound only
                # delays discovering a wedge. Recovery, not patience, fixes this.
                client_kwargs={"timeout": CHAT_TIMEOUT_S},
            )
            _LLM_CACHE[key] = llm
        return llm


def close_llms() -> None:
    """Close every cached client and empty the cache.

    Called from the harness between runs so one run's sockets cannot outlive it
    and be counted against the next.
    """
    with _CACHE_LOCK:
        for llm in _LLM_CACHE.values():
            inner = getattr(getattr(llm, "_client", None), "_client", None)
            if inner is not None:
                with contextlib.suppress(Exception):  # best effort only
                    inner.close()
        _LLM_CACHE.clear()
