"""Regression tests for the demo's Ollama client lifecycle.

Both tests exist because of bugs found on 2026-08-11, and neither needs a live
Ollama: client construction is pure configuration.
"""

from demo.llm import _LLM_CACHE, MODEL, NUM_CTX, close_llms, make_llm


class TestClientCaching:
    def setup_method(self):
        close_llms()

    def teardown_method(self):
        close_llms()

    def test_same_parameters_reuse_one_client(self):
        """Every call site used to build its own client — per turn and per
        sub-agent invocation — leaving ~24 unclosed connection pools per run."""
        assert make_llm() is make_llm()
        assert len(_LLM_CACHE) == 1

    def test_different_parameters_get_distinct_clients(self):
        a, b = make_llm(temperature=0.0), make_llm(temperature=0.7)
        assert a is not b
        assert a._client._client is not b._client._client

    def test_close_empties_the_cache(self):
        make_llm()
        close_llms()
        assert not _LLM_CACHE

    def test_cache_key_includes_the_model(self):
        """`num_ctx` is part of the key, not just the model.

        Two runs at different context windows must not share a client: the
        window is baked into the constructed ChatOllama, so a cached instance
        from an earlier setting would silently apply the wrong one — which is
        exactly the class of bug that let every prior run truncate at 2048
        without anyone noticing.
        """
        make_llm()
        assert next(iter(_LLM_CACHE)) == (MODEL, 0.0, False, NUM_CTX)


class TestRecoveryRebuild:
    def teardown_method(self):
        close_llms()

    def test_rebuild_leaves_a_usable_client(self):
        """The bug this pins: recovery used to set `_client = None`, and
        langchain_ollama raises RuntimeError on a null client instead of
        rebuilding it, so every retry failed before touching the network."""
        llm = make_llm()
        llm._rebuild_clients()
        assert llm._client is not None
        assert llm._async_client is not None

    def test_rebuild_swaps_in_a_fresh_pool(self):
        llm = make_llm()
        before = llm._client._client
        llm._rebuild_clients()
        assert llm._client._client is not before
