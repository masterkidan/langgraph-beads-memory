"""LongMemEval against this memory layer — a public benchmark, not a designed demo.

Why this exists. `results/README.md` opens by conceding that the two built-in
scenarios are "a designed demonstration, not a neutral benchmark": they were
written to exercise a mechanism this library has. That is a fair caveat and it
caps how far any number here can travel. LongMemEval is external, MIT-licensed,
and both Zep/Graphiti and Mem0 publish on it, so a score here is comparable to
something rather than only to itself.

    uv run python -m demo.longmemeval --data <oracle.json> --type knowledge-update
    uv run python -m demo.longmemeval --data <oracle.json> --limit 10 --arms fullcontext memory

Two arms, and the comparison is deliberately about COST at equal accuracy,
because that is what this library has actually been measured to buy:

  fullcontext  every haystack session pasted into the prompt. The accuracy
               ceiling, and the token bill you are trying to avoid.
  memory       sessions ingested through the real capture path, then k facts
               retrieved and injected. Same reader model, same question.

WHAT THIS DOES NOT TEST, and it matters for `knowledge-update` specifically.
A `supersedes` edge is only ever written when an agent calls `remember_fact`
with one. Replaying a transcript has no agent, so nothing emits those edges and
the store ends up holding the stale value and its correction side by side, both
active. So this measures CAPTURE + RANKING — does retrieval surface the newer
claim — and NOT typed invalidation, which is the mechanism the knowledge-update
category would otherwise be a natural fit for. `--ingest agent` runs a real
model call per session so `remember_fact` becomes reachable; it is far slower
and its coverage depends on the model choosing to call the tool, which was
measured at 6-8% of writes on the built-in scenarios.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import uuid

import psycopg
from langchain_core.messages import AIMessage, HumanMessage

DSN = "postgresql://beads:beads@localhost:5433/beads"


def load(path: str, qtype: str | None, limit: int | None) -> list[dict]:
    records = json.loads(pathlib.Path(path).read_text())
    if qtype:
        records = [r for r in records if r.get("question_type") == qtype]
    return records[:limit] if limit else records


def _sessions_in_order(rec: dict) -> list[list[dict]]:
    """Haystack sessions oldest-first.

    Order is load-bearing rather than cosmetic: every mechanism that decides
    which of two conflicting claims is current — the supersede cascade's
    created_at predicate, and any recency signal — reads write order. Feeding
    the correction before the claim it corrects would invert the thing this
    benchmark's `knowledge-update` category is asking about.
    """
    dates = rec.get("haystack_dates") or []
    sessions = rec["haystack_sessions"]
    if len(dates) == len(sessions):
        return [s for _, s in sorted(zip(dates, sessions, strict=False), key=lambda p: p[0])]
    return sessions


def ingest_replay(mw, rec: dict) -> int:
    """Feed a transcript through the real capture hooks, with no generation.

    Calls `before_model` / `after_model` exactly as the middleware would see
    them, so splitting, the substantive filter and directive classification are
    all the shipped code rather than a reimplementation. One message is passed
    per call rather than the whole history: `before_model` re-scans every
    HumanMessage it is given and re-embeds each one before the content-derived
    id dedupes it at the database, so passing the full list would make ingestion
    quadratic in embed calls for an identical store.
    """
    turns = 0
    for session in _sessions_in_order(rec):
        for turn in session:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if turn.get("role") == "user":
                mw.before_model({"messages": [HumanMessage(content=content)]}, None)
            else:
                mw.after_model({"messages": [AIMessage(content=content)]}, None)
            turns += 1
    return turns



_STATS: dict = {"sessions": 0, "ingest_errors": 0, "tool_errors": 0,
                 "supersede_attempts": 0, "auto_links": 0, "last_error": ""}

RECONCILE_PROMPT = (
    "Below is a conversation with the user, and the durable facts you already hold.\n"
    "Record any NEW durable fact the user stated, using remember_fact.\n"
    "IMPORTANT: if something in the conversation CHANGES or REPLACES a fact you already "
    "hold, call remember_fact with the new value and set relates_to to that fact's short "
    "id (like fact-a3f8b2c1) and relation to 'supersedes'.\n"
    "Only record facts about the user. Be brief. If nothing is new, reply DONE.\n"
)


# Teaches the SHAPE of the graph, not just the existence of the tool. The terse
# version (RECONCILE_PROMPT) named the tool and the supersedes relation and got
# 9 supersede attempts across 78 questions that were ALL supersede-shaped — so
# "the model knows the tool exists" was never the constraint.
#
# Two things it is asked to do that the terse version left implicit, each aimed
# at a measured failure:
#
#   self-contained bodies — the fact graph lost 14 points to a store that keeps
#   whole turns, because a split claim like "25:50" carries no subject and only
#   matches a query when the surrounding turn came with it. A body that names
#   its own subject does not need the co-occurrence that splitting destroyed.
#
#   check-before-write — the supersede miss is not a refusal, it is a failure to
#   LOOK. Asking for an explicit scan of held facts for the same subject turns a
#   volunteered observation into a step.
GRAPH_PROMPT = (
    "You maintain a memory graph about the user. Each memory is one node; "
    "`supersedes` links replace an outdated node.\n\n"
    "HOW TO WRITE A NODE\n"
    "- exactly one claim per node\n"
    "- it must stand alone with no conversation around it: name the subject "
    "explicitly. Write 'The user's 5K personal best is 25:50', never '25:50'.\n"
    "- keep the number and its units\n\n"
    "HOW TO LINK\n"
    "- BEFORE recording, scan the held facts for one about the SAME subject.\n"
    "- if you find one and the value has changed, you MUST pass "
    "relates_to=<its short id> and relation='supersedes'.\n"
    "- a changed value is a supersedes link, not a second node.\n\n"
    "EXAMPLE\n"
    "held:  [fact-1a2b3c4d] The user owns 3 bikes.\n"
    "user says: 'picked up another bike last week'\n"
    "-> remember_fact(body='The user owns 4 bikes.', "
    "relates_to='fact-1a2b3c4d', relation='supersedes')\n\n"
    "Record only durable facts about the user. Reply DONE if nothing is new.\n"
)


def _auto_link(store, ns, embedder, body: str, cand_k: int = 8) -> str | None:
    """Find a held fact that `body` supersedes, without asking the model.

    MEASURED reason this exists. Under GRAPH_PROMPT the model wrote exactly the
    node it was asked for — "The user's 5K personal best is 25:50." — and linked
    nothing, because the fact it had to supersede ("...27:12") sat at rank 21 of
    ~180 while the top 8 shown to it were captured assistant prose ("Incorporate
    Interval Training", "tips to improve your endurance"). It was never shown the
    target. The 12% supersede rate was a retrieval failure in the reconciliation
    pass, not a refusal.

    Two changes fix it and neither needs a model:

      candidates come from the PROPOSED FACT, not the conversation. Two claims
      about one subject are near-neighbours of each other even when both are far
      from the surrounding chat.

      the contradiction is READ, not inferred. `contested_values` already answers
      "does the new text assert a different value of the same shape?" — it is
      what the supersede cascade uses, and it is exact where cosine is blind:
      27:12 against 25:50 is 0.9-something similar and definitively different.
    """
    from beads_memory.segment import contested_values

    for cand in store.search(ns.id, embedder.embed(body), k=cand_k):
        if cand.body.strip() == body.strip():
            continue
        stale, new = contested_values(cand.body, body)
        if stale and new:
            return cand.id
    return None


def ingest_agent(mw, store, ns, embedder, llm, rec, prompt: str | None = None,
                 auto_link: bool = False) -> int:
    """Replay capture, plus one reconciliation call per session that can write
    typed edges.

    Why this exists. A `supersedes` edge is only ever written by an agent calling
    `remember_fact`, so a pure transcript replay leaves the stale value and its
    correction both active and this library competes on cosine ranking alone —
    which measured 59.0% on knowledge-update, tying LongMemEval's BM25 reference
    and losing to the stock store by 14 points. This makes the mechanism
    reachable so the question "does typed invalidation close that gap?" can be
    answered rather than assumed.

    Deliberately NOT a full agent turn over the session. Letting an agent
    generate its own replies would change the captured transcript, and then this
    arm would no longer be reading the same haystack as bm25/baseline/fullcontext
    — the comparison would be over different inputs. Passive capture stays
    identical to `ingest_replay`; the only added variable is a pass that may
    write edges. That also makes it a prototype of deferred normalisation: an LLM
    reconciling what capture wrote, off the hot path.
    """
    from beads_memory.ids import short_id

    remember = mw.tools[0]
    _STATS["sessions"] += 1
    calls = 0
    for session in _sessions_in_order(rec):
        text = []
        for turn in session:
            c = (turn.get("content") or "").strip()
            if not c:
                continue
            if turn.get("role") == "user":
                mw.before_model({"messages": [HumanMessage(content=c)]}, None)
            else:
                mw.after_model({"messages": [AIMessage(content=c)]}, None)
            # USER turns only in the reconciliation prompt. Assistant turns are
            # long LLM prose and carry no durable user fact, and including them
            # made this prompt ~14k chars — every call then exceeded the 120s
            # bound and the whole pass silently produced nothing.
            if turn.get("role") == "user":
                text.append(c)
        convo = "\n".join(text)
        if not convo:
            continue
        # Existing facts WITH short ids — supersedes is unusable without them,
        # since the tool resolves its target by short id.
        held = store.search(ns.id, embedder.embed(convo), k=8)
        block = "\n".join(f"- [{short_id(f.id)}] ({f.kind}) {f.body}" for f in held)
        msg = (f"{prompt or RECONCILE_PROMPT}\n--- facts you hold ---\n{block}\n\n"
               f"--- conversation ---\n{convo}")
        try:
            resp = llm.bind_tools([remember]).invoke([HumanMessage(content=msg)])
        except Exception as e:  # noqa: BLE001 - counted, never silent
            # This was `continue` and it hid the experiment failing: every
            # reconciliation call was timing out (the prompt carries a whole
            # session plus 12 held facts) and the run reported 0 supersedes
            # edges as though the model had simply declined to write any.
            _STATS["ingest_errors"] += 1
            _STATS["last_error"] = f"{type(e).__name__}: {e}"[:120]
            continue
        for tc in getattr(resp, "tool_calls", None) or []:
            args = dict(tc.get("args") or {})
            if args.get("relates_to"):
                _STATS["supersede_attempts"] += 1
            elif auto_link and args.get("body"):
                target = _auto_link(store, ns, embedder, args["body"])
                if target is not None:
                    from beads_memory.ids import short_id as _sid

                    args["relates_to"] = _sid(target)
                    args["relation"] = "supersedes"
                    _STATS["auto_links"] += 1
            tc = {**tc, "args": args}
            try:
                remember.invoke(tc["args"])
                calls += 1
            except Exception:  # noqa: BLE001 - a bad tool call is the model's, not ours
                _STATS["tool_errors"] += 1
    return calls


def _transcript(rec: dict) -> str:
    out = []
    for session in _sessions_in_order(rec):
        for turn in session:
            c = (turn.get("content") or "").strip()
            if c:
                out.append(f"{turn.get('role', '?')}: {c}")
    return "\n".join(out)


ANSWER_PROMPT = (
    "You are answering a question about an earlier conversation with this user.\n"
    "Answer from the information given. Be brief and specific — a short phrase or "
    "a single sentence. If several values appear, prefer the most recent one.\n"
)


def ask(llm, question: str, context: str, question_date: str) -> tuple[str, dict]:
    msg = (
        f"{ANSWER_PROMPT}\nToday is {question_date}.\n\n"
        f"--- what you know ---\n{context}\n\n--- question ---\n{question}"
    )
    resp = llm.invoke([HumanMessage(content=msg)])
    usage = getattr(resp, "usage_metadata", None) or {}
    return str(resp.content).strip(), {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "context_chars": len(context),
    }


JUDGE_PROMPT = (
    "Does the candidate answer convey the same information as the reference?\n"
    "Ignore wording, formatting and extra detail. Say YES if the key fact matches, "
    "NO if it is missing, wrong, or a different value.\n"
    "Reply with exactly one word: YES or NO.\n\n"
    "Question: {q}\nReference: {gold}\nCandidate: {pred}"
)


def judge(llm, question: str, gold: str, pred: str) -> bool:
    """LLM-graded, because substring matching has already produced three false
    readings in this project (see results/README.md — "32x" vs "32 times",
    "50k" vs "$50,000", and a mention scored as a recommendation).

    The same judge grades every arm, so a lenient or strict judge shifts all
    arms together rather than favouring one.
    """
    out = llm.invoke(
        [HumanMessage(content=JUDGE_PROMPT.format(q=question, gold=gold, pred=pred))]
    )
    return str(out.content).strip().upper().startswith("YES")


def run_memory_arm(rec: dict, llm, k: int, ingest: str = "replay") -> tuple[str, dict, dict]:
    from beads_memory import BeadsMemoryMiddleware, BeadsStore, OllamaEmbedder

    schema = f"lme_{uuid.uuid4().hex[:10]}"
    conn = psycopg.connect(DSN, autocommit=True)
    try:
        from pgvector.psycopg import register_vector

        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.execute(f'SET search_path TO "{schema}", public')
        register_vector(conn)
        store = BeadsStore(conn)
        store.init_schema()
        embedder = OllamaEmbedder()
        ns = store.get_or_create_namespace(rec["question_id"])
        mw = BeadsMemoryMiddleware(
            store=store, namespace=ns, embedder=embedder,
            agent_id="root", acting_on_behalf_of="user", k=k,
        )
        if ingest in ("agent", "graph", "autolink"):
            ingest_agent(mw, store, ns, embedder, llm, rec,
                         GRAPH_PROMPT if ingest in ("graph", "autolink") else None,
                         auto_link=(ingest == "autolink"))
        else:
            ingest_replay(mw, rec)

        scored = store.search(ns.id, embedder.embed(rec["question"]), k=k, with_scores=True)
        block = "\n".join(f"- ({f.kind}) {f.body}" for f, _d, _dm in scored)
        total = conn.execute(
            "SELECT count(*), coalesce(sum(length(body)),0) FROM facts WHERE status='active'"
        ).fetchone()
        sup = conn.execute(
            "SELECT count(*) FROM fact_edges WHERE relation='supersedes'"
        ).fetchone()[0]
        retired = conn.execute(
            "SELECT count(*) FROM facts WHERE status='superseded'"
        ).fetchone()[0]
        pred, usage = ask(llm, rec["question"], block, rec.get("question_date", "unknown"))
        return pred, usage, {"facts": total[0], "store_chars": total[1], "injected": len(scored),
                            "supersedes_edges": sup, "retired": retired}
    finally:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.close()


def run_augment_arm(rec: dict, llm, k: int, budget_tokens: int,
                    ingest: str = "replay") -> tuple[str, dict, dict]:
    """Transcript AND memory, sized to one budget — the shape actually shipped.

    Every other arm here is a SUBSTITUTE: `memory` sees only facts, `fullcontext`
    only the transcript. Nobody deploys either. `BeadsMemoryMiddleware` sends the
    windowed raw messages *plus* an injected fact block (middleware.py:399), so
    the real question is not "facts or transcript" but "at a fixed budget, does
    spending part of it on facts beat spending all of it on transcript".

    The transcript is filled to the budget MINUS whatever the facts cost, so the
    two arms are compared at equal total context. That is what makes it a test of
    allocation rather than of who was handed more room — and it is the only
    configuration where memory can be shown to *interfere*, by displacing
    transcript that held the answer.

    The header is not decoration. A truncated transcript looks complete to the
    model: there is no marker saying earlier turns were dropped, so it reads the
    excerpt, finds no mention of the fact, and concludes it was never stated.
    Labelling the excerpt as partial is what lets it know to trust the fact block
    for anything the excerpt does not cover.
    """
    from beads_memory import BeadsMemoryMiddleware, BeadsStore, OllamaEmbedder

    schema = f"lme_{uuid.uuid4().hex[:10]}"
    conn = psycopg.connect(DSN, autocommit=True)
    try:
        from pgvector.psycopg import register_vector

        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.execute(f'SET search_path TO "{schema}", public')
        register_vector(conn)
        store = BeadsStore(conn)
        store.init_schema()
        embedder = OllamaEmbedder()
        ns = store.get_or_create_namespace(rec["question_id"])
        mw = BeadsMemoryMiddleware(store=store, namespace=ns, embedder=embedder,
                                   agent_id="root", acting_on_behalf_of="user", k=k)
        if ingest in ("agent", "graph", "autolink"):
            ingest_agent(mw, store, ns, embedder, llm, rec,
                         GRAPH_PROMPT if ingest in ("graph", "autolink") else None,
                         auto_link=(ingest == "autolink"))
        else:
            ingest_replay(mw, rec)
        facts = store.search(ns.id, embedder.embed(rec["question"]), k=k)
        block = "\n".join(f"- ({f.kind}) {f.body}" for f in facts)

        # ~4 chars per token, the same approximation langchain's
        # count_tokens_approximately uses; exactness is not needed because both
        # arms are cut with the identical rule.
        budget_chars = budget_tokens * 4
        remaining = max(0, budget_chars - len(block))
        tail = _transcript(rec)[-remaining:] if remaining else ""
        context = (
            "## Recent conversation (an excerpt \u2014 earlier turns are NOT shown)\n"
            f"{tail}\n\n"
            "## Durable facts recalled from earlier in this session\n"
            f"{block}"
        )
        pred, usage = ask(llm, rec["question"], context, rec.get("question_date", "unknown"))
        return pred, usage, {"facts": len(facts), "block_chars": len(block),
                             "tail_chars": len(tail)}
    finally:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.close()


def run_tail_arm(rec: dict, llm, budget_tokens: int) -> tuple[str, dict, dict]:
    """The control for `augment`: the same budget, spent entirely on transcript.

    Without this, a good `augment` score proves only that more context helps.
    The pair isolates the allocation decision.
    """
    tail = _transcript(rec)[-(budget_tokens * 4):]
    context = ("## Recent conversation (an excerpt \u2014 earlier turns are NOT shown)\n" + tail)
    pred, usage = ask(llm, rec["question"], context, rec.get("question_date", "unknown"))
    return pred, usage, {"tail_chars": len(tail)}


def run_fullcontext_arm(rec: dict, llm) -> tuple[str, dict, dict]:
    pred, usage = ask(llm, rec["question"], _transcript(rec), rec.get("question_date", "unknown"))
    return pred, usage, {}


_WORD_RE = __import__("re").compile(r"[a-z0-9']+")


def _tok(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def run_bm25_arm(rec: dict, llm, k: int) -> tuple[str, dict, dict]:
    """LongMemEval's own `flat-bm25` baseline, turn granularity.

    This is the benchmark's established reference retriever, so it is the arm
    that makes a number here mean something outside this repo. Implemented
    inline (Okapi BM25, k1=1.5, b=0.75) rather than pulled in as a dependency —
    it is twenty lines and the demo extra is already heavy.

    It is also the single most informative comparison for THIS library, because
    it is purely lexical. Retrieval here is cosine-only, and the failures
    measured on the built-in scenarios were repeatedly ones cosine cannot see:
    `$100k` against `$50k`, `13:50` against `13:20`, and a query asking what a
    number WAS losing to eight facts that merely mentioned it. If BM25 beats the
    fact graph on this subset, the missing signal is lexical rather than
    structural, and that is a different fix from any re-ranking.
    """
    import math

    docs, texts = [], []
    for session in _sessions_in_order(rec):
        for turn in session:
            c = (turn.get("content") or "").strip()
            if c:
                docs.append(f"{turn.get('role', '?')}: {c}")
                texts.append(_tok(c))
    if not docs:
        return "", {"input_tokens": 0, "output_tokens": 0, "context_chars": 0}, {}

    n = len(texts)
    avgdl = sum(len(t) for t in texts) / n
    df: dict[str, int] = {}
    for t in texts:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    k1, b = 1.5, 0.75
    q = _tok(rec["question"])
    scores = []
    for i, t in enumerate(texts):
        tf: dict[str, int] = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append((s, i))
    scores.sort(key=lambda p: (-p[0], p[1]))
    block = "\n".join(docs[i] for _s, i in scores[:k])
    pred, usage = ask(llm, rec["question"], block, rec.get("question_date", "unknown"))
    return pred, usage, {"docs": n, "retrieved": min(k, n)}


_BASELINE_STORE = None
# The context manager must outlive the store it yields. Dropping it lets the
# garbage collector run its __exit__, which closes the connection pool — the
# store object survives and every later call raises PoolClosed.
_BASELINE_CM = None


def _baseline_store():
    """LangGraph's own cross-thread memory primitive, opened once.

    Opened once rather than per question because `PostgresStore` builds a
    connection pool and `setup()` runs DDL; per-question construction is the
    same churn that was measured leaking ~24 pools per run in the scenario
    harness. Questions stay isolated by namespace tuple, which is how
    PostgresStore partitions anyway.
    """
    global _BASELINE_STORE, _BASELINE_CM
    if _BASELINE_STORE is None:
        from langchain_ollama import OllamaEmbeddings
        from langgraph.store.postgres import PostgresStore

        _BASELINE_CM = PostgresStore.from_conn_string(
            DSN,
            index={
                "dims": 768,
                "embed": OllamaEmbeddings(
                    model="nomic-embed-text", client_kwargs={"timeout": 120.0}
                ),
            },
            pool_config={"min_size": 1, "max_size": 4},
        )
        _BASELINE_STORE = _BASELINE_CM.__enter__()
        _BASELINE_STORE.setup()
    return _BASELINE_STORE


def run_baseline_arm(rec: dict, llm, k: int,
                     budget_tokens: int | None = None) -> tuple[str, dict, dict]:
    """Stock memory: whole turns stored as documents, retrieved by vector search.

    Ingestion is deliberately non-agentic, matching how the treatment is
    ingested here. LangMem's usual path has the agent decide what to save via
    `manage_memory`, but a transcript replay has no agent to decide — and giving
    one arm an LLM at write time while the other gets none would compare write
    budgets rather than memory architectures. So both arms see the same turns
    and differ in exactly what the README claims they differ in: this one keeps
    a turn whole, the treatment splits it into claims.

    `k` is larger than the treatment's by the same convention `retrieval_eval`
    uses (10 documents against 8 claims): forcing equal item counts would
    compare ten paragraphs against eight sentences. Characters are reported so
    the real cost difference stays visible.
    """
    store = _baseline_store()
    ns = ("memories", rec["question_id"])
    n = 0
    for si, session in enumerate(_sessions_in_order(rec)):
        for ti, turn in enumerate(session):
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            store.put(ns, f"s{si}t{ti}", {"content": f"{turn.get('role', '?')}: {content}"})
            n += 1
    # With a budget, take documents until it is exhausted rather than a fixed
    # count. A whole turn averages ~1.5k chars here, so `limit=10` and a 1,200
    # token budget are wildly different asks — comparing them would measure who
    # was handed more room rather than how the room was spent. Over-fetch, then
    # fill: documents are added whole, because truncating one mid-sentence
    # destroys exactly the co-occurrence that makes a document arm worth testing.
    limit = k if budget_tokens is None else max(k, 40)
    hits = store.search(ns, query=rec["question"], limit=limit)
    contents = [(h.value or {}).get("content", "") for h in hits]
    if budget_tokens is None:
        block = "\n".join(contents)
    else:
        budget_chars, kept, used = budget_tokens * 4, [], 0
        for c in contents:
            if used + len(c) > budget_chars:
                break
            kept.append(c)
            used += len(c) + 1
        block = "\n".join(kept)
    pred, usage = ask(llm, rec["question"], block, rec.get("question_date", "unknown"))
    return pred, usage, {"docs": n, "retrieved": len(hits)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--type", default="knowledge-update")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=8, help="treatment: claims injected")
    ap.add_argument("--kb", type=int, default=10, help="baseline: documents retrieved")
    ap.add_argument("--arms", nargs="+", default=["bm25", "baseline", "memory", "fullcontext"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip question_ids already complete for every requested arm in --out")
    # MEASURED, and it invalidates any run made without it. Ollama's default
    # num_ctx is 2048, and `demo.llm.make_llm` does not set one — so a prompt of
    # 5k, 22k or 135k tokens all reported prompt_eval_count=2051, because the
    # prompt was truncated to the window before evaluation. Setting num_ctx=8192
    # on the identical prompt reported 7025.
    #
    # It matters most for `fullcontext`, whose whole point is that the entire
    # haystack is in the prompt: truncated, it is not a ceiling arm at all, it is
    # a differently-truncated memory arm, and the comparison would flatter this
    # library by crippling the thing it is measured against.
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--budget-all", action="store_true",
                    help="apply --budget to the bm25/baseline arms too, not just tail/augment")
    ap.add_argument("--budget", type=int, default=1200,
                    help="tokens of context the augment/tail arms may use, total")
    ap.add_argument("--ingest", choices=["replay", "agent", "graph", "autolink"], default="replay",
                    help="agent: one reconciliation call per session so supersedes is reachable")
    args = ap.parse_args()

    # ResilientChatOllama, not a bare ChatOllama, and the reason is measured:
    # a bare client with no timeout hung this harness for 1h49m mid-question,
    # blocked in sock_recv on a request Ollama accepted and never answered while
    # /api/version and /api/ps both still returned 200. That is the wedge
    # documented in results/README.md, and demo.llm already solves it — probing
    # /api/generate, restarting the daemon only when genuinely wedged, and
    # rebuilding the client so the retry cannot land on a dead pooled socket.
    # It is a ChatOllama subclass, so num_ctx passes straight through.
    from demo.llm import MODEL, ResilientChatOllama
    from demo.resilient import CHAT_TIMEOUT_S

    llm = ResilientChatOllama(
        model=MODEL,
        temperature=0.0,
        reasoning=False,
        num_ctx=args.num_ctx,
        client_kwargs={"timeout": CHAT_TIMEOUT_S},
    )
    records = load(args.data, args.type, args.limit)
    print(f"{len(records)} records | type={args.type} | arms={args.arms} | k={args.k}\n")

    # Resume, and write after every question. A full 4-arm pass is ~8 model
    # calls per question over 78 questions, and Ollama wedges under exactly that
    # kind of sustained load — one wedge previously destroyed 21 questions of
    # completed work because results were only written at the end.
    rows = []
    done: set[str] = set()
    if args.out and args.resume and pathlib.Path(args.out).exists():
        rows = json.loads(pathlib.Path(args.out).read_text())
        done = {r["question_id"] for r in rows if set(args.arms) <= set(r.get("arms", {}))}
        print(f"resuming: {len(done)} questions already complete\n")

    def _flush() -> None:
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))

    for i, rec in enumerate(records, 1):
        if rec["question_id"] in done:
            continue
        row = {"question_id": rec["question_id"], "question": rec["question"],
               "gold": rec["answer"], "arms": {}}
        for arm in args.arms:
            t0 = time.time()
            if arm == "fullcontext":
                pred, usage, extra = run_fullcontext_arm(rec, llm)
            elif arm == "augment":
                pred, usage, extra = run_augment_arm(rec, llm, args.k, args.budget, args.ingest)
            elif arm == "tail":
                pred, usage, extra = run_tail_arm(rec, llm, args.budget)
            elif arm == "bm25":
                pred, usage, extra = run_bm25_arm(rec, llm, args.kb)
            elif arm == "baseline":
                pred, usage, extra = run_baseline_arm(rec, llm, args.kb,
                                                     args.budget if args.budget_all else None)
            else:
                pred, usage, extra = run_memory_arm(rec, llm, args.k, args.ingest)
            ok = judge(llm, rec["question"], rec["answer"], pred)
            row["arms"][arm] = {"pred": pred, "correct": ok, "seconds": round(time.time() - t0, 1),
                                **usage, **extra}
            print(f"[{i}/{len(records)}] {arm:12s} {'PASS' if ok else 'FAIL'} "
                  f"tok={usage['input_tokens']:6d} ctx={usage['context_chars']:6d}  {pred[:70]!r}")
        rows.append(row)
        _flush()

    print(f"\n{'=' * 78}\n## {args.type}  n={len(rows)}\n")
    print("| arm | accuracy | mean input tokens | mean context chars |")
    print("|---|---|---|---|")
    for arm in args.arms:
        got = [r["arms"][arm] for r in rows if arm in r.get("arms", {})]
        acc = sum(a["correct"] for a in got) / len(got)
        tok = sum(a["input_tokens"] for a in got) / len(got)
        ctx = sum(a["context_chars"] for a in got) / len(got)
        print(f"| {arm} | {acc:.1%} ({sum(a['correct'] for a in got)}/{len(got)}) "
              f"| {tok:.0f} | {ctx:.0f} |")
    if args.ingest in ("agent", "graph", "autolink"):
        print(f"\ningest stats: {_STATS}")
    else:
        print("\nNOTE: replay ingestion emits no `supersedes` edges — see module docstring.")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")

    # Close the pool explicitly. Left to the garbage collector it printed
    # "couldn't stop thread 'pool-1-worker-N' within 5.0 seconds" at exit, and a
    # non-daemon worker that will not stop is the difference between a noisy
    # exit and a run that never returns.
    if _BASELINE_CM is not None:
        import contextlib

        with contextlib.suppress(Exception):  # teardown must not mask the results
            _BASELINE_CM.__exit__(None, None, None)


if __name__ == "__main__":
    main()
