"""Score retrieval quality per arm, offline, with no LLM in the loop.

The end-to-end harness measures retrieval and generation together. This
measures retrieval alone, against a frozen fixture (see
`demo.retrieval_fixture`), so a ranking change can be evaluated in under a
second instead of a 30-minute run — and so a regression can be attributed to
ranking rather than to sampling variance.

    uv run python -m demo.retrieval_eval results/fixtures/incident-run1.json
    uv run python -m demo.retrieval_eval FIXTURE --query-mode distilled

What is scored, and why it is aspect coverage rather than plain recall.
Each query carries the ASPECTS its answer must contain, taken from the same
`PLANTED` ground truth the end-to-end metrics use — so a retrieval score here
predicts an end-to-end metric there rather than correlating with it loosely.
`names_surviving_cause` fails end-to-end exactly when the `surviving_cause`
aspect is missing from the injected block, and that is the quantity reported.

Plain recall@k would be the wrong summary. The breadth question needs three
DIFFERENT subsystems; retrieving five facts about the database and none about
the network scores well on recall and answers nothing. Coverage counts distinct
aspects, so redundancy earns nothing — which is the actual failure mode
observed: six of eight injected slots at conv-3 were the same constraint
restated.

The two arms are NOT scored on equal item counts, deliberately. The baseline
returns whole documents and the treatment returns per-claim facts; forcing both
to k=8 would compare eight paragraphs against eight sentences. Both are scored
at the k each actually used in the measured run (baseline 10, treatment 8), and
`--k` overrides for sweeps.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

from demo.scenario_incident import PLANTED

# Production defaults, mirrored from src/beads_memory/store.py. Duplicated as
# plain values rather than imported so a sweep can vary them without mutating
# module state that the production path also reads.
DESCENDANT_RANK_PENALTY = 0.15
DESCENDANT_MAX_PER_AGENT = 2
OVERFETCH = 4

_ELIM = [m.lower() for m in PLANTED["elimination_markers"]]


def _covers(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(t.lower() in low for t in terms)


def _subsystem_aspect(name: str) -> tuple[str, list[str]]:
    return name, PLANTED["subsystem_variants"][name]


# One entry per scored turn of the incident scenario. `text` is the user message
# VERBATIM, because that is what the treatment actually embeds — see
# middleware.wrap_model_call, which takes the query from the last HumanMessage
# with no rewriting. Keeping it verbatim is the point: any improvement from
# rewriting it must be demonstrated, not assumed.
INCIDENT_QUERIES: list[dict] = [
    {
        "id": "conv-3/next_steps",
        "text": (
            "New shift taking over. Given everything we've established, what should we try "
            "next, and why? Be specific about what we already ruled out so I don't repeat work."
        ),
        # Distilled form: framing stripped, task nouns kept. This is the
        # comparison the conv-3 failure motivates — the baseline's agent WROTE
        # itself a query like this ("incident status investigation progress
        # ruled out facts") and retrieved the root cause; the treatment
        # embedded the framing and retrieved constraints.
        "distilled": "root cause next step mitigation what was ruled out",
        "aspects": [
            ("surviving_cause", PLANTED["surviving_cause_variants"]),
            ("reversible_fix", PLANTED["reversible_fix_variants"]),
        ],
        "predicts": "names_surviving_cause, proposes_reversible_fix",
    },
    {
        "id": "conv-4/buried",
        "text": (
            "One more thing for the postmortem — what exactly did the network investigation "
            "measure for TLS handshake p99?"
        ),
        "distilled": "TLS handshake p99 measurement network",
        "aspects": [("tls_p99", PLANTED["buried_metric_variants"][0])],
        "predicts": "buried_metric_recalled",
    },
    {
        "id": "conv-4/breadth",
        "text": ("And list everything we investigated and ruled out, with the "
                 "measurement behind each one."),
        "distilled": "database network application tier investigated ruled out measurements",
        # NOT elimination-gated, and the first version of this entry was — which
        # was a bug in the metric, not a finding about the library. Gating on
        # `elimination_markers` requires each subsystem to appear beside "ruled
        # out"/"was healthy"/etc. That is true of db and network and FALSE of
        # apptier, which is the surviving root cause: nothing eliminated it. The
        # gate therefore made the apptier aspect unreachable by construction and
        # scored a correct store as a retrieval failure. Its only "qualifying"
        # apptier item was a DNS fact matching on the phrase "checkout service".
        #
        # The production metric (`breadth_subsystems_named` in
        # metrics_incident.py) counts subsystems NAMED in the answer, with no
        # such gate. This mirrors it, because a retrieval score is only useful
        # here if it predicts the end-to-end metric it stands in for.
        "aspects": [_subsystem_aspect(s) for s in ("db", "network", "apptier")],
        "predicts": "breadth_complete",
    },
    {
        "id": "conv-4/timeline",
        "text": ("Last thing for the postmortem header — what time did release 2.14 "
                 "actually go out?"),
        "distilled": "release 2.14 deploy time",
        "aspects": [("corrected_deploy", PLANTED["corrected_deploy_time_variants"])],
        "predicts": "uses_corrected_deploy_time",
    },
]


def _vecdb_queries() -> list[dict]:
    """The two scored turns of the vecdb scenario, gold taken from its own
    PLANTED block so these predict `metrics.constraint_carry` the same way the
    incident set predicts `metrics_incident.incident_carry`."""
    from demo.scenario import PLANTED as V

    return [
        {
            "id": "conv-3/pick",
            "text": ("Given everything we've established, which vector database should we "
                     "pick and why? Be specific about how it fits our constraints."),
            "distilled": "vector database recommendation budget self-hostable benchmark",
            "aspects": [
                # The REVISED budget, not the stale one. Retrieval has to
                # surface $50k; surfacing $100k is the failure this library's
                # supersede edge exists to prevent.
                ("revised_budget", V["revised_budget_variants"]),
                ("selfhost", V["selfhost_variants"]),
                ("primary_sources", [V["constraint_primary_sources"]]),
                ("feasible_option", V["expected_pick_one_of"]),
            ],
            "predicts": "uses_revised_budget, mentions_selfhost, mentions_feasible_option",
        },
        {
            "id": "conv-3/buried",
            "text": ("And remind me — what was that big memory optimization the Qdrant "
                     "researcher found, and roughly how much did it save?"),
            "distilled": "Qdrant memory optimization savings quantization",
            # Both halves are required end-to-end (`all(...)` in
            # constraint_carry), so they are separate aspects here: naming the
            # technique without its magnitude is the exact failure the metric
            # was written to catch.
            "aspects": [
                ("quantization", V["buried_detail_variants"][0]),
                ("magnitude_32x", V["buried_detail_variants"][1]),
            ],
            "predicts": "buried_detail_recalled",
        },
    ]


# Kept as a module-level name because the sweep and the scratch analyses import
# it directly; `for_scenario` is what new code should call.
QUERIES = INCIDENT_QUERIES


def for_scenario(name: str) -> list[dict]:
    return INCIDENT_QUERIES if name == "incident" else _vecdb_queries()


# --- vector math (pure python; 768-d over ~100 items is nothing) ------------
def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 1.0
    return 1.0 - dot / (na * nb)


# --- rankers ---------------------------------------------------------------
def rank_baseline(fixture: dict, qvec: list[float], k: int) -> list[dict]:
    """Flat cosine over whole documents — what LangMem's `search_memory` does."""
    scored = [
        {"body": d["content"], "distance": cosine_distance(qvec, d["embedding"]), "tag": "doc"}
        for d in fixture["baseline"]["documents"]
        if d["embedding"] and d["content"]
    ]
    scored.sort(key=lambda r: r["distance"])
    return scored[:k]


def _redundancy(a: str, b: str) -> float:
    """Token-overlap similarity, not cosine.

    Deliberately lexical. The whole problem being fixed is that cosine cannot
    separate these facts — six near-identical restatements of one constraint sat
    within 0.065 of each other — so using cosine to detect the duplication would
    inherit exactly the blindness it is meant to correct. Jaccard over lowercased
    word sets catches "All actions must be reversible within 30 minutes" against
    "Constraint: Actions must be reversible within 30 minutes." outright.
    """
    ta = {w for w in a.lower().split() if len(w) > 3}
    tb = {w for w in b.lower().split() if len(w) > 3}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rank_treatment(fixture: dict, qvec: list[float], k: int, *, penalty: float | None = None,
                   cap: int | None = None, drop_directives: bool = True,
                   exempt_summaries: bool = False, drop_echo: bool = False,
                   mmr: float | None = None, claim_weight: float = 1.0,
                   max_per_source: int = 0) -> list[dict]:
    """Reproduce `BeadsStore.search` for the ROOT namespace.

    Scope, filters, penalty and the per-agent cap are all mirrored from
    production. The one thing NOT modelled is the middleware's dedup of facts
    derived from still-visible user messages (`excluded=2` in the measured
    run): it depends on the message window at that turn, not on the store, so
    it belongs to the end-to-end harness. Its effect is to free up to two slots,
    which makes this evaluation very slightly pessimistic for the treatment.
    """
    penalty = DESCENDANT_RANK_PENALTY if penalty is None else penalty
    cap = DESCENDANT_MAX_PER_AGENT if cap is None else cap

    ns = fixture["treatment"]["namespaces"]
    root = next((n for n in ns if n["parent_id"] is None), None)
    if root is None:
        raise SystemExit("fixture has no root namespace")
    descendants = {n["id"] for n in ns if n["parent_id"] is not None}
    scope = {root["id"]} | descendants

    rows = []
    for f in fixture["treatment"]["facts"]:
        if f["namespace_id"] not in scope:
            continue
        if f["status"] != "active" or not f["embedding"]:
            continue
        if drop_directives and f["kind"] == "directive":
            continue
        # The agent quoting its own prose back into the store: 38 facts and 41%
        # of the bytes in the measured run, almost all of it restating
        # constraints the user had already stated. It is what fills the top of
        # the ranking with duplicates.
        if drop_echo and f["kind"] == "conclusion" and f["source"] == "passive_capture":
            continue
        demoted = f["namespace_id"] in descendants
        # A `conclude_task` summary is a sub-agent's CURATED output, not its raw
        # exploration — it is the one artefact the fork/rollup design exists to
        # produce. Penalising it as "child chatter" inverts the intent: measured,
        # the two best-matching root-cause facts by raw distance were rollups,
        # and +0.15 moved them from ~rank 40 to rank 69 and 73.
        exempt = exempt_summaries and f["source"] in ("conclude_task", "fallback_conclude")
        d = cosine_distance(qvec, f["embedding"])
        # Context blend, mirroring store.search. A fact with no context vector
        # falls back to its own distance — the SQL does this with COALESCE, so a
        # fixture captured before the column existed scores exactly as it did.
        ctx = f.get("context_embedding")
        dctx = cosine_distance(qvec, ctx) if ctx else d
        blended = claim_weight * d + (1 - claim_weight) * dctx
        rows.append(
            {
                "body": f["body"],
                "distance": d,
                # Identity of the capture this claim came from, for the
                # per-source cap. A prefix of the vector is enough to group by
                # and keeps the key hashable.
                "ctx_key": tuple(ctx[:8]) if ctx else None,
                "sort": blended + (penalty if (demoted and not exempt) else 0.0),
                "demoted": demoted,
                "source": f["source"],
                "agent_id": f["agent_id"],
                "tag": f"{f['kind']}/{f['source']}",
            }
        )
    rows.sort(key=lambda r: r["sort"])
    rows = rows[: k * OVERFETCH]

    kept, per_agent, per_source = [], {}, {}
    for r in rows:
        # Per-source ceiling, mirroring store._cap_per_descendant_agent. Blended
        # ranking lifts a whole capture's claims together by design; without a
        # ceiling one verbose answer takes every slot.
        if max_per_source and r["ctx_key"] is not None:
            seen_src = per_source.get(r["ctx_key"], 0)
            if seen_src >= max_per_source:
                continue
            per_source[r["ctx_key"]] = seen_src + 1
        if r["demoted"] or r["source"] in ("conclude_task", "fallback_conclude"):
            seen = per_agent.get(r["agent_id"], 0)
            if seen >= cap:
                continue
            per_agent[r["agent_id"]] = seen + 1
        # Maximal-marginal-relevance gate. A candidate that mostly repeats
        # something already selected does not get a slot, however close it sits
        # to the query. This is the direct answer to six of eight slots holding
        # one constraint: relevance alone cannot see redundancy.
        if mmr is not None and any(_redundancy(r["body"], s["body"]) > mmr for s in kept):
            continue
        kept.append(r)
        if len(kept) == k:
            break
    return kept


# --- scoring ---------------------------------------------------------------
def score(results: list[dict], query: dict) -> dict:
    gate = query.get("gate")
    covered, first_rank = {}, None
    for rank, r in enumerate(results, 1):
        body = r["body"]
        if gate and not _covers(body, gate):
            continue
        for name, terms in query["aspects"]:
            if name in covered:
                continue
            if _covers(body, terms):
                covered[name] = rank
                if first_rank is None or rank < first_rank:
                    first_rank = rank
    n = len(query["aspects"])
    return {
        "covered": len(covered),
        "aspects": n,
        "coverage": len(covered) / n,
        "detail": covered,
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
    }


def evaluate(fixture: dict, embed, *, query_mode: str, kb: int, kt: int, **kw) -> list[dict]:
    out = []
    for q in QUERIES:
        text = q["text"] if query_mode == "verbatim" else q["distilled"]
        qvec = embed(text)
        b = score(rank_baseline(fixture, qvec, kb), q)
        # The baseline's query is agent-authored in the real run, so verbatim
        # mode understates it. Reported anyway; the honest comparison is
        # distilled-vs-distilled.
        t = score(rank_treatment(fixture, qvec, kt, **kw), q)
        out.append({"query": q["id"], "predicts": q["predicts"], "baseline": b, "treatment": t})
    return out


def _print(rows: list[dict], label: str) -> None:
    print(f"\n### {label}\n")
    print("| query | predicts | baseline cov | treatment cov | b MRR | t MRR |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        b, t = r["baseline"], r["treatment"]
        print(
            f"| {r['query']} | {r['predicts']} | {b['covered']}/{b['aspects']} "
            f"| {t['covered']}/{t['aspects']} | {b['mrr']:.2f} | {t['mrr']:.2f} |"
        )
    bc = sum(r["baseline"]["coverage"] for r in rows) / len(rows)
    tc = sum(r["treatment"]["coverage"] for r in rows) / len(rows)
    print(f"\nmean aspect coverage — baseline {bc:.0%}, treatment {tc:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("--query-mode", choices=["verbatim", "distilled", "both"], default="both")
    ap.add_argument("--kb", type=int, default=10, help="baseline k (docs returned in the real run)")
    ap.add_argument("--kt", type=int, default=8, help="treatment k (injected slots)")
    ap.add_argument("--penalty", type=float, default=None)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--keep-directives", action="store_true")
    ap.add_argument("--exempt-summaries", action="store_true")
    ap.add_argument("--drop-echo", action="store_true")
    ap.add_argument("--mmr", type=float, default=None, help="redundancy ceiling, e.g. 0.5")
    ap.add_argument("--sweep", action="store_true", help="compare stacked configurations")
    args = ap.parse_args()

    fixture = json.loads(pathlib.Path(args.fixture).read_text())

    from beads_memory.embeddings import OllamaEmbedder

    emb = OllamaEmbedder()
    cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in cache:
            cache[text] = emb.embed(text)
        return cache[text]

    kw = {
        "penalty": args.penalty,
        "cap": args.cap,
        "drop_directives": not args.keep_directives,
        "exempt_summaries": args.exempt_summaries,
        "drop_echo": args.drop_echo,
        "mmr": args.mmr,
    }

    if args.sweep:
        # Stacked, not factorial: each row adds one change to the row above, so
        # the table reads as "what did this change buy on top of the last one?"
        # A full factorial over four knobs would be 16 rows and would still not
        # separate them, because they interact — dropping echo removes most of
        # what MMR would have suppressed.
        configs = [
            ("production (as measured)", {}),
            ("+ exempt rollups from penalty", {"exempt_summaries": True}),
            ("+ drop self-echo", {"exempt_summaries": True, "drop_echo": True}),
            ("+ MMR 0.5", {"exempt_summaries": True, "drop_echo": True, "mmr": 0.5}),
            ("MMR 0.5 alone", {"mmr": 0.5}),
            ("drop self-echo alone", {"drop_echo": True}),
        ]
        for mode in ("verbatim", "distilled"):
            print(f"\n{'=' * 78}\n## query = {mode}\n")
            print("| config | conv-3 | buried | breadth | timeline | mean cov |")
            print("|---|---|---|---|---|---|")
            base = {"penalty": args.penalty, "cap": args.cap,
                    "drop_directives": not args.keep_directives}
            for label, over in configs:
                rows = evaluate(fixture, embed, query_mode=mode, kb=args.kb, kt=args.kt,
                                **{**base, **{"exempt_summaries": False, "drop_echo": False,
                                              "mmr": None}, **over})
                cells = [f"{r['treatment']['covered']}/{r['treatment']['aspects']}" for r in rows]
                mean = sum(r["treatment"]["coverage"] for r in rows) / len(rows)
                print(f"| {label} | {' | '.join(cells)} | {mean:.0%} |")
            b = evaluate(fixture, embed, query_mode=mode, kb=args.kb, kt=args.kt, **base)
            cells = [f"{r['baseline']['covered']}/{r['baseline']['aspects']}" for r in b]
            bm = sum(r["baseline"]["coverage"] for r in b) / len(b)
            print(f"| **baseline (reference)** | {' | '.join(cells)} | **{bm:.0%}** |")
        return

    modes = ["verbatim", "distilled"] if args.query_mode == "both" else [args.query_mode]
    for mode in modes:
        rows = evaluate(fixture, embed, query_mode=mode, kb=args.kb, kt=args.kt, **kw)
        _print(rows, f"query={mode}  kb={args.kb} kt={args.kt}")


if __name__ == "__main__":
    main()
