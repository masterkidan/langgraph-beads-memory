# Competitive Brief: Agent Memory Layer for LangGraph (vs. langgraph-beads-memory)

*Research date: 2026-07-31. This space moves fast — treat pricing/funding figures as directional.*

Related design spec: [2026-07-31-beads-memory-design.md](2026-07-31-beads-memory-design.md)

## Competitive Landscape

**Direct competitors** — memory layers usable with LangGraph agents today:
- **LangGraph-native** (`BaseStore`/`PostgresStore` + LangMem) — the first-party incumbent, i.e. what you get by doing nothing extra
- **Mem0** — the well-funded, fast-retrieval SaaS-first memory API
- **Zep / Graphiti** — temporal knowledge graph, framework-agnostic
- **Cognee** — open-source, single-Postgres memory platform with an explicit LangGraph interface
- **Letta (MemGPT)** — OS-style tiered agent memory, native multi-agent

**Adjacent/inspiration**: **beads** itself (steveyegge/beads) — no LangGraph integration exists; it's a CLI issue tracker for coding agents, Dolt-backed, repo-scoped.

**Indirect/substitute**: teams hand-rolling their own `pgvector` table and calling it a day (very common, not a product, but real competition for adoption).

## Competitor Overview

| | LangGraph-native | Mem0 | Zep / Graphiti | Cognee | Letta | beads |
|---|---|---|---|---|---|---|
| **Backing** | LangChain Inc (first-party) | Series A, ~$24-25.5M raised, YC/Basis Set/Peak XV | Seed, ~$0.5-2.3M raised, 8 employees | Seed, $7.5M raised | Platform co. (ex-MemGPT/Berkeley) | Solo/small OSS (Steve Yegge), Oct 2025 |
| **License/model** | OSS (MIT), self-hosted | OSS core + metered SaaS ($19-249+/mo) | OSS (Graphiti) + hosted Zep platform | Apache 2.0, full features, no caps | OSS core + hosted platform | OSS, free |
| **Storage** | Postgres | Managed vector store; graph mode needs Neo4j/Memgraph | Neo4j (or FalkorDB) | Postgres only (graph+vector+metadata unified) | Postgres + pgvector | Dolt (git-like SQL) |

## Feature Comparison

Rated against the specific angle **langgraph-beads-memory** stakes out: LangGraph-native middleware, Postgres-only, typed fact/edge graph, explicit dual-path capture, and enforced sub-agent fork+rollup.

| Capability | LangGraph-native | Mem0 | Zep/Graphiti | Cognee | Letta | beads | **langgraph-beads-memory (proposed)** |
|---|---|---|---|---|---|---|---|
| LangGraph-native integration | Strong | Adequate | Weak | Adequate | Weak | Absent | **Strong** |
| Postgres-only (no extra infra) | Strong | Weak (graph mode needs Neo4j, Pro-tier only) | Absent (Neo4j required) | Strong | Adequate | Absent (Dolt) | **Strong** |
| Typed fact relationship graph (supersedes/contradicts/etc.) | Absent | Weak (OSS extraction is ADD-only as of SDK v2; graph relations gated at $249/mo) | **Strong** (bi-temporal invalidation is their whole architecture) | Adequate | Absent | Strong (direct inspiration) | **Strong** |
| Explicit, deliberate capture (vs. blind auto-extraction) | Adequate | Weak (auto-extraction is the core pitch) | Weak (fully automatic ingestion) | Weak | Adequate (agent manages its own memory blocks) | Strong (`bd remember`) | **Strong** |
| Sub-agent fork + enforced rollup | Absent | Absent | Absent | Absent | Adequate (native multi-agent, but agent-scoped, not fork+forced-summary) | Adequate (Dolt branching, but repo-scoped not conversation-scoped) | **Strong** |
| Actor/delegation provenance | Weak | Weak/Adequate (user_id/agent_id/run_id exist) | Adequate (episode→fact provenance) | Weak | Weak | Weak (assignee field only) | **Strong** |

**The gap is real and specific.** Every competitor does *some* piece of this — Zep/Graphiti nails the typed temporal fact graph, Mem0 nails low-latency retrieval, Cognee nails Postgres-only simplicity + a LangGraph interface, Letta nails deliberate agent-driven memory management and native sub-agent calling — but **none of them combine all of it, and specifically none of them have a first-class, enforced "fork memory on sub-agent spawn, must roll up a summary before returning" primitive.** That's the one cell where every competitor is Absent or only partially adequate.

## Positioning Analysis

- **LangGraph-native**: "It's already in the box." Category: infrastructure primitive, not a product. Differentiator: zero extra dependency. Vulnerable on: LangMem's own benchmarked p95 search latency (~60s in cited comparisons) is a real weakness if accurate for typical configs — worth your own benchmark before citing externally.
- **Mem0**: "The fast, framework-agnostic memory API." Differentiator: sub-200ms retrieval, 21 framework integrations. Vulnerable on: their more structured/graph capability has visibly regressed (OSS graph drivers removed in v2, contradiction handling weak in base config) and is paywalled — they're optimizing for breadth and speed, not depth or correctness of the fact graph.
- **Zep/Graphiti**: "Temporal knowledge graph, memory that doesn't go stale." Differentiator: bi-temporal fact invalidation, source provenance. Vulnerable on: requires Neo4j (real infra cost/complexity for a team that's Postgres-only elsewhere), tiny team, unclear LangGraph-specific ergonomics.
- **Cognee**: "One Postgres instance, full memory stack." Differentiator: infra simplicity, Apache 2.0, explicit LangGraph interface. This is the **closest competitor** to beads-memory's infra story. Vulnerable on: generic `remember/recall/forget/improve` API doesn't give you beads' specific semantics (typed edges, explicit dual capture, forced sub-agent rollup) — it's a memory platform, not opinionated about *how* an agent decides what's worth keeping.
- **Letta**: "Stateful agents with OS-style memory." Differentiator: deliberate agent-managed memory blocks, native multi-agent. Vulnerable on: it's a full agent runtime you adopt, not a lightweight adapter you drop into an existing LangGraph app.
- **beads**: "Memory upgrade for your coding agent." Not a competitor in the market sense (different runtime, no LangGraph story) — it's the pattern language beads-memory borrows, not something you'd choose instead of it.

## Strengths & Weaknesses Summary

| | Strengths | Weaknesses |
|---|---|---|
| LangGraph-native | Zero dependency, official, free | No fact graph, latency concerns, generic KV shape |
| Mem0 | Fastest retrieval, widest framework coverage, well-funded/momentum | Weak fact/graph correctness in OSS tier, structure paywalled |
| Zep/Graphiti | Best-in-class temporal fact graph, strong provenance | Neo4j dependency, tiny team, not LangGraph-specific |
| Cognee | Postgres-only simplicity, generous OSS license, LangGraph interface exists | Generic memory API, no typed-edge or fork/rollup semantics |
| Letta | Deliberate agent-driven memory, native multi-agent | Full runtime adoption, not an adapter; no fork/rollup pattern |
| beads | The actual pattern (typed edges, explicit remember, compaction) proven in production for coding agents | No LangGraph integration, no conversational/sub-agent-conversation model, Dolt not Postgres |

## Opportunities

- **The sub-agent fork+rollup gap is uncontested.** No competitor treats "spawn a sub-agent → isolated memory → forced summary back to parent" as a first-class primitive. This is beads-memory's sharpest, most defensible differentiator — lead with it.
- **"Postgres-only + typed fact graph" is a narrow but real niche.** Cognee owns Postgres-only; Zep/Graphiti owns the typed graph; nobody owns both at once with LangGraph-specific ergonomics.
- **Explicit capture as a stance, not just a mechanism.** Mem0 and Zep are both betting on automatic extraction as the differentiator (speed, no agent cooperation needed). There's a credible counter-position — "auto-extraction hallucinates and over-collects; deliberate capture is more auditable and cheaper" — that beads-memory can own if you want to publish it, especially paired with beads' own track record of that stance working for coding agents.
- **LangMem's cited latency weakness** is a specific, checkable wedge against the "just use what's built in" objection, if your own benchmarks confirm it.

## Threats

- **Cognee is the nightmare-scenario mover.** It's well-funded, Postgres-only already, has a LangGraph interface, and Apache 2.0. The most likely competitive response if beads-memory gets attention is Cognee (or a fork of it) adding beads-style typed edges and a sub-agent fork primitive — they have the infra and the funding to do it fast.
- **LangChain could ship this themselves.** LangMem is actively developed by the LangGraph team; if they decide typed fact graphs + sub-agent memory scoping matter, first-party wins by default distribution alone, regardless of design quality.
- **Mem0's distribution advantage.** 21 framework integrations and a large user base mean even a less-precise memory model can win on convenience and momentum.

## Strategic Implications

- **For idea validation**: this is a real, uncrowded gap — proceed. The core differentiators (explicit dual-path capture, typed fact graph, *enforced* sub-agent rollup, Postgres-only) are not fully covered by any single existing tool. Cognee is the one to watch most closely as a comparison point during implementation, not a reason to stop.
- **For open-sourcing/publishing**: position narrowly and specifically — "beads-style memory for LangGraph sub-agents," not "another agent memory framework." Don't compete on retrieval latency benchmarks against Mem0 (you'll likely lose that specific fight); compete on **correctness/auditability of the fork+rollup model** and **infra simplicity** (Postgres-only, no Neo4j). Explicitly cite beads as the design lineage — it's a credibility signal, not something to obscure.
- **Monitor**: Cognee's roadmap (closest competitor), whether LangMem adds graph-typed memory, and Mem0's OSS graph-memory feature (they've moved on this before — v1→v2 removed the Neo4j driver, they could move again).
