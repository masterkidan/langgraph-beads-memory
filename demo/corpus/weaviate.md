# Weaviate Evaluation Notes

Weaviate is an open-source vector database with a GraphQL-flavored API.

## Deployment
Self-hostable (BSD-3) via Docker/K8s; managed cloud available. Self-hosted
HA deployment for our workload: approximately $60,000/year — the K8s
operator effectively requires a small dedicated cluster, and we'd need the
commercial tier for the module ecosystem we want.

## Performance
p95 ~15ms at 10M vectors, 96% recall. Built-in hybrid (BM25+vector) search
is the standout feature; it noticeably improved our relevance on short
queries.

## Operational notes
Heaviest operational footprint of the three candidates. Module system
(rerankers, vectorizers) is powerful but adds upgrade complexity. GraphQL
API is a new paradigm for the team.
