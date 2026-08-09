# pgvector Evaluation Notes

pgvector is a Postgres extension providing vector similarity search.

## Deployment
Self-hosting is trivial if you already run Postgres: install the extension,
no separate service. Fully open source (PostgreSQL license). Annual
infrastructure cost for our workload (10M vectors, 768-d): approximately
$18,000/year on managed Postgres, or $12,000/year self-hosted on EC2.

## Performance
HNSW index: ~40ms p95 at 10M vectors with 95% recall in our load test.
Write throughput degrades ~20% while the HNSW index builds.

## Operational notes
Backups ride the existing Postgres backup pipeline. No new on-call surface.
Index rebuild after bulk load takes ~50 minutes at 10M vectors.
Team already knows Postgres; zero new operational skills required.

## Limits
Single-node vertical scaling only, without Citus. Metadata filtering is
just SQL WHERE clauses (a strength). No built-in quantization in the
version we tested; memory footprint is ~6GB for 10M 768-d vectors.
