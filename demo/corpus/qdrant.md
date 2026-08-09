# Qdrant Evaluation Notes

Qdrant is a dedicated vector database written in Rust.

## Deployment
Self-hostable (Apache 2.0) as a single binary or via Docker/K8s. Managed
cloud also available. Self-hosted cluster for our workload (10M vectors,
768-d, HA pair): approximately $30,000/year including the extra on-call
burden we priced at half an SRE-week per quarter.

## Performance
p95 ~12ms at 10M vectors with 97% recall. Strong filtered-search
performance with payload indexes.

## Memory optimization
Binary quantization reduces RAM usage up to 32x with a modest recall hit
(recovered via oversampling + rescoring). In our test it cut a 24GB
deployment to under 2GB of hot RAM. This was the single largest memory
saving we measured across all candidates.

## Operational notes
New service to run: monitoring, upgrades, snapshots are all new surface.
Snapshot/restore tooling is solid. Team has no prior Rust-service
operational experience, though none is strictly required.
