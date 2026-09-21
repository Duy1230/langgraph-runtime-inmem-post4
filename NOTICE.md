# Modification notice

This repository is a modified distribution of `langgraph-runtime-inmem`, whose
package metadata identifies LangChain as the author and Elastic-2.0 as the
license.

The local baseline came from
`https://github.com/tapati0127/langgraph-runtime-inmem.git` at commit
`259d5b4f12b2e187eebbf51ce9606e2f17164720`, followed by the post4 work and the
review fixes dated 2026-09-19. The resulting package version is
`0.33.3.post4+review.20260919`.

The modifications add and repair TTL cleanup, local persistence lifecycle,
stream cleanup, PostgreSQL Store/checkpointer hooks, PostgreSQL pruning and run
deletion, bounded reconnecting pools, fractional Store TTL, numeric filters,
and concurrent deletion behavior. See `README.md` for the known durability
limitation: Agent Server thread/run/assistant metadata is still in-memory and
can be lost when the process is killed immediately after acknowledging work.

This is not an official LangChain release. The Docker image in this repository
is only a source-transfer archive and does not expose the software as a hosted
or managed service.
