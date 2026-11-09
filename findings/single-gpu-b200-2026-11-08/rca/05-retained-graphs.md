# Memory growth from retaining computation graphs

## Trigger and evidence

The unhealthy drill uses batch 8, stores each graph-bearing loss, and calls
backward with `retain_graph=True` for 12 updates. The healthy control does not
keep the graphs. Both use the same model and data setup.

Healthy live allocation stays near 0.247 GiB after updates. The retained-graph
case grows from approximately 0.512 GiB after update 1 to 2.886 GiB after update
10 and 3.413 GiB after update 12. Peak allocation reaches approximately 3.94 GiB.
The saved allocator snapshot preserves live allocation history.

These are live-tensor measurements, not merely a caching allocator's reserved
capacity. Growth across updates is the key signal; a large stable reserved
pool alone would not prove a leak.

## Cause and correction

Holding graph-bearing losses while retaining their graphs preserves tensors
that backward would normally release. Log detached scalar values or summaries
and allow each microbatch graph to be freed after backward. Gradient accumulation
requires retaining accumulated gradients, not keeping all the activation graphs.

The lab's first cleanup also retained one final graph through a materialized
`locals()` dictionary. Removing that introspection reference and explicitly
dropping the losses returned live allocation from 3,665,093,632 bytes to
265,716,736 bytes (approximately 0.247 GiB). That was a cleanup issue in the
diagnostic harness; the ordinary trainer was not using that cleanup path.

The corrected drill is retained alongside the first one so this evidence is
not overwritten. Calling `empty_cache()` alone would not release live graphs;
their references first have to be removed. The repeated run demonstrates release
after removing those references, rather than relying only on process exit.

## Artifacts

Under `runs/single-gpu-b200-2026-11-08/`:

- `detached-control/`: healthy control and allocator snapshot.
- `retained-graphs/`: original growth and initial cleanup measurements.
- `retained-graphs-cleanup-fixed/`: corrected cleanup and verification.

The drill's 25% process allocation ceiling bounds the isolated experiment. It
did not reach that ceiling and does not imply the model naturally leaks memory.
