# Measured source revisions

`worker-before-cleanup.py.txt` preserves the exact initial worker source used by
the early benchmarks. Its SHA-256 matches their per-run `source_sha256` field.
The current executable is `experiments/single_gpu/worker.py`.

`cleanup-change.diff` records the only worker change during the suite: removing
the retained `locals()` dictionary reference from the graph-retention cleanup.
That branch executes only in the retention drill; ordinary training computation
was unchanged. The historical source is a text artifact, not another entry point.

The other measured source change was the original trainer's RNG-loading fix.
The before-fix `train.py` is available from Git commit `4672cbb`; current
`train.py` loads onto CPU. Its pre/post hashes are retained in the run summaries.
