# Vendored PaperBench paper bundles

Each subdirectory is one PaperBench paper bundle in the format consumed by
`backend/evals/paperbench/bundle.py`. Required files per bundle:

```
<paper_id>/
  config.yaml             # id: <paper_id>, title: "<title>", url: "<openreview>"
  paper.md                # full paper Markdown (PaperBench uses pandoc-converted .md)
  addendum.md             # PaperBench addendum (clarifications + judge notes)
  rubric.json             # PaperBench rubric tree (used for scoring + ceiling math)
  task_instructions.md    # optional: per-paper instructions; otherwise upstream defaults apply
  paper.pdf               # optional original PDF
  blacklist.txt           # optional: URLs/terms blocked from the agent's web access
  assets/                 # optional supplementary materials
```

## Swap in the real PaperBench artifacts

The placeholder bundles in this directory have **synthetic** rubrics so the
end-to-end pipeline can be exercised without API calls. To produce numbers that
are honestly comparable to PaperBench's published baselines (Tables 11 / 15),
overwrite `paper.md`, `addendum.md`, and `rubric.json` with the real upstream
artifacts:

1. Clone the public PaperBench repo (released alongside the OpenAI April 2025
   PaperBench paper). Look for `frontier_evals/paperbench/papers/<paper_id>/`.
2. Copy `paper.md`, `addendum.md`, `rubric.json`, and (if present)
   `instructions.txt` into this directory's `<paper_id>/` subfolder.
3. Re-run `reprolab paperbench summary --paper-id <paper_id>` and verify the
   `node_count`, `leaf_count`, and `task_category_weights` match the values
   reported in PaperBench Table 7.
4. The published baselines used for comparison are stored in
   `backend/cli_paperbench.py::PUBLISHED_BASELINES` and reflect Tables 11 + 15
   of the PaperBench paper.

The `bundle.rubric()` loader is deliberately tolerant of extra fields so the
upstream JSON schema can change without breaking the system.
