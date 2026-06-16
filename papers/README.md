# Bundled papers — the default reproduction targets

These are the top reproduction targets, **shipped in the repo so a fresh clone can
select and reproduce them immediately — no network fetch required.**

| id | paper | arXiv | datasets | reproduce with |
|----|-------|-------|----------|----------------|
| `sdar` | Self-Distilled Agentic Reinforcement Learning | 2605.15155† | ALFWorld, Search-QA, WebShop | `reproduce sdar` |
| `adam` | Adam: A Method for Stochastic Optimization | 1412.6980 | MNIST, IMDB, CIFAR-10 | `reproduce adam` |
| `allcnn` | Striving for Simplicity: The All Convolutional Net | 1412.6806 | CIFAR-10/100 | `reproduce allcnn` |

† SDAR's arXiv id is future-dated and does **not** resolve on arxiv.org — bundling
is what makes it reproducible offline (a plain `reproduce 2605.15155` otherwise
produced a degraded, near-empty run).

## How it works
- `papers/registry.json` maps each paper's **id / aliases / arXiv-id** → its bundled
  `papers/*.pdf` + its `--paper-hint`.
- On `python -m backend.cli reproduce <id|alias|arxiv>`, the CLI resolves a registered
  source to the in-repo PDF and **auto-applies its `--paper-hint`** — so any of
  `reproduce sdar`, `reproduce 2605.15155`, `reproduce self-distilled` just work.
- The same registry feeds `GET /papers` (the lab's selectable presets).
- Resolution is fail-soft: an unregistered id / missing registry falls straight
  through to the normal arXiv / PDF / DOI fetch path.

The PDFs are rendered from each paper's parsed full text (the reproduction pipeline
is text-only). To add a paper: drop its PDF here and add an entry to `registry.json`.
Code: `backend/services/ingestion/paper_registry.py`.
