# A/B report — allcnn-ab-20260611

Generated 2026-06-11T16:26:59.463015+00:00 · select=latest · arms found: bes×1, control×1

| arm | project | score | adj | verdict | meets target | iters | wall | cost |
|---|---|---|---|---|---|---|---|---|
| control | prj_0a3202fc187bb692_ab_control | 0.6526 | 0.6304 | reproduced | yes | 10 | 13.4h | 4.951 USD |
| bes | prj_0a3202fc187bb692_ab_bes | 0.7378 | 0.6925 | partial | yes | 10 | 13.3h | 3.265 USD |

## Δ (bes − control)

- overall_score: **0.08519**
- compute_adjusted_score: 0.06207
- wall_clock: -0.06361h
- cost: -1.685 USD

## BES candidate pool (static SELECT scores)

| candidate | ok | static score |
|---|---|---|
| rlm_impl#0 | True | 0.5488 |
| rlm_impl#1 ← selected | True | 0.5567 |

## Top leaf-level moves

| leaf | control | bes | Δ |
|---|---|---|---|
| d442e584dfd249018d62f1648846c183 | 0 | 0.4 | 0.4 |
| 985ca8baab0441a69c93f17eeb61d80c | 0 | 0.4 | 0.4 |
| b7325d4fe66147368e9b5d63d05a60f5 | 0.4 | 0.7 | 0.3 |
| ad46e5ab07ad4fe9a66ff4c35cea005b | 0.4 | 0.7 | 0.3 |
| 85042ac08ac94165bddb5a6e8fd606a4 | 0.4 | 0.7 | 0.3 |
| 2fb0fff0618746be84f5b7ff621f0542 | 0.2 | 0 | -0.2 |

