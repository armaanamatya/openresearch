# A/B report — adam-ab-20260611

Generated 2026-06-12T06:49:44.147501+00:00 · select=latest · arms found: bes×1, unstamped×1

| arm | project | score | adj | verdict | meets target | iters | wall | cost |
|---|---|---|---|---|---|---|---|---|
| control | prj_6d41d2f09c026403 | 0.716 | 0.716 | reproduced | yes | 6 | 14.6h | 2.917 USD |
| bes | prj_6d41d2f09c026403_ab_bes | 0.5327 | 0.5327 | failed | no | 7 | 20.0h | 0 USD |

## Δ (bes − control)

- overall_score: **-0.1833**
- compute_adjusted_score: -0.1833
- wall_clock: 5.455h
- cost: -2.917 USD

## BES candidate pool (static SELECT scores)

| candidate | ok | static score |
|---|---|---|
| rlm_impl#0 | True | 0.5464 |
| rlm_impl#1 ← selected | True | 0.643 |

## Top leaf-level moves

| leaf | control | bes | Δ |
|---|---|---|---|
| 4218ef05ad214ec487169e41e17914a2 | 0.2 | 1 | 0.8 |
| 7f454bc91d3747e496d21adbd4554e28 | 1 | 0.4 | -0.6 |
| 52348d0d94744d75b9a1d4df411b9da5 | 1 | 0.4 | -0.6 |
| e678b76684f04df0a19e12821c2c97e1 | 0.4 | 1 | 0.6 |
| 699d7e2bec3c4b529d23435428cdec23 | 0.4 | 1 | 0.6 |
| 5554e12e683e4e9d81572a8274548d14 | 1 | 0.4 | -0.6 |
| 93f6709b58244cbfada11975d9d9e86c | 1 | 0.4 | -0.6 |
| 728ec345e4184245bc5a3eb6e6a70613 | 0.4 | 0 | -0.4 |

