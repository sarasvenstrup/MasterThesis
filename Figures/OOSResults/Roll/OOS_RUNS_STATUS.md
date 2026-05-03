# OOS Rolling Run Status

_Generated: 2026-05-03 09:16 -- re-run `python Code/gen_oos_status.py` after pulling to refresh._

## train5Y / test6M / step6M (main comparison)

| Model | Epochs | Started | Finished | Windows |
|---|---|---|---|---|
| dim2_baseline | 2500 | 2026-03-27T10:57:44 | 2026-03-27T15:56:19 | 18/18 done |
| dim2_baseline | **3500** | 2026-05-02T07:03:56 | 2026-05-02T11:35:12 | 18/18 done |
| dim2_stable | 2500 | 2026-03-30T18:56:02 | 2026-03-30T23:43:53 | 18/18 done |
| dim2_stable | **3500** | 2026-05-01T17:23:19 | 2026-05-01T21:41:51 | 18/18 done |
| dim3_baseline | 2500 | 2026-03-29T09:06:56 | 2026-03-29T15:38:44 | 18/18 done |
| dim3_baseline | **3500** | 2026-05-02T11:35:17 | 2026-05-02T16:15:13 | 18/18 done |
| dim3_stable | 2500 | 2026-03-28T21:08:57 | 2026-03-29T03:33:42 | 18/18 done |
| dim3_stable | **3500** | - | - | MISSING |
| dim4_baseline | 2500 | 2026-03-29T15:38:52 | 2026-03-29T21:12:47 | 18/18 done |
| dim4_baseline | **3500** | 2026-05-02T16:15:18 | 2026-05-02T21:02:10 | 18/18 done |
| dim4_stable | 2500 | 2026-03-29T03:33:47 | 2026-03-29T08:55:08 | 18/18 done |
| dim4_stable | **3500** | - | - | MISSING |

## train3Y / test3M / step6M (early experiments)

| Model | Epochs | Started | Finished | Windows |
|---|---|---|---|---|
| dim1_baseline | 2500 | 2026-03-18T22:52:08 | 2026-03-19T03:35:14 | 34/34 done |
| dim2_baseline | 2500 | 2026-03-19T07:57:22 | 2026-03-19T13:50:22 | 34/34 done |

## Summary
- Complete: 10/12 main runs
- MISSING: dim3_stable ep3500, dim4_stable ep3500
- To run missing: `python Code/run_stable_ep3500.py`
