# AU alignment calibration

Status: calibrated on RAF-DB and ExpW train; automated three-step smoke test passed.

## Inputs

- Seed 42, batch size 64, original usable train samples only.
- RAF-DB AU cache: `runs/au_cache/raf_db_train.csv`, SHA-256 `84697558482cedaa0f9b90a68f1b6759932175bbdab76d074c3dd5977e419095`.
- ExpW AU cache: `runs/au_cache/expw_train.csv`, SHA-256 `5eed79bc158d4c223c8cb5a5498d31e000c7611fd48036bf37f1ce156c02fa3d`.
- AU missing: RAF-DB 1.63%; ExpW 7.70%. Invalid landmark/near-zero mask: RAF-DB 0.24%; ExpW 2.72%.
- ExpW epoch-0 pseudo labels: 65,948 samples, 25,236 trusted.

## Mask and batch reliability

Chosen `tau_mask=1.0`. The lowest P1 mask mass across AUs was 1.6886 on RAF-DB and 0 on ExpW; the lowest P5 was 2.8944 on RAF-DB and 1.2807 on ExpW. Full per-AU mask statistics are produced by `notebooks/au_alignment_calibration.ipynb`.

| `n_eff_min` | Source valid AU | Target valid AU | Valid in both | Batches with local loss |
|---:|---:|---:|---:|---:|
| 2 | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | 100.0% | 99.7% | 99.7% | 100.0% |
| 8 | 97.2% | 94.7% | 92.8% | 100.0% |
| 16 | 69.6% | 64.9% | 53.2% | 100.0% |

Chosen `n_eff_min=8`. Coverage uses `drop_last=True`, seed 42, source cycling, and the same sampled batches for every candidate.

## Fixed AU weights and gradient scale

`q_c` follows AU order `(1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)`:

```text
[0.0004547530261334032, 0.0001833344140322879, 0.009413017891347408,
 0.00044700285070575774, 0.00035880241193808615, 0.000688203377649188,
 0.0018218063050881028, 0.0002874583879020065, 0.0007104425458237529,
 0.0002812081656884402, 0.010985778644680977, 0.001102050649933517]
```

Gradient calibration used source initialization and epoch-0 target pseudo labels. Four fixed train batch pairs gave valid AU counts of 11, 11, 12, and 10. `L_POST` and `L_local` gradients were measured on the same BCS Block 1 encoder parameters, with `r_target=0.1`.

| Batch | `L_POST` | `L_local` | Raw gradient ratio | `eta_0` |
|---:|---:|---:|---:|---:|
| 1 | 1.4949 | 151.0712 | 156.7086 | 0.0006381272 |
| 2 | 1.0905 | 105.1830 | 126.4622 | 0.0007907502 |
| 3 | 0.9057 | 90.5518 | 106.8121 | 0.0009362239 |
| 4 | 1.9095 | 103.3935 | 84.1764 | 0.0011879811 |

Chosen `eta=0.0008634870173409581` (median); P10 `0.0006839141133241355`, P90 `0.0011124538723379374`. With this fixed `eta`, the four weighted gradient ratios are approximately 0.135, 0.109, 0.092, and 0.073. Local gradient was nonzero and finite in all four batches; non-finite count was 0. No ExpW validation or test accuracy was used to choose these values.

The calibrated values are in `configs/training/adaptation.yaml`; alignment remains disabled for normal runs until explicitly enabled.

## Three-step smoke test

The automated smoke test follows the existing adaptation-test pattern: two train steps, save checkpoint, construct a fresh trainer, resume, then run step three. It checks finite metrics and gradients, local-loss coverage and per-AU `N_eff`, `global_step=3`, attention-off behavior, and rejection of changed alignment config or AU cache identity.

```bash
uv run --python 3.12 pytest \
  tests/test_adaptation.py::test_alignment_smoke_steps_checkpoint_and_resume -q
```

User-run result: `1 passed in 2.12s`. This verifies the three train steps, checkpoint/resume, finite metrics and gradients, AU coverage, and rejection of changed config or cache identity on the small deterministic fixture. Full RAF-DB/ExpW execution remains for Task 8.2.
