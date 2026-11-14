# Task 9.2 — G1 technical gate and evidence inventory

## Decision

**Stop G1 before the 100-epoch run.** The registered G1-fixed settings (`g_scale=3e6`, `lambda_learning_rate=1e-6`) failed the attention gate after 1,000 pilot steps: P90 change was 0.0051 (t→s) and 0.0058 (s→t), below 0.01 in both directions. The other three exploratory configurations also failed at least one gate. This is the negative technical-gate outcome allowed by [Task 9.2](../tasks/au_heatmap_learned_spatial_guidance.md); it is not an ExpW performance result.

| Pilot | `g_scale` | `lr_lambda` | P90 t→s / s→t | Saturation t→s / s→t | Gate failure |
|---|---:|---:|---:|---:|---|
| A (G1-fixed) | 3e6 | 1e-6 | 0.0051 / 0.0058 | 1.79% / 2.06% | Both P90 values below 0.01 |
| B | 3e6 | 3e-6 | 0.0107 / 0.0114 | 2.61% / 5.48% | s→t saturation above 5% |
| C | 3e6 | 1e-5 | 0.0184 / 0.0184 | 3.49% / 6.08% | s→t saturation above 5% |
| D | 1e7 | 9e-7 | 0.0188 / 0.0151 | 5.65% / 3.87% | t→s saturation above 5% |

The matrix measured 3,736 valid pairs and 20 strong pairs per pilot. AMP skipped 2–3 of 1,000 steps. A separate 768-batch audit of B found 10,911 valid pairs and 60 strong pairs; P90 was 0.0097 / 0.0103 and saturation 2.48% / 5.55%. B therefore still failed both t→s P90 and s→t saturation. Sources: [matrix report](../../runs/adaptation/g1-pilot-matrix-next/report.md) and [B long audit](../../runs/adaptation/g1-pilot-matrix-next/B.calibration-768.log). The matrix report has checkpoints, resolved configs and train/calibration logs for A–D.

## Data and comparison evidence

The [cache audit](../../runs/au_heatmap_cache/audit.json) covers 11,912 RAF-DB train and 65,948 ExpW train samples. Heatmap success is 100% in both audited sets; G0 and G1 valid rates match (98.14% RAF-DB, 90.66% ExpW), with zero G0-valid samples excluded by heatmap failure. The audit contains per-AU peak and mask-mass statistics. Median cosine similarity for AU1/AU2, AU12/AU25 and AU25/AU26 is respectively 0.372 / 0.387 / 0.957 on RAF-DB and 0.377 / 0.342 / 0.966 on ExpW. Similarity alone is diagnostic, not a pass gate. Both original train-cache directories contain `spatial_masks.npy`, `peaks.npy`, `index.csv` and `metadata.json`.

The existing E3 [ExpW test artifact](../../runs/adaptation/e3-both-100epochs-seed42/expw_test.json) reports 8,062 samples, accuracy 65.65%, balanced accuracy 44.01%, macro F1 43.83% and weighted F1 63.80%, using `best.pt` at epoch 21. Per-class precision/recall/F1 and the confusion matrix are in that artifact. The [ablation report](au_alignment_ablation.md) documents its one-seed limitation. E3 is a **candidate G0 baseline**; reuse requires an explicit identity check against Section 6.3 (source checkpoint, manifests, AU caches, seed, horizon, STR schedule, model selection and code contract). No G1 best checkpoint or ExpW test evaluation exists, so no G0–G1 accuracy delta can be reported. The FANAU checkpoint originates from DISFA; transfer to RAF-DB/ExpW remains a domain-shift limitation.

## Acceptance evidence still to close

The focused test command covering `test_au_heatmap_artifacts.py`, `test_au_attention_pilot_matrix.py`, `test_adaptation.py` and `test_adaptation_end_to_end.py` produced **28 passed in 4.66s**; see [test log](../../runs/adaptation/g1-pilot-matrix-next/task-9-2-tests.log). This is a passing result for those four files, not a complete nine-scenario result.

A subsequent full `pytest -q tests` run produced **146 passed in 12.78s**, exit code 0; see [full test log](../../runs/adaptation/g1-pilot-matrix-next/task-9-2-full-tests.log). This covers the existing unit/integration suite; the scenario table below distinguishes coverage from a complete scenario proof.

| Scenario | Evidence in the 28-test run | Status against full scenario |
|---|---|---|
| 1. Heatmap conversion | `test_heatmap_conversion_applies_relu_pool_normalization_and_peak_gate` passes in the full suite. | Partial: no explicit Inf or near-zero denominator assertion found. |
| 2. Export and identity | `test_batched_heatmap_export_is_resumable_and_keeps_rows` checks shape, retry and row stability. | Partial: identity mismatch, duplicate IDs and atomic failure cases not established by this run. |
| 3. Dataset join and flip | Flip and dataset spatial-validity tests pass in the full suite. | Partial: reordered cache index join is not explicitly tested. |
| 4. Spatial validity and guidance | Guidance direction, missing fallback, dataset validity and adaptation missing-rate checks pass. | Partial: combined landmark/heatmap failure and unaffected POST loss are not pinned together. |
| 5. Alignment | Native row pooling, invalid-region zero loss/gradient and adaptation smoke tests pass. | Partial: exact full G1 loss contract remains unproven by this run alone. |
| 6. G0 equivalence | No explicit old-versus-current comparison of `p`, `M`, `G`, alignment and four BCS outputs found in the 28-test run. | Not established. |
| 7. STR cache lifecycle | Generic STR resume/isolation tests and original AU-cache immutability test passed. | Partial: learned heatmap working-cache lifecycle not established. |
| 8. G1 checkpoint and resume | Generic checkpoint/resume and config-mismatch tests passed. | Partial: G1 artifact identity mismatch and next-batch mask/output equality not established. |
| 9. Real-data gate | Pilot and train-cache audit artifacts above; pilot gate fails. | Gate outcome documented; local-gradient diagnostic still missing. |

E3's resolved config matches G0's training horizon (100), batch size (64), seed (42), STR epochs (25/50/75), guidance scale, lambda learning rate and AU alignment settings. Its ExpW test artifact names `best.pt` at epoch 21. The E3 best checkpoint and G1 pilot A checkpoint both name `runs/source-pretraining-val/best.pt`; both record the same source and target AU-cache SHA-256 identities, matching the current original RAF-DB and ExpW AU caches (`84697558…` and `5eed79bc…`). The E3 working source cache has a different hash after STR, as expected. Neither checkpoint establishes the exact train-manifest identities or historical code contract, so **G0 reuse is not fully verified** against Section 6.3. The G1 pilot resolved config uses batch size 16, one epoch and no STR rounds by design; it is not a full-protocol G1 run.

| Task 9.2 criterion | Evidence found | Remaining verification |
|---|---|---|
| 1. Cache export/schema/identity/atomic write | Original train cache artifacts and `tests/test_au_heatmap_artifacts.py` | Confirm test run and artifact identity/atomic-write checks; files alone do not prove the contract. |
| 2–4. Dataset join/flip, attention/alignment contract, STR lifecycle | Partial mapping above and 146 passing tests | Verify the specific unpinned cases, including STR heatmap working-cache isolation and resume. |
| 5. Nine scenarios, G0 equivalence | Partial mapping above and 146 passing tests | No complete nine-scenario proof or G0 equivalence result established here. |
| 6. Train-only audit and gate | Cache audit, attention probes and finite pilot gradients | Report raw and weighted local-gradient ratio on the same encoder parameters, local loss and nonzero local gradient where eligible; attention gate has failed. |
| 7. G0/G1 protocol and artifacts | E3 test artifact; matching source checkpoint and AU-cache hashes; G1 pilot logs/checkpoints/configs | Verify train manifests and historical code contract before calling E3 a fully validated G0. No G1 full-run/evaluation artifacts are expected after this gate decision. |
| 8. ExpW comparison and limitations | E3 metrics above | State explicitly that G1 test metrics and a performance comparison are unavailable. Retain one-seed and DISFA domain-shift limitations. |

This inventory is based on existing files and logs. It does not claim that the test suite ran or that all acceptance criteria passed. No further parameter search or full training is authorized by this gate decision.
