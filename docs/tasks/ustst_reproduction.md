# USTST reproduction (closed-set RAF-DB → ExpW)

Baseline for the comparison table: Guo et al., *USTST: unsupervised self-training
similarity transfer for cross-domain facial expression recognition*, MTAP 2024
(83:41703–41723). The paper has no public code, so this is a re-implementation on
the POST scaffold (the E0 reproduction) under the same protocol as E0.

## Code map

| Paper component | Repo |
| --- | --- |
| CST with CAST blocks (W-MCA/SW-MCA), Eq. 2 `Attention(Q_s, K_t, V_t)` → `F_c` | `models/ustst.py::window_attention`, `cast_block`, `USTSTModel.forward_features` |
| Linear heads `f` and `f_c` | `models/ustst.py::USTSTHeads` |
| `X_Tw` / `X_Ts` (RandAugment) | `data/transforms.py::WeakStrongTransform`, `data/datasets.py::build_weak_strong_target_loader` |
| K-means pseudo-labels `ŷ_k` before every epoch | inherited `refresh_target_supervision` (`kmeans_interval: 1`) |
| `L_th`, `L_ts`, `L_ch`, `L_cs` (Eqs. 7–12) | `training/ustst_losses.py::target_gates`, `hard_pseudo_label_loss`, `soft_pseudo_label_loss` |
| `L_sim` (Eqs. 3–4) | `training/ustst_losses.py::similarity_loss` |
| `L_m` MMD (Eq. 5) | `training/ustst_losses.py::gaussian_mmd` |
| `L_a` adversarial (Eq. 6) | reused `models/post/adversarial.py::AdversarialAlignment` |
| Eq. 14 total, update step | `training/ustst.py::USTSTTrainer.train_step` |
| STR, facial kernel patch fusion with random-sized patches, `thr_r` | reused `SelfTrainingResampler` (`str.confidence_threshold: 0.2`, `str.patch_scale_range`) |
| Entry points | `scripts/train_ustst.py`, `scripts/evaluate_ustst.py`, `configs/training/ustst.yaml` |

## Settings taken from the paper

- `L = L_s + L_t + 0.01 L_c + 0.01 L_sim + 0.01 L_m + 0.1 L_a`.
- `thr = 0.95`, `thr_r = 0.2`, 256 epochs, 10 resamplings, SGD lr 0.001 with cosine decay.
- K-means on target features before every epoch; RandAugment for `X_Ts`.

## Settings shared with E0 (protocol, not paper)

Same source checkpoint (`runs/source-pretraining-val/best.pt`), RAF-DB v5 and
ExpW v1 manifests, batch size 64 per stream, momentum 0.9, weight decay 0.001,
AMP, seed 42, STR at epochs 25, 50, …, 250, adversarial head 768→256→1 with
dropout 0.5 and GRL λ = 1.

**Model selection uses target labels.** As in E0, `best.pt` is the epoch with the
highest ExpW *validation* accuracy, so target labels take part in choosing the
model; report it as "epoch selected on ExpW validation". The final epoch-256
checkpoint (`adaptation.pt`, a rule fixed in advance with no target labels) is also
evaluated on ExpW test and reported next to it.

## Run, resume and results

- A run without `--resume` refuses an `--output` that already holds run state
  (`adaptation*.pt`, `best.pt`, `metrics.jsonl`, `test_results.json`, `str/`) before
  writing anything, so two runs never mix results or share STR transactions.
- `--resume` accepts only the run's rolling `adaptation.pt`. Training writes to that
  file, so resuming from `best.pt` would overwrite the selected checkpoint.
  `best.pt` and the boundary snapshots are read-only results.
- The checkpoint is saved before an epoch is validated. If a run stops between the
  two, the resume validates the saved epoch first, writes its row with
  `recovered_after_interrupt: true` and updates `best.pt` if needed. Only then does
  training continue. A partially written last row of `metrics.jsonl` is dropped and
  recovered the same way. Invalid JSON before the last row raises an error. The file
  is rewritten with `os.replace`, so an interruption cannot lose earlier rows.
- `--max-steps-per-epoch` (smoke) is stored in the checkpoint config. A smoke
  checkpoint therefore cannot be resumed as a full run, and a full one cannot be
  resumed as a smoke run. `evaluate_ustst.py` writes the value into its output.
- When epoch `max_epochs` completes, `test_results.json` holds the ExpW test metrics
  for `best` (selected on validation) and for `final` (epoch 256), each with its
  epoch and selection rule. A resume of a finished run rewrites it.

## Repo decisions where the paper is silent or ambiguous

1. **CAST blocks.** One weight-shared Swin-T. Every block of every stage
   (`model.cross_start_block: 0`) carries a cross stream `F_c` next to the native
   streams. `F_c` starts from the source patch embedding. In each block its queries
   come from its own tokens and its keys/values from the `X_Tw` stream of the same
   block, inside the same 7×7 window partition, so regular blocks are W-MCA and
   shifted blocks are SW-MCA with the standard shift mask. The residual and MLP
   follow the query (cross) stream; patch merging and the final norm are shared.
   The native `F_S`, `F_Tw` and `F_Ts` streams are exactly the pretrained Swin-T.
   `cross_start_block: 11` gives the earlier Stage-4-only variant for ablation.
   Cost: four streams per step instead of three.
2. **Inference.** Native self-attention target path plus classifier `f`.
3. **Heads initialisation.** `f` and `f_c` are bias-free linear heads initialised
   from the source prototypes, so epoch 0 equals the source model. The k-means
   centroids are initialised from the rows of `f`, so cluster index = class index.
4. **Soft labels.** The paper's text says "clear if entropy is higher", which
   contradicts Fig. 7 and Eqs. 9/12. We implement the equations: the soft branch
   fires when normalised entropy `H(p)/log K > thr_e`, with `thr_e = 0.5` (value not
   given in the paper). Soft labels are enabled only when the batch contains at
   least one prediction above `thr` (Section 3.3). The soft target is
   `softmax(f(X_Tw))`, detached.
5. **Hard labels.** `ŷ = argmax f(X_Tw)` is used only when `max p > thr` and
   `ŷ = ŷ_k`.
6. **`L_sim`.** Source images are anchors. The candidates are the target weak views
   with `ŷ = ŷ_k`, labelled by `ŷ`. Batch-hard selection: the positive is the
   same-class target with the largest Eq. 3 distance and the negative the
   different-class target with the smallest. The hinge uses the squared L2 norms
   of Eq. 4 (`n` × the Eq. 3 mean, with `n` = 7 classes). The margin is 0.2 (not
   given in the paper).
7. **`L_m`.** Multi-bandwidth Gaussian-kernel MMD (DAN style: 5 kernels, ×2 spacing,
   bandwidth from the batch mean distance) between pooled `F_s` and `F_Tw`.
8. **`L_a`.** Discriminator on pooled `F_s` vs `F_Tw`, with BCE and a GRL.
9. **Gated losses** average over the full batch (the "otherwise 0" form), so a
   batch with no selected sample adds 0.
10. **Random-sized STR patches.** Each of the four ROIs (two eyes, two mouth
    corners) has side `U(0.3, 0.5) × inter-ocular distance`. The range is an
    assumption centred on POST's fixed 0.4. The same four factors are used for the
    target crop and the donor paste. They are seeded by `(seed, round, target_id)`,
    so a resumed STR round recreates identical images, and they are recorded in the
    STR transaction. POST keeps its fixed ROIs, since the option defaults to off.

## Verification

- CPU tests: `tests/test_ustst.py`.
  - Native path equals the pretrained Swin-T.
  - The cross stream matches Eq. 2 computed by hand.
  - Gate selection, masked losses, the batch-hard triplet value and MMD sanity.
  - Tiny end-to-end train → STR → resume → evaluate.
- GPU smoke on AIVN with `--max-steps-per-epoch`: VRAM at bs 64, finite losses,
  gate ratios, k-means time.

## Known differences from the paper

The paper reports ExpW only as a target for its own source setting. Our number
uses the repo's RAF-DB v5 / ExpW v1 split and E0's checkpoint selection, so it is
comparable to E0 and OURs, not to the paper's table.
