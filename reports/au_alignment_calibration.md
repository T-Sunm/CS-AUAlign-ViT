# AU alignment calibration

Status: pending real RAF-DB/ExpW manifests and Task 7 AU caches.

Run `notebooks/au_alignment_calibration.ipynb` with seed 42 and batch size 64,
then record:

- AU cache identities and missing rates for both domains;
- mask-mass min/P1/P5/P10/median/P95, near-zero rate and chosen `tau_mask`;
- batch `N_eff` coverage for the fixed candidates and chosen `n_eff_min`;
- the 12 fixed `q_c` values;
- raw and weighted gradient ratios, zero-gradient rate, non-finite count, and
  chosen `eta` with median/P10/P90 evidence.

Do not use ExpW validation or test accuracy for these choices. Copy the final
values into `configs/training/adaptation.yaml` before enabling alignment.
