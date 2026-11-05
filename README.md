# CS-AUAlign-ViT

Research implementation progressing from POST–Swin to POST-style ViT and
AU-guided cross-domain adaptation. Source pretraining fine-tunes ImageNet
Swin-Tiny on RAF-DB while learning a seven-class prototype bank.

The RAF-DB → ExpW data pipeline is described in [data/README.md](data/README.md):
native annotation mapping, frozen splits, independent MTCNN preprocessing,
landmark metadata, source/target/evaluation loaders and audit commands.

Use Python 3.12 and `uv sync --python 3.12 --extra preprocess` for preparation.
Fixture checks are `uv run --python 3.12 pytest -q`; no real data or detector
weights are needed for those tests. Full-data acceptance requires execution on
the selected data host, not just passing fixture tests.

Run source pretraining after preparing and auditing RAF-DB:

```bash
uv run --python 3.12 python scripts/train_source.py \
  --manifest /path/to/raf-v1/processed.jsonl \
  --config configs/training/source_pretraining.yaml \
  --output /path/to/runs/source-pretraining
```

The default config uses CUDA automatic mixed precision (AMP) and holds out a
fixed, stratified 10% of RAF-DB train for validation. It shows batch progress,
writes `last.pt`, saves the highest validation-accuracy model as `best.pt`, and
stops after 10 epochs without improvement. Resume an interrupted run
with the same command plus `--resume /path/to/runs/source-pretraining/last.pt`.
`best.pt` contains the backbone, source prototypes, class mapping and the
model, preprocessing and training configurations required by the next task.

Resume the current RAF-DB → ExpW adaptation run with:

```bash
uv run --python 3.12 python scripts/train_adaptation.py \
  --source-manifest data/processed/raf-v5-deduplicated/processed.jsonl \
  --target-manifest /home/hung/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --source-checkpoint runs/source-pretraining-val/best.pt \
  --resume runs/adaptation/run-001/adaptation.pt
```

The target manifest above is an absolute path on the data host. When AU
guidance or AU alignment is enabled, also pass `--source-au-cache` and
`--target-au-cache` pointing to the corresponding Task 7 CSV caches. The AU
alignment calibration notebook is
[`notebooks/au_alignment_calibration.ipynb`](notebooks/au_alignment_calibration.ipynb).
