# CS-AUAlign-ViT

The standalone 12-AU heatmap extractor is documented in
[`tools/au_heatmap/README.md`](tools/au_heatmap/README.md).

Build and audit the learned spatial caches for Task 9.2 before starting G1:

```bash
tools/au_heatmap/.venv/bin/python scripts/extract_au_heatmap_cache.py \
  --manifest data/processed/raf-v5-deduplicated/processed.jsonl \
  --output runs/au_heatmap_cache/raf_db_train \
  --checkpoint tools/au_heatmap/checkpoints/disfa_adaptation_f0.pth \
  --device cuda:0 --batch-size 16

tools/au_heatmap/.venv/bin/python scripts/extract_au_heatmap_cache.py \
  --manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --output runs/au_heatmap_cache/expw_train \
  --checkpoint tools/au_heatmap/checkpoints/disfa_adaptation_f0.pth \
  --device cuda:0 --batch-size 16

uv run --active python scripts/audit_au_heatmap_cache.py \
  --source-manifest data/processed/raf-v5-deduplicated/processed.jsonl \
  --source-au-cache runs/au_cache/raf_db_train.csv \
  --source-heatmap-cache runs/au_heatmap_cache/raf_db_train \
  --target-manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --target-au-cache runs/au_cache/expw_train.csv \
  --target-heatmap-cache runs/au_heatmap_cache/expw_train \
  --output runs/au_heatmap_cache/audit.json
```

The extractor checkpoints atomically every 5,000 pending records. Re-running
the same command skips successful rows and retries failed rows.

After both caches pass the audit, run the short G1 training pilot and attention
gate before the full experiment:

```bash
uv run --active python scripts/pilot_au_attention_training.py \
  --target-manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --source-heatmap-cache runs/au_heatmap_cache/raf_db_train \
  --target-heatmap-cache runs/au_heatmap_cache/expw_train \
  --config configs/training/ablation/g1_heatmap.yaml \
  --output runs/adaptation/g1-heatmap-pilot \
  --steps 3 --batch-size 16 \
  --g-scale 3000000 --lambda-learning-rate 0.000001

uv run --active python scripts/calibrate_au_attention.py \
  --checkpoint runs/adaptation/g1-heatmap-pilot/adaptation.pt \
  --target-manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --source-heatmap-cache runs/au_heatmap_cache/raf_db_train \
  --target-heatmap-cache runs/au_heatmap_cache/expw_train \
  --scales 3000000
```

Only after the gate passes, start the full G1 run:

```bash
uv run --active python scripts/train_adaptation.py \
  --source-manifest data/processed/raf-v5-deduplicated/processed.jsonl \
  --target-manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --source-checkpoint runs/source-pretraining-val/best.pt \
  --source-au-cache runs/au_cache/raf_db_train.csv \
  --target-au-cache runs/au_cache/expw_train.csv \
  --source-heatmap-cache runs/au_heatmap_cache/raf_db_train \
  --target-heatmap-cache runs/au_heatmap_cache/expw_train \
  --config configs/training/ablation/g1_heatmap.yaml \
  --output runs/adaptation/g1-heatmap-100epochs-seed42
```

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
  --target-manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --source-checkpoint runs/source-pretraining-val/best.pt \
  --resume runs/adaptation/run-001/adaptation.pt
```

The target manifest above is an absolute path on the data host. When AU
guidance or AU alignment is enabled, also pass `--source-au-cache` and
`--target-au-cache` pointing to the corresponding Task 7 CSV caches. The AU
alignment calibration notebook is
[`notebooks/au_alignment_calibration.ipynb`](notebooks/au_alignment_calibration.ipynb).

Evaluate the completed E1, E2 and E4 ablations on the frozen ExpW test split:

```bash
uv run --python 3.12 python scripts/evaluate_adaptation.py \
  --manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --checkpoint runs/adaptation/e1_attention-100epochs-seed42-bs45/best.pt \
  --split test \
  --device cuda:0 \
  --output runs/adaptation/e1_attention-100epochs-seed42-bs45/expw_test.json

uv run --python 3.12 python scripts/evaluate_adaptation.py \
  --manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --checkpoint runs/adaptation/e2_alignment-100epochs-seed42-bs45/best.pt \
  --split test \
  --device cuda:0 \
  --output runs/adaptation/e2_alignment-100epochs-seed42-bs45/expw_test.json

uv run --python 3.12 python scripts/evaluate_adaptation.py \
  --manifest /home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl \
  --checkpoint runs/adaptation/e4_alignment_shuffled-100epochs-seed42-bs64/best.pt \
  --split test \
  --device cuda:0 \
  --output runs/adaptation/e4_alignment_shuffled-100epochs-seed42-bs64/expw_test.json
```

Audit the landmarks and LibreFace AU guidance generated after STR with
[`notebooks/audit_str_au_landmarks.ipynb`](notebooks/audit_str_au_landmarks.ipynb).
It reads the E3 artifacts directly from
`runs/adaptation/e3-both-100epochs-seed42/str/manifest.jsonl` and
`runs/adaptation/e3-both-100epochs-seed42/source_au_cache.csv`, samples up to
150 STR images, compares donor-inherited landmarks with a fresh MTCNN detection,
and writes `audit_results.csv` plus `review_queue.csv` under `str/audit/`.

Install the development dependencies and register the project environment as a
kernel once:

```bash
uv sync --python 3.12 --group dev
uv run --python 3.12 --group dev python -m ipykernel install --user \
  --name cs-aualign-vite --display-name "CS-AUAlign-ViT"
```

Then open the notebook with the VS Code Jupyter extension and select the
`cs-aualign-vite` kernel. JupyterLab is not required.

The ExpW evaluation manifest remains
`/home/minhvduc/FER-DomainAdaptation/CS-AUAlign-ViT/data/processed/expw-v1/processed.jsonl`
on the data host. The STR audit does not load it because the committed STR
manifest already contains the image path, donor provenance, and inherited
landmarks.
