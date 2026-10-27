# RAF-DB → ExpW data pipeline

Code/configuration belong in Git. Raw images, processed crops, sample-level
manifests and detector weights live on the execution host. Supply paths explicitly;
the shared configuration contains no host paths. Run commands from the repository
root in an installed Python 3.12 environment (`uv sync --python 3.12 --extra
preprocess`). The current detector dependency constrains Torch to 2.2–2.3.

## Native inputs and mapping

RAF Basic7: original images plus `list_partition_label.txt` (some distributions
spell it `list_patition_label.txt`). Each line is `filename label`; `train_` and
`test_` prefixes define official roles. Aligned/224px distributions must not be
silently substituted for original images.

ExpW: `label/label.lst` plus `data/image/origin/`. Read all eight fields, preserving
`filename + face_id` identity and the annotated face bbox. Label meanings are
documented in the distributed `label/readme.txt` from the
[ExpW release](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html).

| Canonical ID | Class | RAF native label | ExpW native label |
|---|---|---|---|
| 0 | anger | 6 | 0 |
| 1 | disgust | 3 | 1 |
| 2 | fear | 2 | 2 |
| 3 | happiness | 4 | 3 |
| 4 | neutral | 7 | 6 |
| 5 | sadness | 5 | 4 |
| 6 | surprise | 1 | 5 |

Duplicate identical ExpW face records are removed; conflicting repeated IDs fail
preparation. Split groups combine all faces of a photograph and byte-identical
photographs under different names. By default a seeded SHA-256 rule assigns
groups to 80% train, 10% validation and 10% test, without consulting target labels.
These are expected proportions, not exact sample quotas. The seed is 20261027.

To use a suitable benchmark split, supply `--benchmark-split split.json`, an exact
mapping of every unique `expw:<filename>:<face_id>` to `train`, `validation` or
`test`. Cross-group overlap or incomplete coverage fails. The existing AIVN native
lists were inspected and had photograph-name overlap (train/validation 2768,
train/test 3900, validation/test 970), so they are unsuitable for this protocol.
Byte grouping does not establish subject independence or rule out near duplicates.

## Independent preparation

```bash
uv run --python 3.12 --extra preprocess python scripts/prepare_data.py \
  --dataset raf_db --image-root /path/to/RAF/basic/Image/original \
  --annotation /path/to/RAF/basic/EmoLabel/list_patition_label.txt \
  --face-boxes /path/to/RAF/basic/Annotation/boundingbox \
  --output /path/to/output/raf-v1 --device cuda:0

uv run --python 3.12 --extra preprocess python scripts/prepare_data.py \
  --dataset expw --image-root /path/to/ExpW/data/image/origin \
  --annotation /path/to/ExpW/label/label.lst \
  --output /path/to/output/expw-v1 --device cuda:0
```

Use `--limit 32` with a **separate** output for a smoke check. A full run cannot
resume a subset output. The config is `configs/data/raf_cross_domain.yaml`.

Both datasets use the same pretrained MTCNN, minimum face size 20, thresholds
0.6/0.7/0.7 and scale factor 0.709. Its three detector weights are bundled with
`facenet-pytorch==2.6.0`; their hashes are saved. No recognition backbone is loaded.
See the [upstream detector](https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py).

For ExpW, select the detection with maximum IoU to the annotated bbox, rejecting
IoU below 0.3 rather than assigning another person's expression. For RAF without
face annotations, choose the largest clipped detected face; this heuristic is an
explicit implementation choice, not a guarantee of annotated identity. Ties use
detection probability. Prefer RAF's official `boundingbox.zip` via `--face-boxes`:
its per-image floating-point xyxy annotation uses the same IoU matching policy as
ExpW, and a digest of all reference box files is recorded. This option is used for
the original RAF archive supplied for Task 1. MTCNN operates on the input image; crops have no alignment,
margin or offline resize. Boxes use clipped integer xyxy edges (floor left/top,
ceil right/bottom); crops smaller than 20px on either axis are rejected. RGB crops
are lossless PNGs. Original and crop coordinates for all five landmarks are stored.
Image-left/right landmark names refer to image coordinates, not anatomical sides.

Outputs per preparation:

- `preparation.json`: detector/config/input identity and coordinate policy.
- `splits.jsonl`: frozen split and annotation rows, including canonical labels.
- `journal.jsonl`: per-sample completion records for resume.
- `processed.jsonl`: final rows including usability, rejection reason, crop path,
  native size, boxes, landmarks and checksums.
- `crops/`: processed images; paths in manifests are relative to their directory.
- `summary.json`: original unique face/photo counts, usable/rejected counts and
  usable class distribution for each split; duplicate annotation count is separate.
- `status.json`: running/completed/failed state and progress.

Rerun the identical command to resume; changing inputs, split or preparation
configuration requires a new output. Use only one writer per output directory.
Operational snapshots, environments and run provenance belong outside crop data.

## Audit and load

```bash
uv run --python 3.12 python scripts/audit_data.py \
  --manifest /path/to/output/expw-v1/processed.jsonl \
  --output /path/to/run/expw-audit.json --check-loaders --verify-originals --batch-size 8

uv run --python 3.12 python scripts/audit_data.py \
  --manifest /path/to/output/raf-v1/processed.jsonl \
  --output /path/to/run/raf-audit.json --check-loaders --verify-originals \
  --raf-annotation /path/to/RAF/basic/EmoLabel/list_patition_label.txt
```

The audit checks every crop checksum/shape, landmark coordinate conversion,
unchanged splits and byte-hash split overlap. It then exercises loaders without
importing MTCNN. A missing/invalid crop fails the audit.
With `--verify-originals`, every usable PNG must also match the RGB pixels of
the recorded bounding box in its original input image exactly; original hashes
and sizes must agree. A split with input records but no usable samples fails
the loader check rather than being silently skipped.

RAF must retain the official partition. The supplied original release has three
byte-identical image groups (eight rows) spanning official train/test. The audit
reports this limitation after verifying every row against the checksum-bound
original annotation supplied through `--raf-annotation`; it does not move or
delete those rows. Without that verification, cross-split bytes fail the audit.
ExpW still requires zero cross-split byte overlap. RAF source-test measurements
must disclose this official-partition limitation.

```python
from cs_aualign_vit.data.datasets import build_loader

source = build_loader(raf_manifest, "source_train", batch_size=32, seed=42)
target = build_loader(expw_manifest, "target_adaptation", batch_size=32, seed=42)
test = build_loader(expw_manifest, "evaluation", batch_size=64)
```

Source returns `(tensor, int_label)`. Target returns `(tensor, sample_id)` and
retains no ground-truth labels in dataset samples or `metadata(sample_id)`.
Evaluation returns `(tensor, int_label)` and defaults to test; explicit
`split="validation"` is supported. Preparation manifests contain labels for
auditing/evaluation and must not be handed directly to adaptation objectives or
used for target-based checkpoint selection.

Training uses resize → random horizontal flip(0.5) → tensor → normalization;
evaluation has no random flip. Defaults are 224×224 and ImageNet normalization;
`build_loader` exposes image size/mean/std. Batch size is a required caller
argument, independent of preprocessing configuration. No loader reruns detection.
`dataset.metadata(sample_id)` retrieves native crop landmarks for STR. Any later
online STR integration must transform coordinates along with image resize/flip;
this task does not implement STR or source-target pairing.

## Checks

```bash
uv run --python 3.12 pytest -q
uv run --python 3.12 ruff check src scripts tests
uv run --python 3.12 ruff format --check src scripts tests
```

Fixture tests do not instantiate MTCNN, fetch pretrained weights or require GPU.
Full preprocessing and the audit must run on actual data before claiming Task 1
completion. Neither constitutes POST model reproduction or training evidence.
