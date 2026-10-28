# CS-AUAlign-ViT

Research implementation progressing from POST–Swin to POST-style ViT and
AU-guided cross-domain adaptation. Model training and reproduction are not yet
implemented.

The RAF-DB → ExpW data pipeline is described in [data/README.md](data/README.md):
native annotation mapping, frozen splits, independent MTCNN preprocessing,
landmark metadata, source/target/evaluation loaders and audit commands.

Use Python 3.12 and `uv sync --python 3.12 --extra preprocess` for preparation.
Fixture checks are `uv run --python 3.12 pytest -q`; no real data or detector
weights are needed for those tests. Full-data acceptance requires execution on
the selected data host, not just passing fixture tests.
