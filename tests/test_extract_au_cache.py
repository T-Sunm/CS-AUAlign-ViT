import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "extract_au_cache.py"
    spec = importlib.util.spec_from_file_location("extract_au_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_cache(path, columns):
    rows = (
        {"sample_id": "success", "status": "success", "reason": ""},
        {
            "sample_id": "software",
            "status": "failed",
            "reason": "AttributeError: compatibility failure",
        },
        {
            "sample_id": "no-face",
            "status": "failed",
            "reason": "RuntimeError: No face landmarks",
        },
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sample_id", "status", "reason", *columns]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {column: 1.0 if row["status"] == "success" else "" for column in columns})


def test_libreface_compat_uses_lanczos_and_limits_detection_to_one_face(monkeypatch):
    script = load_script()
    calls = []

    def face_mesh(*args, **kwargs):
        calls.append((args, kwargs))
        return "mesh"

    face_mesh_module = SimpleNamespace(FaceMesh=face_mesh)
    detector = SimpleNamespace(
        mp=SimpleNamespace(solutions=SimpleNamespace(face_mesh=face_mesh_module))
    )
    monkeypatch.delattr(Image, "ANTIALIAS", raising=False)
    monkeypatch.setattr(script.importlib, "import_module", lambda _name: detector)

    script._apply_libreface_compat()
    result = face_mesh_module.FaceMesh(static_image_mode=True, max_num_faces=2)

    assert Image.ANTIALIAS == Image.Resampling.LANCZOS
    assert result == "mesh"
    assert calls == [((), {"static_image_mode": True, "max_num_faces": 1})]


def test_retry_replaces_selected_failures_without_touching_other_records(tmp_path, monkeypatch):
    script = load_script()
    manifest = tmp_path / "processed.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "split": "train",
                    "usable": True,
                    "processed_path": f"{sample_id}.png",
                    "class_name": "anger",
                }
            )
            + "\n"
            for sample_id in ("success", "software", "no-face")
        )
    )
    output = tmp_path / "cache.csv"
    write_cache(output, script.AU_COLUMNS)
    calls = []

    def get_attributes(path, device):
        calls.append((Path(path).name, device))
        return {"au_intensities": dict.fromkeys(script.AU_COLUMNS, 2.0)}

    monkeypatch.setitem(sys.modules, "libreface", SimpleNamespace(get_facial_attributes=get_attributes))
    monkeypatch.setattr(script, "_apply_libreface_compat", lambda: None)
    result = script.run(
        SimpleNamespace(
            manifest=manifest,
            output=output,
            device="cpu",
            flush_every=1,
            retry_failed=True,
            retry_error_type=["AttributeError"],
        )
    )

    rows = {row["sample_id"]: row for row in csv.DictReader(output.open())}
    assert calls == [("software.png", "cpu")]
    assert len(rows) == 3
    assert rows["success"]["au_1_intensity"] == "1.0"
    assert rows["software"]["status"] == "success"
    assert rows["no-face"]["reason"] == "RuntimeError: No face landmarks"
    assert result["retried"] == result["retry_success"] == 1
    assert result["retry_failed"] == 0


def test_retry_error_type_requires_retry_flag(tmp_path, monkeypatch):
    script = load_script()
    monkeypatch.setitem(sys.modules, "libreface", SimpleNamespace())
    monkeypatch.setattr(script, "_apply_libreface_compat", lambda: None)

    try:
        script.run(
            SimpleNamespace(
                manifest=tmp_path / "unused.jsonl",
                output=tmp_path / "unused.csv",
                device="cpu",
                flush_every=1,
                retry_failed=False,
                retry_error_type=["AttributeError"],
            )
        )
    except ValueError as error:
        assert str(error) == "--retry-error-type requires --retry-failed"
    else:
        raise AssertionError("Expected invalid retry arguments to fail")
