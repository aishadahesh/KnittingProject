"""drive_store: which of a model's files go to Drive, and reading rclone's progress."""

import json
from pathlib import Path

import drive_store  # noqa: E402


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_pack_keeps_what_cannot_be_made_again(tmp_path):
    scans = tmp_path / "robot_scans"
    capture = scans / "ur5_a"
    _write(capture / "ur5_0001.png")
    _write(capture / "ur5_0001.json")
    model = scans / "ur5_a_rescan"
    (model / "combined.json").parent.mkdir(parents=True)
    (model / "combined.json").write_text(json.dumps({"sessions": [str(capture)]}), encoding="utf-8")
    output = model / "splat_output"
    kept = ["export/splat_clean.ply", "model_scale.json", "fabric_yarns.npy", "stitch_fit_region_0/stitch_model.json",
            "data/splatfacto/run/config.yml"]
    dropped = ["data/splatfacto/run/nerfstudio_models/step-1.ckpt", "renders/a.png", "stitch_fit_region_0/region.ply"]
    for name in kept + dropped:
        _write(output / name)
    _write(model / "splat_dataset_colmap" / "images" / "ur5_0001.png")
    _write(model / "splat_dataset_colmap" / "colmap_masks" / "ur5_0001.png")
    _write(model / "splat_dataset_colmap" / "database.db")
    _write(model / "splat_dataset_colmap" / "transforms.json")

    assert drive_store.scan_folders(output) == [capture, model]
    root, files = drive_store.files_to_save(output)
    names = {f.as_posix() for f in files}
    assert root == scans
    assert {"ur5_a/ur5_0001.png", "ur5_a/ur5_0001.json", "ur5_a_rescan/combined.json",
            "ur5_a_rescan/splat_dataset_colmap/colmap_masks/ur5_0001.png",
            "ur5_a_rescan/splat_dataset_colmap/transforms.json"} <= names
    assert {f"ur5_a_rescan/splat_output/{n}" for n in kept} <= names
    assert not {f"ur5_a_rescan/splat_output/{n}" for n in dropped} & names
    assert "ur5_a_rescan/splat_dataset_colmap/images/ur5_0001.png" not in names
    assert "ur5_a_rescan/splat_dataset_colmap/database.db" not in names


def test_progress_from_rclone_lines():
    line = "2026/11/08 15:00:01 INFO  :   12.000 MiB / 150.000 MiB, 8%, 2.000 MiB/s, ETA 1m9s"
    assert drive_store.progress_of(line) == 0.08
    assert drive_store.progress_of("NOTICE: something else") is None
