from types import SimpleNamespace

from camsvc import models


def test_bundled_models_copied_and_kept_up_to_date(tmp_path, monkeypatch):
    bundled, data = tmp_path / "opt", tmp_path / "data"
    bundled.mkdir()
    monkeypatch.setattr(models, "BUNDLED_DIR", bundled)
    for name in ("yolo11n.onnx", "yolo11n.xml", "yolo11n.bin"):
        (bundled / name).write_text("v1 " + name)
    cfg = SimpleNamespace(model={"path": str(data / "yolo11n.xml"), "fallback_path": str(data / "yolo11n.onnx")})
    models.ensure_models(cfg)
    assert sorted(p.name for p in data.iterdir()) == ["yolo11n.bin", "yolo11n.onnx", "yolo11n.xml"]
    (bundled / "yolo11n.onnx").write_text("v2")          # app update with a newer model
    models.ensure_models(cfg)
    assert (data / "yolo11n.onnx").read_text() == "v2"
    assert (data / "yolo11n.xml").read_text() == "v1 yolo11n.xml"
