"""HA app: the Configuration tab <-> config.yaml."""

import json

import pytest
import yaml

from camsvc import ha_options
from camsvc.config import load_config

TAB = {"cameras": [{"name": "Oprit", "brand": "dahua", "ip": "10.0.0.5", "username": "admin",
                    "password": "p@ss", "sub_stream": "2", "faces": True},
                   {"name": "Tuin", "brand": "other", "main_url": "rtsp://x/1"}],
       "live_view": True, "storage": {"retention_days": 5, "person_days": 20, "max_gb": 40},
       "log_level": "info"}


def test_tab_into_config_keeps_the_rest(tmp_path):
    existing = {"cameras": [{"name": "oprit", "main_url": "rtsp://old", "mask": [[[0, 0], [1, 1], [2, 0]]]},
                            {"name": "weg", "main_url": "rtsp://gone"}],
                "ai": {"enabled": False}}
    cfg = ha_options.options_to_config(TAB, existing, {"host": "core-mosquitto", "port": 1883,
                                                       "username": "u", "password": "p"}, "http://1.2.3.4:8080")
    oprit, tuin = cfg["cameras"]
    assert [c["name"] for c in cfg["cameras"]] == ["oprit", "tuin"]           # 'weg' left the tab
    assert oprit["mask"] and oprit["faces"] == {"enabled": True} and "admin:p%40ss@" in oprit["main_url"]
    assert oprit["sub_url"].endswith("subtype=2") and "sub_url" not in tuin
    assert cfg["ai"] == {"enabled": False} and cfg["mqtt"]["host"] == "core-mosquitto"
    assert cfg["storage"]["retention_days_by_label"]["person"] == 20 and cfg["live"]
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    assert len(load_config(tmp_path / "c.yaml").cameras) == 2


def test_config_back_into_tab():
    cfg = ha_options.options_to_config(TAB, {}, None, None)
    tab = ha_options.config_to_options(cfg, TAB)
    assert tab["cameras"][0] == TAB["cameras"][0]                 # brand form kept
    assert tab["storage"] == TAB["storage"] and tab["live_view"] is True
    cfg["cameras"][0]["main_url"] = "rtsp://changed/in/the/editor"
    changed = ha_options.config_to_options(cfg, TAB)["cameras"][0]
    assert changed["brand"] == "other" and changed["main_url"] == "rtsp://changed/in/the/editor"


def test_bad_tab_keeps_config(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        ha_options.options_to_config({"cameras": [{"name": "A", "brand": "other", "main_url": "rtsp://1"},
                                                  {"name": "a", "brand": "other", "main_url": "rtsp://2"}]},
                                     {}, None, None)
    opts, conf = tmp_path / "options.json", tmp_path / "config.yaml"
    opts.write_text(json.dumps({"cameras": [{"name": "Cam", "brand": "dahua"}]}))   # no IP
    conf.write_text("cameras:\n  - name: hek\n    main_url: rtsp://x/1\n")
    monkeypatch.setattr(ha_options, "OPTIONS_FILE", opts)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")
    monkeypatch.setattr("camsvc.supervisor._get", lambda path: None)
    ha_options.sync(conf)
    assert "hek" in conf.read_text()


def test_detector_choice(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("CAMSVC_MODEL", "/data/models/yolo11n.onnx")
    monkeypatch.setenv("CAMSVC_CORAL_MODEL", "/opt/m/coral.tflite")
    monkeypatch.setenv("CAMSVC_BUNDLED_MODELS", str(tmp_path))
    # no iGPU model in the image: gpu becomes the CPU
    m = ha_options.model_config("gpu", {})
    assert m["path"].endswith(".onnx") and m["coral_path"] == "" and "fallback_path" not in m
    (tmp_path / "yolo11n.xml").write_text("x")
    m = ha_options.model_config("gpu", {"input_size": 320})
    assert m["path"] == "/data/models/yolo11n.xml" and m["device"] == "GPU"
    assert m["fallback_path"] == "/data/models/yolo11n.onnx" and m["coral_path"] == ""
    m = ha_options.model_config("auto", {})
    assert m["coral_path"] == "/opt/m/coral.tflite" and m["path"].endswith(".xml")
    m = ha_options.model_config("coral", {})
    assert m["coral_path"] == "/opt/m/coral.tflite" and m["path"].endswith(".onnx")
    m = ha_options.model_config("cpu", m)
    assert m["coral_path"] == "" and m["path"].endswith(".onnx") and "fallback_path" not in m
    with pytest.raises(ValueError):
        ha_options.model_config("tpu", {})
    assert ha_options.config_to_options({}, {"detector": "gpu"})["detector"] == "gpu"
