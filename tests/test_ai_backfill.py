"""Describing the recordings of today afterwards (Manage → AI descriptions)."""

import json
import os
import time
from datetime import date
from types import SimpleNamespace

import numpy as np

from camsvc import ai_backfill
from camsvc.ai_backfill import Backfill, best_moment, todo_for_day
from camsvc.ai_describe import DEFAULTS, PROMPT_VERSION

DAY = date(2026, 11, 18)


def _event(root, cam, name, data, boxes=None, age=600):
    d = root / cam / DAY.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    js = d / f"{name}.json"
    js.write_text(json.dumps({"camera": cam, "start": "2026-11-18T10:00:00+01:00", **data}))
    (d / f"{name}.mp4").write_bytes(b"x")
    if boxes is not None:
        (d / f"{name}.boxes").write_text(json.dumps({"version": 2, "frames": boxes}))
    old = time.time() - age
    os.utime(js, (old, old))
    return js


def test_todo_only_undescribed_settled_people_and_vehicles(tmp_path):
    a = _event(tmp_path, "voor", "100000_person", {"label": "person", "labels": ["person"]})
    _event(tmp_path, "voor", "101000_person", {"label": "person", "ai": {"status": "ok"}})
    b = _event(tmp_path, "car", "110000_vehicle", {"label": "vehicle", "ai": {"status": "expired"}})
    _event(tmp_path, "car", "111000_animal", {"label": "animal"})
    _event(tmp_path, "car", "112000_person", {"label": "person"}, age=10)      # still being written
    (tmp_path / "voor" / DAY.isoformat() / "100000_person.face1.json").write_text("{}")
    todo = todo_for_day(tmp_path, ["voor", "car", "achter"], DAY, DEFAULTS["labels"])
    assert todo == [b, a]                     # newest first


def test_best_moment_prefers_moving_box_near_detection(tmp_path):
    js = _event(tmp_path, "voor", "100000_person", {"label": "person"}, boxes=[
        [2.0, [["person", "person", 0.9, 0.1, 0.1, 0.2, 0.3, 0]]],
        [10.5, [["car", "vehicle", 0.9, 0.0, 0.0, 0.9, 0.9, 1],          # parked: not this one
                ["person", "person", 0.8, 0.4, 0.2, 0.6, 0.9, 0]]],
        [11.0, [["person", "person", 0.8, 0.4, 0.2, 0.5, 0.5, 0]]],
    ])
    assert best_moment(js, ["person", "vehicle"], 10.0) == (10.5, [0.4, 0.2, 0.6, 0.9])
    bare = _event(tmp_path, "voor", "100500_person", {"label": "person"})
    assert best_moment(bare, ["person"], 10.0) == (10.0, None)


class FakeClient:
    def __init__(self):
        self.calls = 0

    def describe(self, jpeg, gray, timeout):
        self.calls += 1
        return {"description": "A person with a parcel.", "objects": ["parcel"],
                "attributes": ["yellow jacket", "parcel"], "image_quality": "good",
                "uncertainties": [], "person_visible": True, "vehicle_visible": False,
                "animal_visible": False}


def test_backfill_describes_and_stores(tmp_path, monkeypatch):
    js = _event(tmp_path, "voor", "100000_person", {"label": "person", "labels": ["person"]},
                boxes=[[10.0, [["person", "person", 0.9, 0.2, 0.2, 0.4, 0.8, 0]]]])
    seen = []
    monkeypatch.setattr(ai_backfill, "frame_at",
                        lambda clip, at: seen.append(at) or np.full((360, 640, 3), 120, np.uint8))
    client = FakeClient()
    svc = SimpleNamespace(
        cfg=SimpleNamespace(clips_dir=tmp_path, cameras=[SimpleNamespace(name="voor", pre_seconds=10.0, mask=[])]),
        ai=SimpleNamespace(cfg=dict(DEFAULTS, model="m"), client=client, paused=False))
    bf = Backfill(svc)
    started = bf.start(DAY)
    assert started["total"] == 1 and started["running"]
    bf._thread.join(5)
    st = bf.status()
    assert st == {"running": False, "day": DAY.isoformat(), "total": 1, "done": 1, "failed": 0}
    data = json.loads(js.read_text())
    assert data["ai"]["status"] == "ok" and data["ai"]["prompt_version"] == PROMPT_VERSION
    assert "parcel" in data["ai"]["attributes"] and data["false_alarm"] is None
    assert seen == [10.0] and client.calls == 1
    assert bf.start(DAY)["total"] == 0        # nothing left: done is done


def test_backfill_stops_when_ai_switched_off(tmp_path, monkeypatch):
    for i in range(3):
        _event(tmp_path, "voor", f"10{i}000_person", {"label": "person"})
    monkeypatch.setattr(ai_backfill, "frame_at", lambda clip, at: np.zeros((90, 160, 3), np.uint8))
    svc = SimpleNamespace(
        cfg=SimpleNamespace(clips_dir=tmp_path, cameras=[SimpleNamespace(name="voor", pre_seconds=10.0, mask=[])]),
        ai=SimpleNamespace(cfg=dict(DEFAULTS), client=FakeClient(), paused=True))
    bf = Backfill(svc)
    bf.start(DAY)
    bf._thread.join(5)
    assert bf.status()["done"] == 0 and svc.ai.client.calls == 0
