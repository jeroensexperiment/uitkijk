"""24/7 recording (Rewind): segments, their lengths, cleanup and the writer's rotation."""

import time
from datetime import date, datetime, timedelta
from types import SimpleNamespace

from camsvc import continuous
from camsvc.continuous import ContinuousWriter, cleanup, segments


def _seg(root, cam, when: datetime, size=10):
    d = root / ".continuous" / cam / when.date().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{when:%H%M%S}.mp4"
    f.write_bytes(b"x" * size)
    return f


def test_segments_with_lengths_and_gaps(tmp_path):
    day = datetime(2026, 11, 17)
    for hhmm in ("10:00", "10:05", "10:10", "11:00"):   # 10:15-11:00 the camera was offline
        _seg(tmp_path, "oprit", day.replace(hour=int(hhmm[:2]), minute=int(hhmm[3:])))
    found = segments(tmp_path, "oprit", day.replace(hour=10, minute=3), day.replace(hour=11, minute=30))
    assert [s["clip"] for s in found] == [
        ".continuous/oprit/2026-11-17/100000.mp4", ".continuous/oprit/2026-11-17/100500.mp4",
        ".continuous/oprit/2026-11-17/101000.mp4", ".continuous/oprit/2026-11-17/110000.mp4"]
    assert [s["seconds"] for s in found[:3]] == [300, 300, 360]   # before a gap: at most a segment + 60 s
    assert found[0]["start"].startswith("2026-11-17T10:00:00")
    assert segments(tmp_path, "achter", day, day + timedelta(hours=1)) == []


def test_cleanup_by_days_and_size(tmp_path):
    today = datetime.combine(date.today(), datetime.min.time()).replace(hour=12)
    old = _seg(tmp_path, "oprit", today - timedelta(days=5))
    keep = [_seg(tmp_path, "oprit", today - timedelta(minutes=10 * i), size=600) for i in range(3)]
    removed = cleanup(tmp_path, days=2, max_gb=1000 / 1024 ** 3)   # 1800 bytes of recent ones, 1000 allowed
    assert not old.exists() and removed == 3
    assert [f.exists() for f in keep] == [True, False, False]   # the oldest go first
    # event recordings next to it are not touched
    ev = tmp_path / "oprit" / "2026-11-17" / "120000_person.mp4"
    ev.parent.mkdir(parents=True)
    ev.write_bytes(b"x")
    cleanup(tmp_path, days=0, max_gb=0.000001)
    assert ev.exists()


def test_writer_rotates_on_a_keyframe_after_five_minutes(tmp_path, monkeypatch):
    made = []

    class FakeWriter:
        def __init__(self, path, streams, video_index):
            self.path, self.packets, self.closed = path, [], False
            made.append(self)

        def write(self, bp):
            self.packets.append(bp)

        def close(self):
            self.closed = True
            return 1.0

    monkeypatch.setattr("camsvc.recorder.ClipWriter", FakeWriter)
    clock = [1000.0]
    monkeypatch.setattr(continuous.time, "monotonic", lambda: clock[0])
    w = ContinuousWriter(tmp_path, "oprit")
    pkt = lambda key, idx=0: SimpleNamespace(stream_index=idx, is_keyframe=key, arrival=clock[0])
    w.write({0: "video"}, 0, pkt(False))          # waits for the first keyframe
    assert made == []
    w.write({0: "video"}, 0, pkt(True))
    w.write({0: "video"}, 0, pkt(False, idx=1))   # audio: not in the 24/7 recording
    clock[0] += 200
    w.write({0: "video"}, 0, pkt(True))           # keyframe, but not yet 5 minutes
    clock[0] += 120
    w.write({0: "video"}, 0, pkt(False))
    w.write({0: "video"}, 0, pkt(True))           # past 5 minutes: new segment
    assert len(made) == 2 and made[0].closed and len(made[0].packets) == 3
    assert made[0].path.parent.parent == tmp_path / ".continuous" / "oprit"
    w.close()
    assert made[1].closed
