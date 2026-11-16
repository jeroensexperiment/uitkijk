import time
from types import SimpleNamespace

from camsvc.health import Usage


def test_usage_detector_busy_and_rate():
    det = SimpleNamespace(device="Edge TPU", backend="TFLite", accelerated=True, busy_seconds=0.0, runs=0)
    u = Usage()
    first = u.sample(det)
    assert first["detector"]["device"] == "Edge TPU" and first["detector"]["busy"] is None
    time.sleep(0.6)
    det.busy_seconds, det.runs = 0.06, 6      # 6 runs of 10 ms
    s = u.sample(det)
    assert s["detector"]["ms"] == 10.0
    assert 5 <= s["detector"]["fps"] <= 10.5
    assert 5 <= s["detector"]["busy"] <= 10.5
    assert s["app_cpu"] is not None


def test_usage_without_detector():
    u = Usage()
    u.sample(None)
    time.sleep(0.6)
    assert u.sample(None)["detector"] is None


def test_usage_two_pages_share_the_window():
    det = SimpleNamespace(device="GPU", backend="OpenVINO", accelerated=True, busy_seconds=0.0, runs=0)
    u = Usage()
    u.sample(det)
    time.sleep(0.6)
    det.busy_seconds, det.runs = 0.1, 5
    a = u.sample(det)
    det.busy_seconds, det.runs = 0.2, 10
    assert u.sample(det) is a          # asked again right away: same numbers, no 0-second window
