"""Describe recordings that have no AI description yet (e.g. "all of today"), on request.

Started from the web page (Manage → AI descriptions). One recording at a time, in a background
thread, from a frame of the clip itself; stops when the AI is switched off. Also used by
deploy/redescribe.py to describe old recordings again with a newer prompt.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

SETTLE_SECONDS = 120     # a recording that changed this recently may still be in progress


def frame_at(clip: Path, seconds: float):
    """BGR frame at `seconds` into the clip (or the last one if the clip is shorter)."""
    import av
    with av.open(str(clip)) as box:
        stream = box.streams.video[0]
        last = None
        for frame in box.decode(stream):
            last = frame
            if frame.time is not None and frame.time >= seconds:
                break
        return last.to_ndarray(format="bgr24") if last is not None else None


def best_moment(js: Path, labels: list[str], default: float) -> tuple[float, list[float] | None]:
    """When and where in the clip the AI should look: the biggest moving person/vehicle from
    the saved detection boxes (<clip>.boxes), closest to the moment of detection. Without boxes:
    the end of the pre-recording, whole frame."""
    try:
        frames = json.loads(js.with_suffix(".boxes").read_text()).get("frames") or []
    except (OSError, ValueError, AttributeError):
        return default, None
    best = None
    for t, boxes in frames:
        for b in boxes:
            if len(b) < 8 or b[1] not in labels or b[7]:
                continue                          # other kind, or parked
            area = (b[5] - b[3]) * (b[6] - b[4])
            key = (abs(t - default) > 3, -area)   # around the detection first, then the biggest
            if best is None or key < best[0]:
                best = (key, float(t), [float(v) for v in b[3:7]])
    return (best[1], best[2]) if best else (default, None)


def needs_description(data: dict, labels: list[str]) -> bool:
    """No successful description yet, and something the AI describes (person, vehicle...)."""
    if (data.get("ai") or {}).get("status") == "ok":
        return False
    found = set(data.get("labels") or [data.get("label")])
    return bool(found & set(labels))


def todo_for_day(clips_dir: Path, cameras: list[str], day: date, labels: list[str],
                 now: float | None = None) -> list[Path]:
    """Event files of `day` without a description, newest first."""
    now = time.time() if now is None else now
    todo = []
    for cam in cameras:
        folder = clips_dir / cam / day.isoformat()
        if not folder.is_dir():
            continue
        for js in folder.glob("*.json"):
            if js.name.count(".") > 1 or now - js.stat().st_mtime < SETTLE_SECONDS:
                continue                          # side files (x.face1.json) / still being written
            try:
                data = json.loads(js.read_text())
            except (OSError, ValueError):
                continue
            if needs_description(data, labels) and js.with_suffix(".mp4").exists():
                todo.append(js)
    return sorted(todo, key=lambda p: p.name, reverse=True)


def describe_recording(js: Path, ai_cfg: dict, client, pre_seconds: float, mask: list) -> bool:
    """Describe one recording and store the result in its event file. True when done."""
    from .ai_describe import PROMPT_VERSION, crop_around, false_alarm, prepare_image
    from .camera import AI_CROP_MARGIN, AI_CROP_MIN
    from .motion import build_mask
    at, frac = best_moment(js, list(ai_cfg["labels"]), pre_seconds)
    frame = frame_at(js.with_suffix(".mp4"), max(0.0, at))
    if frame is None:
        return False
    h, w = frame.shape[:2]
    if mask:                                       # privacy mask first, on the whole frame
        frame[build_mask(mask, w, h) == 0] = 0
    if frac:
        x1, y1, x2, y2 = frac
        box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        frame = crop_around(frame, box, (h, w), margin=AI_CROP_MARGIN, min_side=int(AI_CROP_MIN * h))
    jpeg, gray = prepare_image(frame, [], int(ai_cfg["image_max_edge"]),
                               int(ai_cfg["image_max_bytes"]), int(ai_cfg["image_max_pixels"]))
    t0 = time.perf_counter()
    result = client.describe(jpeg, gray, float(ai_cfg["request_timeout"]))
    data = json.loads(js.read_text())              # re-read: the page may have changed it meanwhile
    old = data.get("ai") or {}
    data["ai"] = {"status": "ok", "duration_ms": int((time.perf_counter() - t0) * 1000), **result,
                  "model": ai_cfg["model"], "prompt_version": PROMPT_VERSION,
                  "captured": old.get("captured") or data.get("start"),
                  "processed": datetime.now().astimezone().isoformat(timespec="seconds")}
    if data.get("false_alarm") in (None, "ai"):    # a verdict of your own ('nothing') stays
        data["false_alarm"] = "ai" if false_alarm(data.get("labels") or [data.get("label")], data["ai"]) else None
    tmp = js.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(js)
    return True


class Backfill:
    """At most one run at a time; status() for the web page."""

    def __init__(self, svc):
        self.svc = svc
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.state = {"running": False, "day": None, "total": 0, "done": 0, "failed": 0}

    def status(self) -> dict:
        with self._lock:
            return dict(self.state)

    def start(self, day: date) -> dict:
        ai = self.svc.ai
        with self._lock:
            if self.state["running"]:
                return dict(self.state)
            cams = {c.name: c for c in self.svc.cfg.cameras}
            todo = todo_for_day(self.svc.cfg.clips_dir, list(cams), day, ai.cfg["labels"])
            self.state = {"running": bool(todo), "day": day.isoformat(), "total": len(todo),
                          "done": 0, "failed": 0}
            if todo:
                self._cancel.clear()
                self._thread = threading.Thread(target=self._run, args=(todo, cams),
                                                name="ai-backfill", daemon=True)
                self._thread.start()
            return dict(self.state)

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self, todo: list[Path], cams: dict) -> None:
        from .ai_describe import AiError
        ai = self.svc.ai
        log.info("AI: describing %d recordings of %s", len(todo), self.state["day"])
        try:
            for js in todo:
                if self._cancel.is_set() or ai.paused:
                    log.info("AI: describing recordings stopped")
                    break
                cam = cams.get(js.parent.parent.name)
                ok = False
                try:
                    ok = describe_recording(js, ai.cfg, ai.client, cam.pre_seconds if cam else 10.0,
                                            cam.mask if cam else [])
                except FileNotFoundError:
                    pass                           # deleted or archived meanwhile
                except AiError as exc:
                    log.warning("AI: %s not described (%s)", js.name, exc.category)
                    if exc.category == "model_missing":
                        break
                    if exc.transient:             # AI PC busy or briefly away: give it a moment
                        self._cancel.wait(10)
                except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the rest
                    log.warning("AI: %s not described: %s", js.name, exc)
                with self._lock:
                    self.state["done" if ok else "failed"] += 1
        finally:
            with self._lock:
                self.state["running"] = False
            log.info("AI: describing recordings finished (%d done, %d not)",
                     self.state["done"], self.state["failed"])
