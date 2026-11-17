"""Continuous recording (24/7), next to the event clips: to scroll back and see who touched the car.

- Only the main-stream video is copied into 5-minute MP4 segments (no decoding, no audio), so it
  costs next to no CPU. Fragmented MP4: the segment being written can already be played.
- <clips>/.continuous/<camera>/<YYYY-MM-DD>/<HHMMSS>.mp4, the time of its first frame (local).
  The dot keeps it out of the event list, the event retention and the NAS archive.
- Own retention: `storage.continuous_days` (0 = off) and `storage.continuous_gb`.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

DIR = ".continuous"
SEGMENT_SECONDS = 300
MAX_GAP = 60          # a segment is assumed to end here when the next one starts much later


class ContinuousWriter:
    """Fed every packet of the main stream by the recorder thread."""

    def __init__(self, root: Path, camera: str):
        self.dir = root / DIR / camera
        self.camera = camera
        self._writer = None
        self._started = 0.0

    def write(self, streams: dict, video_index: int, bp) -> None:
        from .recorder import ClipWriter
        if bp.stream_index != video_index:
            return                                   # video only: no audio transcoding all day
        if bp.is_keyframe and (self._writer is None or time.monotonic() - self._started >= SEGMENT_SECONDS):
            self.close()
            wall = datetime.now() - timedelta(seconds=max(0.0, time.monotonic() - bp.arrival))
            day = self.dir / wall.strftime("%Y-%m-%d")
            try:
                day.mkdir(parents=True, exist_ok=True)
                self._writer = ClipWriter(day / f"{wall.strftime('%H%M%S')}.mp4",
                                          {video_index: streams[video_index]}, video_index)
                self._started = time.monotonic()
            except Exception as exc:  # noqa: BLE001 - never disturb the event recording
                log.warning("[%s] continuous recording: cannot start a segment (%s)", self.camera, exc)
                self._writer = None
                return
        if self._writer is not None:
            try:
                self._writer.write(bp)
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s] continuous recording: write failed (%s); new segment", self.camera, exc)
                self.close()

    def close(self) -> None:
        if self._writer is not None:
            writer, self._writer = self._writer, None
            writer.close()


def segments(clips_dir: Path, camera: str, start: datetime, end: datetime) -> list[dict]:
    """The segments of a camera that overlap [start, end): start (ISO), seconds, clip path.
    The length is up to the next segment (or now for the last one), at most a segment."""
    base = clips_dir / DIR / camera
    files: list[tuple[datetime, Path]] = []
    d = start.date() - timedelta(days=1)
    while d <= end.date():
        for f in sorted((base / d.isoformat()).glob("*.mp4")):
            try:
                files.append((datetime.combine(d, datetime.strptime(f.stem, "%H%M%S").time()), f))
            except ValueError:
                continue
        d += timedelta(days=1)
    now = datetime.now()
    out = []
    for i, (t, f) in enumerate(files):
        nxt = files[i + 1][0] if i + 1 < len(files) else now
        seconds = min((nxt - t).total_seconds(), SEGMENT_SECONDS + MAX_GAP)
        if seconds <= 0 or t + timedelta(seconds=seconds) <= start or t >= end:
            continue
        out.append({"start": t.astimezone().isoformat(timespec="seconds"), "seconds": round(seconds, 1),
                    "clip": f"{DIR}/{camera}/{f.parent.name}/{f.name}"})
    return out


def cleanup(clips_dir: Path, days: float, max_gb: float) -> int:
    """Remove whole days older than `days`, then the oldest segments above `max_gb`."""
    root = clips_dir / DIR
    if not root.is_dir():
        return 0
    removed = 0
    oldest_kept = date.today() - timedelta(days=max(0.0, days))
    for cam in (p for p in root.iterdir() if p.is_dir()):
        for day in (p for p in cam.iterdir() if p.is_dir()):
            try:
                if date.fromisoformat(day.name) < oldest_kept:
                    removed += sum(1 for _ in day.glob("*.mp4"))
                    shutil.rmtree(day, ignore_errors=True)
            except ValueError:
                continue
    files = sorted(root.rglob("*.mp4"), key=lambda p: (p.parent.name, p.name))   # oldest first
    sizes = []
    for f in files:
        try:
            sizes.append((f, f.stat().st_size))
        except FileNotFoundError:
            continue
    total, limit = sum(s for _, s in sizes), max_gb * 1024 ** 3
    for f, size in sizes:
        if total <= limit:
            break
        f.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed


class ContinuousCleanup(threading.Thread):
    def __init__(self, clips_dir: Path, days: float, max_gb: float, interval: float = 600):
        super().__init__(name="continuous-cleanup", daemon=True)
        self.clips_dir, self.days, self.max_gb, self.interval = clips_dir, days, max_gb, interval
        self._stop = threading.Event()

    def run(self) -> None:
        while True:
            try:
                n = cleanup(self.clips_dir, self.days, self.max_gb)
                if n:
                    log.info("Continuous recording: removed %d old segment(s)", n)
            except Exception:  # noqa: BLE001
                log.exception("Cleaning up the continuous recording failed")
            if self._stop.wait(self.interval):
                return

    def stop(self) -> None:
        self._stop.set()
