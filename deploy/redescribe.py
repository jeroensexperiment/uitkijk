"""Describe old recordings again with the current AI prompt (PROMPT_VERSION).

One-off, preferably at night: one frame at a time, from the clip itself (the snapshot has
detection boxes). Only events with a successful description of an older version.

  sudo -u camsvc /opt/camera-service/.venv/bin/python /opt/camera-service/deploy/redescribe.py \
      --config /etc/camera-service/config.yaml [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camsvc.ai_backfill import describe_recording  # noqa: E402
from camsvc.ai_describe import PROMPT_VERSION, AiError, OllamaClient  # noqa: E402
from camsvc.config import load_config  # noqa: E402

log = logging.getLogger("redescribe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    cams = {c.name: c for c in cfg.cameras}
    client = OllamaClient(cfg.ai)
    roots = [cfg.clips_dir] + ([Path(cfg.archive["dir"])] if cfg.archive else [])
    todo = []
    for root in roots:
        for js in root.rglob("*.json"):
            try:
                data = json.loads(js.read_text())
            except (OSError, ValueError):
                continue
            ai = data.get("ai") or {}
            if ai.get("status") == "ok" and ai.get("prompt_version") != PROMPT_VERSION:
                todo.append(js)
    todo.sort(reverse=True)                      # newest first: those are searched most
    if args.limit:
        todo = todo[:args.limit]
    log.info("Describing %d recordings again (%s)", len(todo), PROMPT_VERSION)
    done = 0
    for js in todo:
        try:
            cam = cams.get(json.loads(js.read_text()).get("camera"))
            if not describe_recording(js, cfg.ai, client, cam.pre_seconds if cam else 10.0,
                                      cam.mask if cam else []):
                continue
            done += 1
            if done % 10 == 0:
                log.info("%d/%d done", done, len(todo))
        except FileNotFoundError:
            continue                             # archived or cleaned up in the meantime
        except (AiError, OSError, ValueError, av.FFmpegError) as exc:
            log.warning("%s skipped: %s", js.name, exc)
    log.info("Done: %d of %d described again", done, len(todo))


if __name__ == "__main__":
    main()
