"""The Configuration tab of the Home Assistant app <-> config.yaml.

- The tab (the app options in /data/options.json) holds the cameras and the main settings. At
  every start they are written into config.yaml; everything else in config.yaml stays (masks,
  zones, per-camera overrides, AI, passkeys).
- The setup wizard and the configuration editor write config.yaml; afterwards the tab is updated
  from it through the Supervisor, so both always show the same cameras.
- Outside the HA app (Docker, Proxmox) nothing here runs.

Run once before go2rtc starts: python -m camsvc.ha_options
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import yaml

from . import supervisor
from .setup import BRANDS, camera_urls, go2rtc_yaml, push_go2rtc, slug

log = logging.getLogger(__name__)

OPTIONS_FILE = Path(os.environ.get("CAMSVC_OPTIONS", "/data/options.json"))
HEADER = ("# Cameras and the main settings come from the Configuration tab of the Uitkijk app and are\n"
          "# written here at every start; everything else here is kept. All options: config.example.yaml.\n")
DEFAULT_GO2RTC = "http://localhost:1984"


def load_options() -> dict | None:
    try:
        return json.loads(OPTIONS_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return None


def options_to_config(opts: dict, existing: dict, mqtt: dict | None, public_url: str | None) -> dict:
    """config.yaml (as a dict) from the tab, on top of the existing file. ValueError on bad input."""
    cfg = copy.deepcopy(existing or {})
    old = {c.get("name"): c for c in cfg.get("cameras") or []}
    cams, seen = [], set()
    for o in opts.get("cameras") or []:
        friendly = str(o.get("name") or "").strip()
        if not friendly:
            raise ValueError("Every camera needs a name")
        name = slug(friendly)
        if name in seen:
            raise ValueError(f"The name '{friendly}' is used twice")
        seen.add(name)
        main, sub = camera_urls({**o, "name": friendly, "sub": o.get("sub_stream")})
        cam = copy.deepcopy(old.get(name) or {})     # keeps overrides set in config.yaml
        cam.update(name=name, friendly_name=friendly, main_url=main)
        cam.pop("onvif", None)
        if sub:
            cam["sub_url"] = sub
        else:
            cam.pop("sub_url", None)
        live_mode = BRANDS.get(o.get("brand") or "", {}).get("live_mode")
        if live_mode:
            cam.setdefault("live_mode", live_mode)
        for feature in ("faces", "plates"):
            if feature in o:
                cam[feature] = {**(cam.get(feature) or {}), "enabled": bool(o[feature])}
        cams.append(cam)
    # cameras found via ONVIF cannot be entered in the tab; keep them as they are
    cams += [c for c in cfg.get("cameras") or [] if not c.get("main_url") and c.get("name") not in seen]
    cfg["cameras"] = cams

    st = opts.get("storage") or {}
    storage = cfg.setdefault("storage", {})
    storage["clips_dir"] = os.environ.get("CAMSVC_CLIPS", storage.get("clips_dir", "/clips"))
    storage["state_dir"] = os.environ.get("CAMSVC_STATE", storage.get("state_dir", "/data/state"))
    storage["retention_days"] = float(st.get("retention_days", storage.get("retention_days", 7)))
    storage["max_gb"] = float(st.get("max_gb", storage.get("max_gb", 50)))
    person = float(st.get("person_days", 30))
    storage["retention_days_by_label"] = {**(storage.get("retention_days_by_label") or {}),
                                          "person": person, "doorbell": person}
    cfg["log_level"] = str(opts.get("log_level") or "info").upper()

    if mqtt:
        cfg["mqtt"] = {**(cfg.get("mqtt") or {}), **mqtt}
        cfg.setdefault("summary_time", "22:00")
    else:
        cfg.setdefault("mqtt", {"host": "", "port": 1883})
    api = cfg.setdefault("api", {})
    api.setdefault("host", "0.0.0.0")
    api["port"] = int(os.environ.get("CAMSVC_PORT", api.get("port", 8080)))
    if public_url and not cfg.get("auth"):
        api.setdefault("public_url", public_url)
    cfg.setdefault("model", {"path": os.environ.get("CAMSVC_MODEL", "/data/models/yolo11n.onnx"),
                             "input_size": 320})
    if opts.get("live_view", True):
        cfg["live"] = {**(cfg.get("live") or {}),
                       "go2rtc_url": (cfg.get("live") or {}).get("go2rtc_url") or DEFAULT_GO2RTC}
    else:
        cfg.pop("live", None)
    return cfg


def config_to_options(cfg: dict, current: dict | None) -> dict:
    """The tab from config.yaml. A camera entered with brand and IP stays that way when its
    addresses did not change; otherwise it becomes 'other' with the full RTSP addresses."""
    current = current or {}
    before = {}
    for o in current.get("cameras") or []:
        try:
            before[slug(str(o.get("name") or ""))] = (o, camera_urls({**o, "sub": o.get("sub_stream")}))
        except (ValueError, TypeError, KeyError):
            pass
    cams = []
    for c in cfg.get("cameras") or []:
        if not c.get("main_url"):
            continue        # ONVIF: stays in config.yaml only
        prev, urls = before.get(c["name"], (None, None))
        if prev and urls == (c["main_url"], c.get("sub_url")):
            o = dict(prev)
            o["name"] = c.get("friendly_name") or prev["name"]
        else:
            o = {"name": c.get("friendly_name") or c["name"], "brand": "other", "main_url": c["main_url"]}
            if c.get("sub_url"):
                o["sub_url"] = c["sub_url"]
        for feature in ("faces", "plates"):
            if isinstance(c.get(feature), dict) and "enabled" in c[feature]:
                o[feature] = bool(c[feature]["enabled"])
        cams.append(o)
    storage = cfg.get("storage") or {}
    labels = storage.get("retention_days_by_label") or {}
    level = str(cfg.get("log_level") or "info").lower()
    return {
        "cameras": cams,
        "live_view": bool(cfg.get("live")),
        "storage": {"retention_days": max(1, round(float(storage.get("retention_days", 7)))),
                    "person_days": max(1, round(float(labels.get("person", 30)))),
                    "max_gb": max(1, round(float(storage.get("max_gb", 50))))},
        "log_level": level if level in ("debug", "info", "warning", "error") else "info",
    }


def save_to_tab(cfg: dict) -> None:
    """After the wizard or the editor: show the new cameras in the Configuration tab. Raises
    RuntimeError when the Supervisor refuses: then config.yaml must not change either, or the
    next start would overwrite it with the old tab."""
    if not supervisor.in_app():
        return
    opts = config_to_options(cfg, load_options())
    supervisor.set_options(opts)
    # the service restarts inside the app, so the Supervisor does not write options.json again
    OPTIONS_FILE.write_text(json.dumps(opts))


def write_config(config_path: Path, cfg: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(".yaml.new")
    tmp.write_text(HEADER + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.chmod(0o600)
    if config_path.exists():
        config_path.replace(config_path.with_suffix(".yaml.previous"))
    tmp.replace(config_path)


def sync(config_path: Path) -> None:
    """At start: the tab into config.yaml, or (still empty tab) config.yaml into the tab."""
    if not supervisor.in_app():
        return
    opts = load_options()
    if opts is None:
        return
    try:
        existing = yaml.safe_load(config_path.read_text()) or {}
    except FileNotFoundError:
        existing = {}
    if not opts.get("cameras"):
        if existing.get("cameras"):
            try:
                save_to_tab(existing)   # set up with the wizard: show those cameras in the tab
            except RuntimeError as exc:
                log.warning("%s", exc)
        return                          # nothing yet: the setup wizard starts
    ip = supervisor.host_ip()
    try:
        cfg = options_to_config(opts, existing, supervisor.mqtt_service(),
                                f"http://{ip}:{os.environ.get('CAMSVC_PORT', '8080')}" if ip else None)
    except (ValueError, TypeError) as exc:
        log.error("Configuration tab not applied, the previous config.yaml stays: %s", exc)
        return
    if cfg != existing:
        write_config(config_path, cfg)
        log.warning("Configuration tab written to %s", config_path)
    if cfg.get("live"):
        g = config_path.parent / "go2rtc" / "go2rtc.yaml"
        g.parent.mkdir(parents=True, exist_ok=True)
        g.write_text(go2rtc_yaml(cfg))
        g.chmod(0o600)              # contains the camera passwords
        push_go2rtc(cfg["live"]["go2rtc_url"], cfg)


if __name__ == "__main__":
    from .logredact import setup as setup_logging
    setup_logging()
    sync(Path(os.environ.get("CAMSVC_CONFIG", "/config/config.yaml")))
