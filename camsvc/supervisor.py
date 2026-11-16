"""Running as a Home Assistant app (add-on): ingress and the Supervisor API.

- Inside an app the Supervisor sets SUPERVISOR_TOKEN; outside an app everything here is a no-op.
- Ingress (the page in the HA sidebar) always comes from the Supervisor's proxy at 172.30.32.2,
  and HA has already logged the user in, so those requests need no passkey or setup code.
- The Mosquitto app shares its login through the Supervisor, so the wizard can fill it in.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

INGRESS_PROXY = "172.30.32.2"


def in_app() -> bool:
    return bool(os.environ.get("SUPERVISOR_TOKEN"))


def via_ingress(client_host: str | None) -> bool:
    return in_app() and client_host == INGRESS_PROXY


def _get(path: str) -> dict | None:
    req = urllib.request.Request(f"http://supervisor{path}",
                                 headers={"Authorization": f"Bearer {os.environ['SUPERVISOR_TOKEN']}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()).get("data")
    except (OSError, ValueError) as exc:
        log.info("Supervisor %s not available: %s", path, exc)
        return None


def mqtt_service() -> dict | None:
    """Login of the Mosquitto app: {host, port, username, password}, or None."""
    if not in_app():
        return None
    data = _get("/services/mqtt")
    if not data or not data.get("host"):
        return None
    return {"host": data["host"], "port": int(data.get("port") or 1883),
            "username": data.get("username") or "", "password": data.get("password") or ""}


def host_ip() -> str | None:
    """IPv4 address of the HA machine on the home network (for links in notifications)."""
    if not in_app():
        return None
    data = _get("/network/info") or {}
    for iface in data.get("interfaces") or []:
        if iface.get("primary"):
            for addr in (iface.get("ipv4") or {}).get("address") or []:
                return addr.split("/")[0]
    return None


def set_options(options: dict) -> None:
    """Store the app options (the Configuration tab); RuntimeError with the Supervisor's reason."""
    req = urllib.request.Request("http://supervisor/addons/self/options", method="POST",
                                 data=json.dumps({"options": options}).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['SUPERVISOR_TOKEN']}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except urllib.error.HTTPError as exc:
        try:
            reason = json.loads(exc.read()).get("message") or str(exc)
        except ValueError:
            reason = str(exc)
        raise RuntimeError(f"Configuration tab not updated: {reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"Configuration tab not updated: {exc}") from exc
