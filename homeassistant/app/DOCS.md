# Uitkijk as a Home Assistant app

The same service as the Docker install, run by Home Assistant (OS or Supervised, on an Intel/AMD
machine). What is different:

- **Sidebar**: the web page opens under *Uitkijk* in the HA sidebar, behind the HA login (also via
  the HA app on your phone, at home and away). No setup code and no passkeys needed there.
- **MQTT**: the setup wizard fills in the login of the Mosquitto app by itself.
- **Recordings** are in `/media/uitkijk`, so you also see them in the HA media browser. They are not
  part of the app backup (they can be large); settings, faces and plates in `/data` are.
- **Detection on the Coral, the Intel iGPU or the CPU** (Configuration tab → *Detection on*):
  - `coral`: Google Coral USB with Google's SSDLite MobileDet (fast, a little less accurate than
    YOLO). Plug it in before starting the app; the log says `... via TFLite on Edge TPU`.
  - `gpu`: YOLO11n via OpenVINO on the Intel iGPU; the log says `... via OpenVINO on GPU`.
  - `cpu`: YOLO11n via ONNX Runtime.
  - `auto` (default): the Coral when it works, otherwise the iGPU, otherwise the CPU.
  Whatever does not work falls back to the CPU. The pill **CPU … · TPU/GPU …** at the top of the
  Uitkijk page shows the load; hover over it for images per second and milliseconds per image.
- **go2rtc** for the live view runs inside the app.
- **Configuration tab** of the app: add, change or remove cameras (brand, IP, login, Dahua sub
  stream, faces, plates), storage and live view. Save and restart the app; masks and zones from the
  Uitkijk page stay.
- **Everything else** in Uitkijk under Manage → Configuration: the whole `config.yaml`, checked before
  saving; the app then restarts by itself and the Configuration tab follows. The file is in
  `/addon_configs/<id>_uitkijk/config.yaml` (File editor or Studio Code Server can open it too;
  restart the app after editing it there).

## Install

1. Settings → Apps (Add-ons) → App store → ⋮ → Repositories → add
   `https://github.com/jeroensexperiment/uitkijk`.
2. Install **Uitkijk**. The first build takes a few minutes; the object models (CPU and Coral) are
   downloaded ready-made from the release "models" of this repository.
3. Start it and open **Uitkijk** in the sidebar: the setup wizard appears.

The web page is also on `http://<ip-of-HA>:8080` on your home network, as before. Notifications from
the blueprints link there, so they work at home; for links that work away from home too, use your
own domain with HTTPS (see the README).

## Ports

The app uses the host network: 8080 (web page), 1984, 8554 and 8555 (go2rtc). They must be free;
Frigate uses the same go2rtc ports, so do not run both at once.
