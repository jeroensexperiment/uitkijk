# Uitkijk as a Home Assistant app

The same service as the Docker install, run by Home Assistant (OS or Supervised, on an Intel/AMD
machine). What is different:

- **Sidebar**: the web page opens under *Uitkijk* in the HA sidebar, behind the HA login (also via
  the HA app on your phone, at home and away). No setup code and no passkeys needed there.
- **MQTT**: the setup wizard fills in the login of the Mosquitto app by itself.
- **Recordings** are in `/media/uitkijk`, so you also see them in the HA media browser. They are not
  part of the app backup (they can be large); settings, faces and plates in `/data` are.
- **Google Coral (USB)**: driver and model are included. Plug it in before starting the app; the log
  then says `Object model ... via TFLite on Edge TPU`. Without a Coral the CPU is used.
- **go2rtc** for the live view runs inside the app.

## Install

1. Settings → Apps (Add-ons) → App store → ⋮ → Repositories → add
   `https://github.com/lexi1986/uitkijk`.
2. Install **Uitkijk**. The first build takes a while (15–30 minutes): the object model is converted
   for the CPU and for the Coral.
3. Start it and open **Uitkijk** in the sidebar: the setup wizard appears.

The web page is also on `http://<ip-of-HA>:8080` on your home network, as before. Notifications from
the blueprints link there, so they work at home; for links that work away from home too, use your
own domain with HTTPS (see the README).

## Ports

The app uses the host network: 8080 (web page), 1984, 8554 and 8555 (go2rtc). They must be free;
Frigate uses the same go2rtc ports, so do not run both at once.
