#!/bin/sh
# Start go2rtc and the service. Runs as root: the Coral (USB) and /dev/dri need no extra groups.
mkdir -p /data/state /data/models /config/go2rtc "$CAMSVC_CLIPS"

# config.yaml used to be in /data (not visible in HA); move it once to /config (= /addon_configs)
if [ ! -f /config/config.yaml ] && [ -f /data/config.yaml ]; then
  mv /data/config.yaml /config/config.yaml
  [ -f /data/go2rtc/go2rtc.yaml ] && mv /data/go2rtc/go2rtc.yaml /config/go2rtc/go2rtc.yaml
  echo "Configuration moved to /addon_configs (config.yaml)"
fi

# go2rtc for the live view; the setup wizard writes /config/go2rtc/go2rtc.yaml
(cd /config/go2rtc && while true; do go2rtc -config /config/go2rtc/go2rtc.yaml; sleep 5; done) &

# After saving the wizard or the configuration editor the service stops with code 3 to start
# again with the new configuration (in Docker the restart policy does that); other codes go to
# the HA watchdog.
while true; do
  python -m camsvc
  rc=$?
  [ "$rc" -eq 3 ] || exit "$rc"
  echo "Restarting with the new configuration"
done
