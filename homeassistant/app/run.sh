#!/bin/sh
# Start go2rtc and the service. Runs as root: the Coral (USB) and /dev/dri need no extra groups.
mkdir -p /data/state /data/models /data/go2rtc "$CAMSVC_CLIPS"

# go2rtc for the live view; the setup wizard writes /data/go2rtc/go2rtc.yaml
(cd /data/go2rtc && while true; do go2rtc -config /data/go2rtc/go2rtc.yaml; sleep 5; done) &

# After saving the wizard the service stops with code 3 to start again in normal mode
# (in Docker the restart policy does that); other codes go to the HA watchdog.
while true; do
  python -m camsvc
  rc=$?
  [ "$rc" -eq 3 ] || exit "$rc"
  echo "Restarting with the new configuration"
done
