#!/bin/sh
# Start go2rtc and the service. Runs as root: the Coral (USB) and /dev/dri need no extra groups.
mkdir -p /data/state /data/models /config/go2rtc "$CAMSVC_CLIPS"

# config.yaml used to be in /data (not visible in HA); move it once to /config (= /addon_configs)
if [ ! -f /config/config.yaml ] && [ -f /data/config.yaml ]; then
  mv /data/config.yaml /config/config.yaml
  [ -f /data/go2rtc/go2rtc.yaml ] && mv /data/go2rtc/go2rtc.yaml /config/go2rtc/go2rtc.yaml
  echo "Configuration moved to /addon_configs (config.yaml)"
fi

# the Configuration tab into config.yaml and go2rtc.yaml before go2rtc reads them
python -m camsvc.ha_options

# Stopping the app (update, restart): pass SIGTERM on, so the service finishes the clip that is
# being recorded and exits cleanly. sh only runs a trap between commands, so the service runs in
# the background and the script waits for it.
STOP=0
PID=""
G2R=""
stop() {
  STOP=1
  [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
  [ -n "$G2R" ] && kill -TERM "$G2R" 2>/dev/null
}
trap stop TERM INT

# go2rtc for the live view (streams from the Configuration tab or the setup wizard)
(cd /config/go2rtc && while true; do go2rtc -config /config/go2rtc/go2rtc.yaml & G=$!; trap 'kill -TERM $G 2>/dev/null; exit 0' TERM; wait $G; sleep 5; done) &
G2R=$!

# After saving the wizard or the configuration editor the service stops with code 3 to start
# again with the new configuration (in Docker the restart policy does that); other codes go to
# the HA watchdog.
while [ "$STOP" -eq 0 ]; do
  python -m camsvc &
  PID=$!
  wait "$PID"
  rc=$?
  # wait returns early when the trap ran: wait again until the service has really stopped
  while kill -0 "$PID" 2>/dev/null; do wait "$PID"; rc=$?; done
  PID=""
  [ "$STOP" -eq 1 ] && break
  if [ "$rc" -ne 3 ]; then
    stop
    wait
    exit "$rc"
  fi
  echo "Restarting with the new configuration"
done
wait
exit 0
