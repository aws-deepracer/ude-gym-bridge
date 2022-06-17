#!/usr/bin/env bash

# Manually start dbus
rm /run/dbus/pid
dbus-daemon --system

export XAUTHORITY=/root/.Xauthority
export DISPLAY=:0 # Select screen 0 by default.

n=0
until [ "$n" -ge 5 ]
do
  nohup xvfb-run -f $XAUTHORITY -l -n 0 -s ":0 -screen 0 1400x800x24" startxfce4 &
  if pgrep -f Xvfb &>/dev/null; then
    break
  fi
  sleep 2
  n=$((n+1))
done

# disable access control, so clients can connect from any host.
# - This is needed to be able to login with user account authentication.
xhost +

# Install Full Gym with Atari support
pip install "gym[all]==0.21.0"
pip install "gym[atari, accept-rom-license]"

exec /bin/bash
