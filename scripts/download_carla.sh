#!/usr/bin/env bash
set -euo pipefail
# Official URL from https://github.com/carla-simulator/carla/releases/tag/0.9.15
sim_root="${1:?Usage: bash scripts/download_carla.sh /path/on/data/disk}"
mkdir -p "$sim_root"
if command -v aria2c >/dev/null; then
  aria2c --continue=true --max-connection-per-server=8 --split=8 --file-allocation=none \
    --dir="$sim_root" --out=CARLA_0.9.15.tar.gz \
    https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz
else
  curl -fL --retry 3 -C - https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz \
    -o "$sim_root/CARLA_0.9.15.tar.gz"
fi
# Record local digest; this is NOT an independently published upstream checksum.
sha256sum "$sim_root/CARLA_0.9.15.tar.gz" > "$sim_root/CARLA_0.9.15.local.sha256"
tar -tzf "$sim_root/CARLA_0.9.15.tar.gz" >/dev/null
mkdir "$sim_root/CARLA_0.9.15"
tar --no-same-owner -xzf "$sim_root/CARLA_0.9.15.tar.gz" -C "$sim_root/CARLA_0.9.15"
