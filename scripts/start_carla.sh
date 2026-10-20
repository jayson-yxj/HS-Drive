#!/usr/bin/env bash
set -euo pipefail
carla_root="${1:?Usage: bash scripts/start_carla.sh /path/to/CARLA_0.9.15 [port] [Epic|Low]}"
carla_port="${2:-2000}"
carla_quality="${3:-Epic}"
case "$carla_quality" in Epic|Low) ;; *) echo "Quality must be Epic or Low" >&2; exit 2;; esac
if [ "$(id -u)" -eq 0 ]; then
  echo 'UE4 refuses root. Run this script as a regular user with GPU and package access.' >&2
  exit 1
fi
cd "$carla_root"
# In this container DISPLAY points at a remote desktop. UE4 crashed with it.
unset DISPLAY
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
exec ./CarlaUE4.sh -RenderOffScreen -quality-level="$carla_quality" -nosound -carla-rpc-port="$carla_port" -stdout -FullStdOutLogOutput
