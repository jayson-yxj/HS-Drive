#!/usr/bin/env bash
set -euo pipefail
printf '\n== HS-Drive M0 server inventory ==\n'
date -Is || true
uname -a || true
if command -v lsb_release >/dev/null; then lsb_release -a; else cat /etc/os-release || true; fi
printf '\n== GPU ==\n'
if command -v nvidia-smi >/dev/null; then nvidia-smi; else echo 'nvidia-smi missing'; fi
printf '\n== Disk ==\n'
df -h
printf '\n== Memory ==\n'
free -h || true
printf '\n== Python and Git ==\n'
python3 --version || true
git --version || true
printf '\n== CARLA port (2000) ==\n'
if command -v ss >/dev/null; then ss -lnt | grep -E ':2000\b' || true; fi
printf '\nDone. Inventory only; no packages were installed.\n'
printf '\n== Container resource limits (not host totals) ==\n'
for resource in /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max; do
  if [ -r "$resource" ]; then printf '%s: ' "$resource"; cat "$resource"; fi
done
printf '\n== Vulkan ICD inventory ==\n'
ls /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null || true
