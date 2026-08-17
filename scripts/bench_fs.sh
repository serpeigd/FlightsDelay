#!/usr/bin/env bash
# Compares filesystem speed: WSL's own disk vs a Windows folder seen through /mnt/c.
# Many small files is the pattern that hurts most: git, caches, Spark shuffle output.
set -uo pipefail

bench() {
  local dir="$1" label="$2" n=300
  rm -rf "$dir/bench_tmp" 2>/dev/null
  mkdir -p "$dir/bench_tmp"

  local start end write read
  start=$(date +%s%N)
  for i in $(seq 1 $n); do echo "linea de prueba $i" > "$dir/bench_tmp/f$i.txt"; done
  end=$(date +%s%N)
  write=$(( (end - start) / 1000000 ))

  start=$(date +%s%N)
  cat "$dir/bench_tmp"/*.txt > /dev/null
  end=$(date +%s%N)
  read=$(( (end - start) / 1000000 ))

  start=$(date +%s%N)
  rm -rf "$dir/bench_tmp"
  end=$(date +%s%N)
  local del=$(( (end - start) / 1000000 ))

  printf "%-24s write: %5d ms   read: %5d ms   delete: %5d ms\n" "$label" "$write" "$read" "$del"
}

# Both Windows locations are taken from the environment rather than written in:
# they are machine-specific, and a repository is not the place for someone's
# home directory. WIN_PLAIN should be any Windows folder that is not synced;
# WIN_SYNCED, one that is (this repo's own checkout does the job).
WIN_PLAIN="${WIN_PLAIN:-/mnt/c/Windows/Temp/fs-bench}"
WIN_SYNCED="${WIN_SYNCED:-$(cd "$(dirname "$0")/.." && pwd)}"

mkdir -p "$WIN_PLAIN"

echo "Test: 300 small files"
echo
bench "$HOME" "WSL native"
bench "$WIN_PLAIN" "Windows, not synced"
bench "$WIN_SYNCED" "Windows, synced folder"
rmdir "$WIN_PLAIN" 2>/dev/null || true
