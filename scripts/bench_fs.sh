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

  printf "%-20s escribir: %5d ms   leer: %5d ms   borrar: %5d ms\n" "$label" "$write" "$read" "$del"
}

SCRATCH="/mnt/c/.../fs-bench"
mkdir -p "$SCRATCH"

echo "Prueba: 300 ficheros pequeños"
echo
bench "$HOME" "WSL nativo"
bench "$SCRATCH" "Windows sin OneDrive"
bench "/mnt/c/.../Amadeus_FlightsDelay" "Windows con OneDrive"
rmdir "$SCRATCH" 2>/dev/null || true
