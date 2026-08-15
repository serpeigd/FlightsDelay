#!/usr/bin/env bash
# Downloads 24 months of US BTS On-Time Performance data into the WSL data lake.
# 2023 is the training year, 2024 the held-out year: a temporal split, never a random one.
set -uo pipefail

DEST="$HOME/data/flight-delay/raw"
BASE="https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
mkdir -p "$DEST"

ok=0
fail=0
for year in 2023 2024; do
  for month in $(seq 1 12); do
    name="${year}_${month}.zip"
    target="$DEST/$name"
    if [ -s "$target" ]; then
      echo "SKIP  $name (ya está)"
      ok=$((ok + 1))
      continue
    fi
    if curl -sSL --fail --max-time 300 -o "$target.part" "${BASE}_${year}_${month}.zip"; then
      mv "$target.part" "$target"
      echo "OK    $name  $(du -h "$target" | cut -f1)"
      ok=$((ok + 1))
    else
      rm -f "$target.part"
      echo "FALLO $name"
      fail=$((fail + 1))
    fi
  done
done

echo
echo "Descargados: $ok   Fallidos: $fail"
du -sh "$DEST" | awk '{print "Tamaño total: " $1}'
