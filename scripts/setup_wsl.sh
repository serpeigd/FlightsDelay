#!/usr/bin/env bash
# Prepares the WSL environment for the flight-delay project.
# Everything installs under $HOME, so no sudo and no password are ever needed.
set -euo pipefail

OPT="$HOME/.local/opt"
mkdir -p "$OPT"

echo "== JDK 17 (Temurin) =="
if [ -d "$OPT/jdk17" ]; then
  echo "   ya instalado, se omite"
else
  cd /tmp
  curl -sSL -o jdk17.tar.gz \
    'https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse'
  echo "   descargado: $(du -h jdk17.tar.gz | cut -f1)"
  rm -rf "$OPT/jdk17" && mkdir -p "$OPT/jdk17"
  tar -xzf jdk17.tar.gz -C "$OPT/jdk17" --strip-components=1
  rm -f jdk17.tar.gz
fi
export JAVA_HOME="$OPT/jdk17"
export PATH="$JAVA_HOME/bin:$PATH"
echo "   $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"

echo "== uv (gestor de Python, sin sudo) =="
if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
  echo "   ya instalado, se omite"
else
  curl -sSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
echo "   $(uv --version)"

echo "== Python 3.12 =="
uv python install 3.12 >/dev/null 2>&1 || true
echo "   $(uv python find 3.12 2>/dev/null || echo 'no encontrado')"

echo "== variables persistentes en ~/.bashrc =="
MARKER="# flight-delay project env"
if ! grep -q "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ""
    echo "$MARKER"
    echo "export JAVA_HOME=\"$OPT/jdk17\""
    echo 'export PATH="$JAVA_HOME/bin:$HOME/.local/bin:$PATH"'
  } >> "$HOME/.bashrc"
  echo "   añadidas"
else
  echo "   ya estaban"
fi

echo "== carpeta del lago de datos =="
mkdir -p "$HOME/data/flight-delay/raw"
echo "   $HOME/data/flight-delay"
df -h "$HOME" | tail -1 | awk '{print "   libre: " $4}'

echo
echo "LISTO."
