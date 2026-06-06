#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# BioGrowth v1.1 — Script de inicialização (Unix/macOS/Linux)
# Uso: bash run.sh
# ──────────────────────────────────────────────────────────────
set -e

VENV_DIR=".venv"
APP="app.py"
PORT=8501

echo ""
echo "  🧫  BioGrowth v1.1 — Microbial Growth Simulator"
echo "  ─────────────────────────────────────────────────"

# ── 1. Python check ──────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "  ❌  Python 3 não encontrado. Instale em https://python.org"
  exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  ✔  Python $PY_VER detectado"

# ── 2. Virtual environment ───────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "  ⚙  Criando ambiente virtual em $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "  ✔  Ambiente virtual ativo"

# ── 3. Install / upgrade dependencies ───────────────────────
echo "  📦  Verificando dependências ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  ✔  Dependências instaladas"

# ── 4. Launch ────────────────────────────────────────────────
echo ""
echo "  🚀  Iniciando BioGrowth em http://localhost:$PORT"
echo "  (pressione Ctrl+C para encerrar)"
echo ""
streamlit run "$APP" \
  --server.port "$PORT" \
  --server.headless false \
  --browser.gatherUsageStats false
