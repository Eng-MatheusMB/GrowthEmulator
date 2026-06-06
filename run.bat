@echo off
REM ──────────────────────────────────────────────────────────────
REM  BioGrowth v1.1 — Script de inicialização (Windows)
REM  Uso: duplo-clique em run.bat  OU  execute no terminal
REM ──────────────────────────────────────────────────────────────

echo.
echo   [*]  BioGrowth v1.1 — Microbial Growth Simulator
echo   ──────────────────────────────────────────────────
echo.

REM ── 1. Python check ─────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo   [X]  Python nao encontrado.
    echo        Baixe em: https://python.org/downloads
    echo        Marque "Add Python to PATH" na instalacao.
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo   [OK] %PY_VER% detectado

REM ── 2. Virtual environment ──────────────────────────────────
if not exist ".venv\" (
    echo   [*]  Criando ambiente virtual...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo   [OK] Ambiente virtual ativo

REM ── 3. Dependencies ─────────────────────────────────────────
echo   [*]  Verificando dependencias...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo   [OK] Dependencias instaladas

REM ── 4. Launch ────────────────────────────────────────────────
echo.
echo   [>>] Iniciando BioGrowth em http://localhost:8501
echo        Pressione Ctrl+C para encerrar
echo.
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
