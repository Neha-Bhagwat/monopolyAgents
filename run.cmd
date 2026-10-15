@echo off
setlocal
cd /d "%~dp0"
set "MONOPOLY_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%MONOPOLY_PYTHON%" (
    "%MONOPOLY_PYTHON%" -B -m monopoly %*
) else (
    py -3 -B -m monopoly %*
)
