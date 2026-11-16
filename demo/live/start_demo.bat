@echo off
rem BEAT Audit Live: double-click to start the stage demo (server and full-screen page).
rem This file is ASCII only; the console is switched to UTF-8 for the Korean messages.
setlocal
chcp 65001 >nul
title BEAT Audit Live
cd /d "%~dp0..\.."
if exist ".venv\Scripts\python.exe" goto run
echo.
echo  [BEAT Audit Live] Python virtual environment not found:
echo    %CD%\.venv\Scripts\python.exe
echo  Create it in the project folder first: python -m venv .venv, then install requirements.txt.
echo.
pause
exit /b 1

:run
".venv\Scripts\python.exe" -X utf8 "demo\live\launch.py"
if errorlevel 1 pause
endlocal
