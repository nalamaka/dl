@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0one_click_smoke.ps1"
endlocal
