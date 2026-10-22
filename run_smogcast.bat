@echo off
rem Double-click to set up, download real data, train, forecast and push SmogCast.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_smogcast.ps1"
