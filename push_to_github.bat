@echo off
rem Double-click to commit and push SmogCast to the repo URL in github_repo.txt.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push_to_github.ps1"
pause
