# Commit everything (code, trained model, outputs) and push to the repo in github_repo.txt.
$ErrorActionPreference = "Continue"     # git writes progress to stderr; don't treat it as fatal
Set-Location -Path $PSScriptRoot
$log = Join-Path $PSScriptRoot "run_log.txt"
$url = (Get-Content "github_repo.txt" -Raw).Trim()

# Find git: PATH first, then the copy bundled with GitHub Desktop
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
if (-not $git) {
    $cand = Get-ChildItem "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
    if ($cand) { $git = $cand.FullName }
}
if (-not $git) { ">>> Git not found. Install it with: winget install Git.Git" | Tee-Object -FilePath $log -Append; exit 1 }
">>> Using git at $git" | Tee-Object -FilePath $log -Append

if (Test-Path "workflows-to-move") {
    New-Item -ItemType Directory -Force ".github\workflows" | Out-Null
    Move-Item "workflows-to-move\*.yml" ".github\workflows\" -Force
    Remove-Item "workflows-to-move" -Recurse -Force
}
$env:GIT_TERMINAL_PROMPT = "0"          # fail instead of hanging on a hidden password prompt
if (-not (Test-Path ".git")) {
    & $git init -b main 2>&1 | Tee-Object -FilePath $log -Append
    & $git remote add origin $url
    # if the GitHub repo already has commits, build on top of them instead of clashing
    & $git fetch origin main 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -eq 0) { & $git reset --soft FETCH_HEAD 2>&1 | Tee-Object -FilePath $log -Append }
}
& $git config user.name "Jayagna Patel"
& $git config user.email "jayagnapatel15@gmail.com"
& $git add -A
$msg = @"
SmogCast: trained model, forecasts and data-pipeline fixes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YRKFSQZtGpQSbqtH92haBe
"@
& $git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { & $git commit -m $msg 2>&1 | Tee-Object -FilePath $log -Append }
& $git remote set-url origin $url 2>$null
Write-Host "A GitHub sign-in window may open: approve it to let Git push." -ForegroundColor Yellow
& $git push -u origin main 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -eq 0) { ">>> PUSHED to $url" | Tee-Object -FilePath $log -Append }
else { ">>> PUSH FAILED (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append }
