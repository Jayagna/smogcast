# One-click setup + real-data run for SmogCast (Windows PowerShell).
# Started by run_smogcast.bat. Everything is logged to run_log.txt.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$log = Join-Path $PSScriptRoot "run_log.txt"
"=== SmogCast run started $(Get-Date -Format s) ===" | Out-File $log -Encoding utf8

function Step($msg) { $line = "`n>>> $msg"; Write-Host $line -ForegroundColor Cyan; $line | Out-File $log -Append -Encoding utf8 }
function Run($exe, [string[]]$argv) {
    # run a native command, show output live AND append it to the log; stop on failure
    $cmd = "`"$exe`" " + ($argv -join " ")
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Step "FAILED: $cmd (exit $LASTEXITCODE)"; Read-Host "Press Enter to close"; exit 1 }
}

# ---------------------------------------------------------------- 0. tidy repo layout
if (Test-Path "workflows-to-move") {
    New-Item -ItemType Directory -Force ".github\workflows" | Out-Null
    Move-Item "workflows-to-move\*.yml" ".github\workflows\" -Force
    Remove-Item "workflows-to-move" -Recurse -Force
}

# ---------------------------------------------------------------- 1. python
Step "Finding Python"
$py = $null
foreach ($c in @("$env:USERPROFILE\anaconda3\python.exe", "py", "python")) {
    try { & $c -c "import sys; assert sys.version_info >= (3, 10)" 2>$null; if ($LASTEXITCODE -eq 0) { $py = $c; break } } catch {}
}
if (-not $py) { Step "No Python 3.10+ found. Install it from python.org and run again."; Read-Host "Press Enter"; exit 1 }
"Using $py" | Tee-Object -FilePath $log -Append

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Step "Creating virtual environment (.venv)"
    Run $py @("-m", "venv", ".venv")
}
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Step "Installing packages"
Run $vpy @("-m", "pip", "install", "--upgrade", "pip", "-q")
Run $vpy @("-m", "pip", "install", "-r", "requirements.txt", "pytest", "-q")

Step "Running tests"
Run $vpy @("-m", "pytest", "-q")

# ---------------------------------------------------------------- 2. keys
if (-not (Test-Path ".env")) {
    Step "API keys (saved only to .env on this PC, which git ignores)"
    $oa = Read-Host "Paste your OpenAQ API key"
    $fk = Read-Host "Paste your NASA FIRMS MAP_KEY"
    "OPENAQ_API_KEY=$($oa.Trim())`nFIRMS_MAP_KEY=$($fk.Trim())" | Out-File ".env" -Encoding ascii
}

# ---------------------------------------------------------------- 3. pipeline
$env:PYTHONUNBUFFERED = "1"
Step "Downloading history (resumable; 20-40 min)"
Run $vpy @("-m", "smogcast", "fetch")
Step "Training and evaluating"
Run $vpy @("-m", "smogcast", "train")
Step "Making today's forecast"
Run $vpy @("-m", "smogcast", "forecast")

# ---------------------------------------------------------------- 4. push to GitHub
if (Test-Path "github_repo.txt") {
    $url = (Get-Content "github_repo.txt" -Raw).Trim()
    Step "Pushing to $url"
    $git = (Get-Command git -ErrorAction SilentlyContinue)
    if (-not $git) {
        Step "Git is not installed. Install it (winget install Git.Git) and double-click push_to_github.bat."
    } else {
        & "$PSScriptRoot\push_to_github.ps1"
    }
} else {
    Step "No github_repo.txt yet; skipped the push. Double-click push_to_github.bat later."
}
Step "ALL DONE"
"=== finished $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
Read-Host "Finished. Press Enter to close"
