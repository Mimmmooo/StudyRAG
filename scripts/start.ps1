# Idempotent local launcher for Windows PowerShell.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$model = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "qwen2.5:3b" }

function Info([string]$message) { Write-Host "[StudyRAG] $message" }
function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user;$env:Path"
}

# Obtain Tesseract only once. Language models live in the user profile.
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Windows App Installer (winget) is needed to install Tesseract."
    }
    Info "Installing Tesseract OCR (one-time)"
    winget install --id UB-Mannheim.TesseractOCR --exact --accept-source-agreements --accept-package-agreements
    Refresh-Path
    $env:Path = "$env:ProgramFiles\Tesseract-OCR;$env:Path"
}
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    throw "Tesseract is not on PATH; restart PowerShell and rerun this command."
}
$tessdata = Join-Path $env:LOCALAPPDATA "StudyRAG\tessdata"
New-Item -ItemType Directory -Path $tessdata -Force | Out-Null
foreach ($lang in @("ben", "eng")) {
    $target = Join-Path $tessdata "$lang.traineddata"
    if (-not (Test-Path $target)) {
        Info "Downloading $lang OCR language data (one-time)"
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/$lang.traineddata" -OutFile $target
    }
}
$env:TESSDATA_PREFIX = $tessdata

$py = $null
foreach ($version in @("3.12", "3.11")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py "-$version" -c "import sys; assert sys.version_info[:2] in ((3,11),(3,12))" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $version; break }
    }
}
if (-not $py) {
    Info "Installing managed Python 3.12 (one-time)"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        irm https://astral.sh/uv/install.ps1 | iex
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
    & uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw "Could not install Python 3.12." }
}
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython -c "import sys; assert sys.version_info[:2] in ((3,11),(3,12))" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Info "Replacing incompatible virtual environment"
        Remove-Item (Join-Path $root ".venv") -Recurse -Force
    }
}
if (-not (Test-Path $venvPython)) {
    Info "Creating virtual environment (one-time)"
    if ($py) {
        & py "-$py" -m venv .venv
    } else {
        & uv venv --python 3.12 .venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}
$stamp = Join-Path $root ".venv\studyrag-deps.txt"
$fingerprint = (Get-FileHash (Join-Path $root "requirements.txt") -Algorithm SHA256).Hash
$stale = (-not (Test-Path $stamp)) -or ((Get-Content $stamp -Raw).Trim() -ne $fingerprint)
if (-not $stale) {
    & $venvPython -c "import streamlit, fitz, fastembed, chromadb" 2>$null
    $stale = $LASTEXITCODE -ne 0
}
if ($stale) {
    Info "Installing or repairing Python dependencies (only if changed/missing)"
    & $venvPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        & $venvPython -m ensurepip --upgrade
    }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
    Set-Content -Path $stamp -Value $fingerprint
} else { Info "Python dependencies are already installed; skipping pip." }

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Info "Installing Ollama (one-time)"
    irm https://ollama.com/install.ps1 | iex
    Refresh-Path
    $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama was installed but is not on PATH; restart PowerShell and rerun."
}
function Ollama-Ready {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
        return $true
    } catch { return $false }
}
if (-not (Ollama-Ready)) {
    Info "Starting local Ollama"
    Start-Process -FilePath (Get-Command ollama).Source -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Ollama-Ready); $i++) { Start-Sleep -Seconds 1 }
}
if (-not (Ollama-Ready)) { throw "Ollama did not start. Check system logs." }
$existing = & ollama list
if (-not (($existing | Select-Object -Skip 1) -match ("^" + [regex]::Escape($model) + "\s"))) {
    Info "Downloading Ollama model $model (one-time)"
    & ollama pull $model
    if ($LASTEXITCODE -ne 0) { throw "Could not pull Ollama model $model." }
} else { Info "Ollama model $model is already installed." }
Info "Launching StudyRAG at http://localhost:8501"
& $venvPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
