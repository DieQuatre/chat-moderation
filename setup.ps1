# Paket 0 - environment setup for chat-moderation
# Run from this folder:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
Start-Transcript -Path "$PSScriptRoot\setup_log.txt" -Force | Out-Null

function Check($step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at step: $step (exit code $LASTEXITCODE)" -ForegroundColor Red
        Stop-Transcript | Out-Null
        exit 1
    }
}

Write-Host "== 1/6 Python =="
$pyCmd = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
& $pyCmd --version 2>&1 | Out-Host
Check "python --version"
$ver = & $pyCmd -c "import sys; print(sys.version_info[0]*100 + sys.version_info[1])"
if ([int]$ver -lt 310) { Write-Host "Python 3.10 or newer is needed." -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

Write-Host "== 2/6 GPU driver =="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | Out-Host
} else { Write-Host "nvidia-smi not found" }

Write-Host "== 3/6 Virtual environment (.venv) =="
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $pyCmd -m venv .venv 2>&1 | Out-Host; Check "venv" }
$vpy = "$PSScriptRoot\.venv\Scripts\python.exe"
& $vpy -m pip install -q --timeout 120 --retries 10 --upgrade pip 2>&1 | Out-Host
Check "pip upgrade"

Write-Host "== 4/6 PyTorch with CUDA =="
& $vpy -m pip install -q --timeout 120 --retries 10 torch --index-url https://download.pytorch.org/whl/cu126 2>&1 | Out-Host
Check "torch"

Write-Host "== 5/6 Other libraries =="
& $vpy -m pip install -q --timeout 120 --retries 10 transformers accelerate scikit-learn pandas gensim onnx onnxscript onnxruntime gradio kagglehub huggingface_hub 2>&1 | Out-Host
Check "libraries"

Write-Host "== 6/6 Environment check =="
& $vpy check_env.py 2>&1 | Out-Host

if (Get-Command git -ErrorAction SilentlyContinue) {
    if (-not (Test-Path ".git")) { git init 2>&1 | Out-Host }
    git --version 2>&1 | Out-Host
} else { Write-Host "git not found" }

Write-Host "DONE" -ForegroundColor Green
Stop-Transcript | Out-Null
