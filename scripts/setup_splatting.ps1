# One-time setup of a Gaussian Splatting trainer (nerfstudio's splatfacto) in its own venv.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_splatting.ps1
#
# Kept out of the app's .venv on purpose: PyTorch with CUDA and nerfstudio are several GB and pin
# versions the app does not need. Downloads, from PyPI and download.pytorch.org:
#   - PyTorch + torchvision built for CUDA 12.8 (what an RTX 50-series card needs), ~3 GB
#   - nerfstudio and gsplat, with their dependencies, ~1-2 GB
#   - pycolmap (CPU), which solves the camera poses of a hand-held capture, ~100 MB
# gsplat compiles its CUDA kernels the first time it trains, which needs the NVIDIA CUDA Toolkit
# 12.8 (nvcc) installed and the Visual Studio Build Tools this machine already has.

$ErrorActionPreference = "Stop"
# The CUDA Toolkit, from the machine-wide CUDA_PATH its installer set - not from this window's PATH,
# which is stale in any terminal opened before the toolkit was installed. gsplat compiles with nvcc.
$cuda = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
if ($cuda -and (Test-Path (Join-Path $cuda "bin\nvcc.exe"))) {
    $env:CUDA_PATH = $cuda
    $env:CUDA_HOME = $cuda
    $env:PATH = (Join-Path $cuda "bin") + ";" + $env:PATH
}
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv-splat"
$python = Join-Path $venv "Scripts\python.exe"

Write-Host "=== 1/4 Creating $venv (Python 3.11) ===" -ForegroundColor Cyan
uv venv $venv --python 3.11

Write-Host "=== 2/4 nerfstudio, gsplat and pycolmap ===" -ForegroundColor Cyan
uv pip install --python $python nerfstudio gsplat pycolmap

Write-Host "=== 3/4 PyTorch for CUDA 12.8 (replacing any CPU build pulled in above) ===" -ForegroundColor Cyan
# Only torch and torchvision are reinstalled, and PyPI stays the fallback index: a blanket --reinstall
# against the PyTorch index alone dropped dependencies it does not host (typing_extensions, filelock, jinja2).
uv pip install --python $python --reinstall-package torch --reinstall-package torchvision torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128

Write-Host "=== 4/4 Checking the GPU is visible ===" -ForegroundColor Cyan
& $python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    Write-Host "nvcc: $((nvcc --version | Select-String 'release').ToString().Trim())"
} else {
    Write-Host "nvcc not found and no machine-wide CUDA_PATH - install the NVIDIA CUDA Toolkit 12.8 before training." -ForegroundColor Yellow
}
Write-Host "Done. In the app: 9. Gaussian Splatting -> 'Use nerfstudio splatfacto', Export, Run reconstruction." -ForegroundColor Green
