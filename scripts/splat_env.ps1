# The environment every nerfstudio/gsplat command needs on this machine, dot-sourced by train_splat.ps1
# and view_splat.ps1. gsplat loads its CUDA kernels through torch's JIT loader, which looks for cl.exe on
# every start - not only the first build - so viewing a finished splat needs this as much as training.

$ErrorActionPreference = "Stop"
# The CUDA Toolkit, from the machine-wide CUDA_PATH its installer set - not from this window's PATH,
# which is stale in any terminal opened before the toolkit was installed. gsplat compiles with nvcc.
$cuda = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
if ($cuda -and (Test-Path (Join-Path $cuda "bin\nvcc.exe"))) {
    $env:CUDA_PATH = $cuda
    $env:CUDA_HOME = $cuda
    $env:PATH = (Join-Path $cuda "bin") + ";" + $env:PATH
}
# nvcc on Windows compiles through MSVC's cl.exe, which is only on PATH inside the Visual Studio
# environment - so that environment is loaded into this one, from wherever Visual Studio is installed.
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    $install = if (Test-Path $vswhere) { & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath }
    $vcvars = if ($install) { Join-Path $install "VC\Auxiliary\Build\vcvars64.bat" }
    if (-not $vcvars -or -not (Test-Path $vcvars)) { throw "No Visual Studio C++ build tools found - gsplat needs cl.exe" }
    cmd /c "call `"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
    }
}
# nerfstudio prints box-drawing characters, which the Windows console code page (cp1252) cannot encode:
# with output piped to the app, that killed training before its first step. Python writes UTF-8 instead.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# torch 2.9-2.11's compiled_autograd.h fails under nvcc on Windows with C2872 "'std': ambiguous symbol"
# (pytorch#173232) unless USE_CUDA is defined, which selects its Windows branch. gsplat takes no extra
# flags, but cl.exe reads the CL variable on every call, nvcc's host passes included.
$env:CL = "/DUSE_CUDA"
# nerfstudio 1.1.5 reads its own checkpoints with a bare torch.load, which torch 2.6+ restricts to plain
# weights - ns-export and ns-viewer then refused the checkpoint training had just written. The file is our own.
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
$bin = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv-splat\Scripts"
if (-not (Test-Path (Join-Path $bin "ns-train.exe"))) { throw "No nerfstudio at $bin - run scripts\setup_splatting.ps1 first" }
