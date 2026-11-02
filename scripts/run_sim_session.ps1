# Build TeleopApp, start it on its emulator, run the simulated-cell tests, then start app.py -
# every step's output mirrored into logs\sim_session\ so it can be read while the GUIs run.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_sim_session.ps1
#
# Leave the UR5e powered off (or its network unplugged): TeleopApp chooses its emulator when the
# control box does not answer at startup.

$ErrorActionPreference = "Stop"
$knitting = Split-Path -Parent $PSScriptRoot
$fleet    = Join-Path (Split-Path -Parent $knitting) "CraftLabXRFleet"
# The MSVC environment of the compiler cpp\build was configured with - read from its CMakeCache, so
# the build never mixes toolsets - then Visual Studio's own locator, then the documented paths.
$vsSetups = @()
$cache = Join-Path $fleet "cpp\build\CMakeCache.txt"
if (Test-Path $cache) {
    $compiler = (Select-String -Path $cache -Pattern '^CMAKE_CXX_COMPILER:FILEPATH=(.+)$').Matches | Select-Object -First 1
    if ($compiler) {
        $vcRoot = ($compiler.Groups[1].Value -replace '/', '\') -replace '\\Tools\\MSVC\\.*$', ''
        $vsSetups += Join-Path $vcRoot "Auxiliary\Build\vcvars64.bat"
    }
}
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($install) { $vsSetups += Join-Path $install "VC\Auxiliary\Build\vcvars64.bat" }
}
$vsSetups += @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars   = $vsSetups | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
$teleop   = Join-Path $fleet "cpp\build\source\apps\TeleopApp\TeleopApp.exe"
$logs     = Join-Path $knitting "logs\sim_session"

New-Item -ItemType Directory -Force -Path $logs | Out-Null
Get-ChildItem $logs -File | Remove-Item -Force
function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan; Add-Content (Join-Path $logs "0_steps.log") "$(Get-Date -Format HH:mm:ss)  $text" }

Step "1/4 Building TeleopApp"
if (Get-Process TeleopApp -ErrorAction SilentlyContinue) { throw "TeleopApp is already running - close it first, the build cannot replace a running exe" }
if (-not $vcvars) { Step "BUILD FAILED - no Visual Studio environment found at any of: $($vsSetups -join '; ')"; exit 1 }
if (-not (Test-Path (Join-Path $fleet "cpp\build\build.ninja"))) { Step "BUILD FAILED - no configured build at $fleet\cpp\build"; exit 1 }
Step "   MSVC environment: $vcvars"
# `call` first: cmd /c strips the outer quotes of a command that starts with one, which broke the path.
cmd /c "call `"$vcvars`" >nul && cd /d `"$fleet\cpp`" && cmake --build build --config Release --target TeleopApp 2>&1" | Tee-Object -FilePath (Join-Path $logs "1_teleop_build.log")
if ($LASTEXITCODE -ne 0) { Step "BUILD FAILED - see logs\sim_session\1_teleop_build.log"; exit 1 }

Step "2/4 Starting TeleopApp (emulator)"
Start-Process -FilePath $teleop -WorkingDirectory (Split-Path -Parent $teleop) `
    -RedirectStandardOutput (Join-Path $logs "2_teleop_stdout.log") -RedirectStandardError (Join-Path $logs "2_teleop_stderr.log")
Start-Sleep -Seconds 8

Step "3/4 Simulated-cell tests"
Set-Location $knitting
cmd /c "uv run pytest tests/test_sim_cell.py 2>&1" | Tee-Object -FilePath (Join-Path $logs "3_pytest.log")

Step "4/4 Starting app.py - UR5 Robot mode: Connect Via TeleopApp, Build table + knitted fabric, camera 'TeleopApp: Gripper camera', grid, Check, Reset, Run"
cmd /c "uv run app.py 2>&1" | Tee-Object -FilePath (Join-Path $logs "4_app.log")
Step "app.py exited"
