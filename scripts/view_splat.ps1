# Open a trained splat in nerfstudio's viewer, then browse to http://localhost:7007.
#
#   powershell -ExecutionPolicy Bypass -File scripts\view_splat.ps1 -Run robot_scans\<session>
#
# -Run is a scan session or any folder above a training run; the newest run under it that finished (left a checkpoint) is opened.

param([Parameter(Mandatory = $true)][string]$Run, [int]$Port = 7007)

. (Join-Path $PSScriptRoot "splat_env.ps1")
$config = Get-ChildItem $Run -Recurse -Filter config.yml | Where-Object { Get-ChildItem (Join-Path $_.DirectoryName "nerfstudio_models") -Filter *.ckpt -ErrorAction SilentlyContinue } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $config) { throw "No finished training run (config.yml with a checkpoint) under $Run" }
Write-Host "Opening $($config.DirectoryName) - browse to http://localhost:$Port" -ForegroundColor Green
& (Join-Path $bin "ns-viewer.exe") --load-config $config.FullName --viewer.websocket-port $Port
