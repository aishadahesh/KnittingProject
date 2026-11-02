# Measure what TeleopApp and app.py cost the machine while they run - per process, per thread, and
# on the GPU - and write it to logs\profile\ for reading afterwards.
#
#   powershell -ExecutionPolicy Bypass -File scripts\profile_apps.ps1 [-Seconds 30]
#
# Start TeleopApp and/or app.py first and leave them as you would use them. A thread that sleeps
# shows near 0%; one that spins while "waiting" shows near 100% of a core however idle the app looks.

param([int]$Seconds = 30)

$ErrorActionPreference = "Stop"
$out = Join-Path (Split-Path -Parent $PSScriptRoot) "logs\profile"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$cores = [Environment]::ProcessorCount

function Snapshot {
    # The venv's python.exe is a launcher stub that starts uv's base interpreter, so the app is found by
    # its command line rather than by where its exe lives.
    $appIds = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like "*app.py*" -or $_.CommandLine -like "*KnittingProject*" } | ForEach-Object { $_.ProcessId })
    $procs = @(Get-Process TeleopApp -ErrorAction SilentlyContinue) + @($appIds | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    $threads = @{}
    foreach ($p in $procs) {
        foreach ($t in $p.Threads) {
            try { $threads["$($p.ProcessName)[$($p.Id)] thread $($t.Id)"] = $t.TotalProcessorTime.TotalSeconds } catch {}
        }
    }
    [pscustomobject]@{ Time = Get-Date; Procs = $procs | ForEach-Object { [pscustomobject]@{ Name = "$($_.ProcessName)[$($_.Id)]"; Cpu = $_.TotalProcessorTime.TotalSeconds; MemMB = [math]::Round($_.WorkingSet64 / 1MB) } }; Threads = $threads }
}

Write-Host "Profiling for $Seconds s - use the apps as usual..." -ForegroundColor Cyan
$first = Snapshot
if (-not $first.Procs) { throw "Neither TeleopApp nor the project's python is running - start them first" }
$gpu = @()
$hasSmi = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
for ($i = 0; $i -lt $Seconds; $i++) {
    Start-Sleep -Seconds 1
    if ($hasSmi) { $gpu += (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) }
}
$last = Snapshot
$elapsed = ($last.Time - $first.Time).TotalSeconds

$report = @("Profiled $([math]::Round($elapsed)) s on $cores logical cores ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))", "")
$report += "PROCESS                         CPU (% of one core)   % of whole machine   memory"
foreach ($p in $last.Procs) {
    $before = ($first.Procs | Where-Object Name -eq $p.Name).Cpu
    if ($null -eq $before) { continue }
    $core = 100.0 * ($p.Cpu - $before) / $elapsed
    $report += ("{0,-32}{1,10:N0} %{2,20:N1} %{3,10} MB" -f $p.Name, $core, ($core / $cores), $p.MemMB)
}
$report += "", "BUSIEST THREADS (a thread near 100% is spinning, not working)"
$busy = foreach ($key in $last.Threads.Keys) {
    if ($first.Threads.ContainsKey($key)) { [pscustomobject]@{ Thread = $key; Core = 100.0 * ($last.Threads[$key] - $first.Threads[$key]) / $elapsed } }
}
$busy | Sort-Object Core -Descending | Select-Object -First 12 | ForEach-Object { $report += ("{0,-45}{1,8:N0} %" -f $_.Thread, $_.Core) }
if ($gpu.Count) {
    $util = $gpu | ForEach-Object { [double]($_.Split(",")[0]) }
    $mem = $gpu | ForEach-Object { [double]($_.Split(",")[1]) }
    $report += "", ("GPU  average {0:N0} %  peak {1:N0} %   memory {2:N0} MB" -f ($util | Measure-Object -Average).Average, ($util | Measure-Object -Maximum).Maximum, ($mem | Measure-Object -Maximum).Maximum)
}
$file = Join-Path $out ("profile_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
$report | Set-Content -Path $file -Encoding utf8
$report | ForEach-Object { Write-Host $_ }
Write-Host "`nSaved to $file" -ForegroundColor Green
