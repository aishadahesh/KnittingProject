# Train a Gaussian splat on an exported dataset with nerfstudio's splatfacto, then export it as a PLY.
#
#   powershell -ExecutionPolicy Bypass -File scripts\train_splat.ps1 -Dataset <splat_dataset> -Output <splat_output>
#
# The app's "Use nerfstudio splatfacto" preset runs exactly this. The poses come from the robot, so
# nerfstudio does not re-orient them, but it does centre and scale them: splatfacto's learning rates
# and split/cull sizes assume a scene about a unit across, and a scan is a hand-width of cloth in
# metres - in metres the splat never settled. The applied centre and scale are in the run's
# dataparser_transforms.json. The export's fabric_seed.ply starts it from points on the cloth.
# Needs scripts\setup_splatting.ps1 to have been run once.

param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$Output,
    # 0 picks by the dataset: an object-only export (one small thing on black) trains 30,000 steps
    # and is allowed to split finely; anything else - a cloth, texture edge to edge - trains 15,000
    # steps and splits only where the pictures really disagree, or it balloons past a million
    # Gaussians at half a second a step.
    [int]$Iterations = 0,
    # -Plain: splatfacto exactly as shipped (no camera optimiser, no culling or splitting changes,
    # no background choice) - the control run when a result is bad and the cause is unknown.
    [switch]$Plain,
    # -ExportOnly: no training; export, judge and render the latest checkpoint already under -Output.
    [switch]$ExportOnly,
    # -Detail: a cloth split as finely as one object (splatfacto's own 0.0008 with absgrad) and kept
    # splitting to five sixths of the run - for the yarn and how the loops interlock, at the price of
    # more Gaussians and a slower run. Worth it only on pictures that are sharp and well posed.
    [switch]$Detail,
    # -SplitAt: how readily a Gaussian splits, in place of the choice above (e.g. 0.0005 for the
    # finest yarn on close, sharp pictures).
    [double]$SplitAt = 0,
    # -Antialiased: gsplat's antialiased rasteriser, which keeps a Gaussian from shrinking below a
    # pixel - steadier when the pictures were taken at several distances.
    [switch]$Antialiased
)

. (Join-Path $PSScriptRoot "splat_env.ps1")
# Training must not take the machine: below-normal priority (inherited by everything it starts), and
# gsplat's one-time kernel build limited to half the cores - unlimited, it ran a compiler per core.
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
$env:MAX_JOBS = [string][Math]::Max(2, [int]([Environment]::ProcessorCount / 2))

# The camera optimiser lets each pose shift a little during training: the arm's poses with the
# calibrated mount are right to about ten pixels, and this takes up the rest instead of smearing it
# into floaters. (COLMAP poses are refined by it too; with poses that are plain wrong it cannot help.)
# An object-only export (everything but the object painted black) trains against a black background,
# so nothing is spent on the paper and the splat is the object alone.
$transforms = Get-Content (Join-Path $Dataset "transforms.json") -Raw | ConvertFrom-Json
$objectOnly = [bool]$transforms.object_only
$background = if ($objectOnly) { "black" } else { "random" }
if ($Iterations -le 0) { $Iterations = if ($objectOnly) { 30000 } else { 15000 } }
# How readily a Gaussian splits: the default 0.0008 is right for one small object that must come
# out finely; on a cloth it makes a Gaussian per fibre and a million-Gaussian model, so there a
# split takes three times the disagreement.
$splitAt = if ($SplitAt -gt 0) { $SplitAt } elseif ($objectOnly -or $Detail) { 0.0008 } else { 0.0025 }
# Against floaters and needles: a Gaussian that fades below 0.15 is culled (default 0.1), one that
# stretches past 5:1 is penalised (scale regularisation), and splitting stops at two thirds of the
# run so the last third only refines what is there.
$stopSplit = if ($Detail) { [int]($Iterations * 5 / 6) } else { [int]($Iterations * 2 / 3) }
# Stopping early keeps the result. ns-train saves a checkpoint every 500 steps; it is run as a
# child process watched from here, and when a STOP file appears in the output folder (the app's
# Stop button writes it) or Ctrl+C is pressed, the child is ended and the latest checkpoint is
# exported, judged and rendered exactly as a finished run would be. -ExportOnly skips training.
$stopFile = Join-Path $Output "STOP"
if (Test-Path $stopFile) { Remove-Item $stopFile }
Write-Host "To stop early and keep the result: press Stop in the app, or create the file $stopFile"
$common = @("--viewer.quit-on-train-completion", "True", "--steps-per-save", "500")
$data = @("nerfstudio-data", "--orientation-method", "none", "--center-method", "poses", "--auto-scale-poses", "True")
if ($Plain) {
    Write-Host "Training $Iterations steps, splatfacto as shipped (control run)"
    $trainArgs = @("splatfacto", "--data", "`"$Dataset`"", "--output-dir", "`"$Output`"", "--max-num-iterations", $Iterations) + $common + $data
} else {
    Write-Host "Training $Iterations steps ($(if ($objectOnly) { 'one object on black' } else { 'cloth, background kept' }))"
    $trainArgs = @("splatfacto", "--data", "`"$Dataset`"", "--output-dir", "`"$Output`"", "--max-num-iterations", $Iterations) + $common + @(
        "--pipeline.model.camera-optimizer.mode", "SO3xR3",
        "--pipeline.model.background-color", $background,
        "--pipeline.model.cull-alpha-thresh", "0.15",
        "--pipeline.model.densify-grad-thresh", $splitAt,
        "--pipeline.model.use-scale-regularization", "True", "--pipeline.model.max-gauss-ratio", "5.0",
        "--pipeline.model.stop-split-at", $stopSplit,
        "--pipeline.model.rasterize-mode", $(if ($Antialiased) { "antialiased" } else { "classic" })) + $data
}
$stopped = $false
if (-not $ExportOnly) {
    $train = Start-Process -FilePath (Join-Path $bin "ns-train.exe") -ArgumentList $trainArgs -NoNewWindow -PassThru
    $null = $train.Handle    # without this PowerShell forgets the exit code once the process ends
    try {
        while (-not $train.HasExited) {
            Start-Sleep -Seconds 2
            if (Test-Path $stopFile) {
                Write-Host "Stop asked for: ending training and keeping the latest checkpoint"
                $stopped = $true
                Stop-Process -Id $train.Id -Force
                $train.WaitForExit()
                Remove-Item $stopFile -ErrorAction SilentlyContinue
            }
        }
    } finally {
        if (-not $train.HasExited) { Stop-Process -Id $train.Id -Force; $stopped = $true }
    }
    $train.WaitForExit()
    if (-not $stopped -and $train.ExitCode -ne 0) {
        throw "ns-train failed with exit code $($train.ExitCode) - the cause is in the lines above (a Traceback), or in splat_output\train.log when run from the app"
    }
}

$config = Get-ChildItem $Output -Recurse -Filter config.yml | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $config) { throw "No config.yml under $Output - nothing was trained" }
$checkpoint = Get-ChildItem $config.DirectoryName -Recurse -Filter "step-*.ckpt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $checkpoint) { throw "No checkpoint yet under $($config.DirectoryName) - the run was stopped before its first save (step 500)" }
Write-Host "Exporting $($checkpoint.Name)$(if ($stopped) { ' (stopped early)' })"
& (Join-Path $bin "ns-export.exe") gaussian-splat --load-config $config.FullName --output-dir (Join-Path $Output "export")
if ($LASTEXITCODE -ne 0) { throw "ns-export failed with exit code $LASTEXITCODE" }
Write-Host "Splat written to $(Join-Path $Output 'export')"

# The only honest measure of the splat: how well it reproduces the pictures held out of training
# (every tenth one), as PSNR in eval.json and as the rendered pictures themselves under renders\.
& (Join-Path $bin "ns-eval.exe") --load-config $config.FullName --output-path (Join-Path $Output "eval.json")
if ($LASTEXITCODE -ne 0) { throw "ns-eval failed with exit code $LASTEXITCODE" }
& (Join-Path $bin "ns-render.exe") dataset --load-config $config.FullName --output-path (Join-Path $Output "renders") --split train+test --rendered-output-names rgb --image-format png
if ($LASTEXITCODE -ne 0) { throw "ns-render failed with exit code $LASTEXITCODE" }
Write-Host "Held-out pictures rendered to $(Join-Path $Output 'renders')"
