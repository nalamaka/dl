$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location $projectRoot

Write-Host "[exp6] project root: $projectRoot"

$pythonExe = Join-Path $projectRoot "mineru_env\Scripts\python.exe"
if (Test-Path $pythonExe) {
    Write-Host "[exp6] python: $pythonExe"
} else {
    $pythonExe = "python"
    Write-Host "[exp6] python: system python"
}

$dataDir = Join-Path $projectRoot "data\CamVid"
$requiredDirs = @(
    "train",
    "train_labels",
    "val",
    "val_labels",
    "test",
    "test_labels"
)

foreach ($dir in $requiredDirs) {
    $full = Join-Path $dataDir $dir
    if (-not (Test-Path $full)) {
        throw "Missing directory: $full`nPlease prepare CamVid first. See src_to_submit/exp6/deploy_data.md"
    }
}

Write-Host "[exp6] install dependencies..."
& $pythonExe -m pip install -r src_to_submit/exp6/requirements.txt
& $pythonExe -m pip install matplotlib

${ckptCandidates} = @(
    "src_to_submit/exp6/checkpoints/segnet_camvid.best.pth",
    "src_to_submit/exp6/checkpoints/segnet_camvid.last.pth",
    "src_to_submit/exp6/checkpoints/segnet_camvid.pth",
    "src/exp6/checkpoints/segnet_camvid.best.pth",
    "src/exp6/checkpoints/segnet_camvid.last.pth",
    "src/exp6/checkpoints/segnet_camvid.pth"
)

$resumeCkpt = $null
foreach ($c in $ckptCandidates) {
    if (Test-Path (Join-Path $projectRoot $c)) {
        $resumeCkpt = $c
        break
    }
}

if ($resumeCkpt) {
    Write-Host "[exp6] quick mode: reuse checkpoint -> $resumeCkpt"
} else {
    Write-Host "[exp6] no checkpoint found, run smoke train (1 epoch)..."
    & $pythonExe src_to_submit/exp6/main.py --mode train --epochs 1 --batch_size 2 --ckpt_name segnet_camvid_smoke.pth
    $resumeCkpt = "src_to_submit/exp6/checkpoints/segnet_camvid_smoke.best.pth"
}

Write-Host "[exp6] evaluate..."
& $pythonExe src_to_submit/exp6/main.py --mode eval --batch_size 2 --resume_ckpt $resumeCkpt

Write-Host "[exp6] predict..."
& $pythonExe src_to_submit/exp6/main.py --mode predict --batch_size 2 --resume_ckpt $resumeCkpt --pred_dir src_to_submit/exp6/visualizations/predictions_smoke

$latestLog = Get-ChildItem -Path "src_to_submit/exp6/logs" -Filter "*.jsonl" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if ($latestLog) {
    $hasEpoch = Select-String -Path $latestLog.FullName -Pattern '"type"\s*:\s*"epoch"' -Quiet
    if ($hasEpoch) {
        Write-Host "[exp6] visualize latest log..."
        & $pythonExe src_to_submit/exp6/visualize_results.py --log_file $latestLog.FullName --out_dir src_to_submit/exp6/visualizations/smoke
    } else {
        Write-Host "[exp6] skip visualize: latest log has no epoch records -> $($latestLog.FullName)"
    }
} else {
    Write-Host "[exp6] skip visualize: no train log found in src_to_submit/exp6/logs"
}

Write-Host "[exp6] done. Check:"
Write-Host "  - src_to_submit/exp6/checkpoints"
Write-Host "  - src_to_submit/exp6/logs"
Write-Host "  - src_to_submit/exp6/visualizations/smoke"
