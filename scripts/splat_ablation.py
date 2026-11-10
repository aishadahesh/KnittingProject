r"""Does adding these pictures make the Gaussian splat better or worse? Trains it both ways and measures.

    .venv-splat\Scripts\python.exe scripts\splat_ablation.py --sessions ur5_20261008_143729 ur5_20261007_180059
    .venv-splat\Scripts\python.exe scripts\splat_ablation.py --latest 3 --per-image 12 --iterations 2000

Steps, all from the scan folders under robot_scans:
  1. Camera poses for every picture of every listed scan, solved together by solve_poses.py (COLMAP).
  2. Every Nth placed picture is held out of every training run; it is what each splat is judged on
     (PSNR / SSIM / LPIPS of the splat rendered from that picture's camera, from ns-eval).
  3. One training with everything (twice, the second run being the noise floor - splat training is not
     deterministic), then for each group of pictures (scan x height pass) one training without it and
     one with only it, and with --per-image K the same leaving out single pictures one by one.
  4. results.csv / results.json and a table: each run's score, its change against the full run, and a
     verdict - a change smaller than the noise floor is reported as "no effect", not as better/worse.

A picture's effect is "PSNR without it minus PSNR with everything": positive means it hurt. Pictures COLMAP
could not place cannot be trained on and are listed apart. Needs scripts\setup_splatting.ps1 to have been run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from persistence import write_json

SCRIPTS = ROOT / "scripts"
BIN = ROOT / ".venv-splat" / "Scripts"
MIN_NOISE_DB = 0.1


def powershell(command: str, log: Path) -> None:
    """Runs `command` in the environment splat_env.ps1 builds (CUDA, MSVC), echoing and keeping its output."""
    script = f'. "{SCRIPTS / "splat_env.ps1"}"; {command}; if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}'
    with log.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            sink.write(line)
    if process.wait() != 0:
        raise SystemExit(f"failed (exit {process.returncode}); see {log}")


def solve_poses(sessions: list[Path], poses: Path, fov: float) -> None:
    if (poses / "transforms.json").is_file():
        print(f"Reusing the poses in {poses} (delete it to solve again)", flush=True)
        return
    poses.mkdir(parents=True, exist_ok=True)
    write_json(poses / "combined.json", {"sessions": [session.name for session in sessions]})
    subprocess.run([sys.executable, str(SCRIPTS / "solve_poses.py"), "--session", str(poses), "--output", str(poses), "--fov", str(fov)], check=True)


def group_of(file_path: str) -> str:
    name = Path(file_path).name
    scan, _, picture = name.partition("__") if "__" in name else ("", "", name)
    height_pass = re.search(r"_(h\d+)_", picture)
    return f"{scan or 'scan'}:{height_pass.group(1) if height_pass else 'h1'}"


def run_experiment(name: str, train: list[dict], held_out: list[dict], meta: dict, poses: Path, work: Path, iterations: int) -> dict:
    out = work / name
    out.mkdir(parents=True, exist_ok=True)
    frames = train + held_out
    transforms = dict(meta, frames=frames, train_filenames=[f["file_path"] for f in train], val_filenames=[f["file_path"] for f in held_out],
                      test_filenames=[f["file_path"] for f in held_out])
    data = poses / f"experiment_{name}.json"
    write_json(data, transforms)
    print(f"[{name}] training on {len(train)} pictures, judged on {len(held_out)}", flush=True)
    powershell(f'$env:MAX_JOBS = "4"; & "{BIN / "ns-train.exe"}" splatfacto --data "{data}" --output-dir "{out}" --max-num-iterations {iterations} '
               f'--vis tensorboard --viewer.quit-on-train-completion True nerfstudio-data --eval-mode filename '
               f'--orientation-method none --center-method poses --auto-scale-poses True', out / "train.log")
    config = max(out.rglob("config.yml"), key=lambda path: path.stat().st_mtime)
    metrics = out / "metrics.json"
    powershell(f'& "{BIN / "ns-eval.exe"}" --load-config "{config}" --output-path "{metrics}"', out / "eval.log")
    scores = json.loads(metrics.read_text())["results"]
    return {"name": name, "train": len(train), "psnr": scores["psnr"], "ssim": scores["ssim"], "lpips": scores["lpips"]}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", nargs="*", default=[], help="scan folders (names under robot_scans, or paths)")
    parser.add_argument("--latest", type=int, default=0, help="or the N newest ur5_ scans")
    parser.add_argument("--out", default=None)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--per-image", type=int, default=0, help="also leave out this many single pictures, one training each")
    parser.add_argument("--fov", type=float, default=55.0)
    args = parser.parse_args()

    scans = ROOT / "robot_scans"
    sessions = [Path(s) if Path(s).is_dir() else scans / s for s in args.sessions]
    if args.latest:
        sessions += sorted(path for path in scans.glob("ur5_*") if path.is_dir() and path not in sessions)[-args.latest:]
    if not sessions:
        raise SystemExit("name scans with --sessions or take the newest with --latest")
    # combined.json names its scans by folder under robot_scans, so a scan elsewhere cannot be named there.
    outside = [str(s) for s in sessions if s.resolve().parent != scans.resolve()]
    if outside:
        raise SystemExit(f"scans must be folders under {scans}: {', '.join(outside)}")
    work = Path(args.out or scans / ("ablation_" + "_".join(s.name.removeprefix("ur5_") for s in sessions)[:80]))
    poses = work / "poses"
    solve_poses(sessions, poses, args.fov)

    meta = json.loads((poses / "transforms.json").read_text())
    frames = sorted(meta.pop("frames"), key=lambda f: f["file_path"])
    report = json.loads((poses / "pose_report.json").read_text())
    if len(frames) < 8:
        raise SystemExit(f"only {len(frames)} pictures were placed; too few to hold some out and compare. Take more overlap (0.6+) and try again.")
    held_out = frames[:: args.holdout_every]
    train = [f for f in frames if f not in held_out]
    groups = sorted({group_of(f["file_path"]) for f in train})
    print(f"{len(frames)} of {report['total']} pictures placed; {len(held_out)} held out, {len(train)} to train on, groups: {', '.join(groups)}", flush=True)

    results = []
    def run(name, subset):
        if not subset:
            return
        row = run_experiment(name, subset, held_out, meta, poses, work, args.iterations)
        results.append(row)
        write_json(work / "results.json", results)
    run("all", train)
    run("all_repeat", train)
    for group in groups:
        safe = group.replace(":", "_")
        if len(groups) > 1:
            run(f"without_{safe}", [f for f in train if group_of(f["file_path"]) != group])
            run(f"only_{safe}", [f for f in train if group_of(f["file_path"]) == group])
    if args.per_image:
        step = max(1, len(train) // args.per_image)
        for picked in train[::step][: args.per_image]:
            run(f"drop_{Path(picked['file_path']).stem}", [f for f in train if f is not picked])

    base = results[0]
    noise = max(MIN_NOISE_DB, abs(results[1]["psnr"] - base["psnr"])) if len(results) > 1 else MIN_NOISE_DB
    with (work / "results.csv").open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["run", "train_pictures", "psnr", "ssim", "lpips", "psnr_change_vs_all", "verdict"])
        print(f"\nNoise floor: {noise:.2f} dB (two identical trainings differ by that much)\n")
        print(f"{'run':60} {'pics':>5} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>6} {'change':>7}  verdict")
        for row in results:
            change = row["psnr"] - base["psnr"]
            if row["name"].startswith("only_"):
                verdict = "alone: " + ("worse" if change < -noise else "as good")
            elif row["name"] in ("all", "all_repeat"):
                verdict = "baseline"
            elif abs(change) <= noise:
                verdict = "no effect"
            else:
                verdict = "HELPS (removing it hurts)" if change < 0 else "HURTS (removing it helps)"
            writer.writerow([row["name"], row["train"], f"{row['psnr']:.3f}", f"{row['ssim']:.4f}", f"{row['lpips']:.4f}", f"{change:+.3f}", verdict])
            print(f"{row['name']:60} {row['train']:>5} {row['psnr']:>7.2f} {row['ssim']:>6.3f} {row['lpips']:>6.3f} {change:>+7.2f}  {verdict}")
    drops = [r["psnr"] - base["psnr"] for r in results if r["name"].startswith("drop_")]
    if drops:
        print(f"\nSingle pictures: mean effect of removing one {statistics.mean(drops):+.3f} dB, spread {statistics.pstdev(drops):.3f} dB (noise floor {noise:.2f})")
    print(f"\nWritten: {work / 'results.csv'}")


if __name__ == "__main__":
    main()
