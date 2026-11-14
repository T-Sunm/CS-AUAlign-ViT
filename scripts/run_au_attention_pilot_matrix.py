"""Run a fixed AU attention pilot matrix and write a reviewable Markdown report."""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# beta = g_scale * lambda; its SGD learning rate scales with g_scale**2 * lr_lambda.
CANDIDATES = (
    ("A", 3_000_000, 1e-6),
    ("B", 3_000_000, 3e-6),
    ("C", 3_000_000, 1e-5),
    ("D", 10_000_000, 9e-7),
)
MIN_STRONG_PAIRS = 20
MAX_SKIP_RATE = 0.05


def run_logged(command, log_path):
    start = time.monotonic()
    with log_path.open("w") as log, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    ) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    return code, round(time.monotonic() - start, 1)


def parse_train(log):
    match = re.search(r"AMP skipped (\d+)/(\d+) optimizer steps", log)
    if not match:
        return None
    rows = [line for line in log.splitlines() if line.startswith("step=")]
    if len(rows) != int(match.group(2)):
        return None
    last = rows[-1]
    loss = re.search(r"\bloss=([\d.eE+-]+)", last)
    effective = re.search(r"g_scale\*lambda_max=\[([^]]+)\]", last)
    if not loss or not effective:
        return None
    return {
        "skipped": int(match.group(1)),
        "steps": int(match.group(2)),
        "last_loss": float(loss.group(1)),
        "effective_lambda": [float(value) for value in effective.group(1).split(",")],
    }


def parse_calibration(log, scale):
    header = re.search(r"valid_pairs=(\d+) strong_pairs=(\d+)", log)
    if not header:
        return None
    for line in log.splitlines():
        parts = line.split()
        if len(parts) != 8:
            continue
        try:
            if float(parts[0]) != scale or parts[-1] not in {"True", "False"}:
                continue
            return {
                "valid_pairs": int(header.group(1)),
                "strong_pairs": int(header.group(2)),
                "p90": (float(parts[1]), float(parts[2])),
                "saturation": (float(parts[3].rstrip("%")) / 100, float(parts[4].rstrip("%")) / 100),
                "bias_ratio": (float(parts[5]), float(parts[6])),
            }
        except ValueError:
            continue
    return None


def eligible(row):
    train, calibration = row.get("train"), row.get("calibration")
    return bool(
        row["status"] == "completed" and train and calibration
        and train["skipped"] / train["steps"] <= MAX_SKIP_RATE
        and calibration["strong_pairs"] >= MIN_STRONG_PAIRS
        and all(value >= 0.01 for value in calibration["p90"])
        and all(value <= 0.05 for value in calibration["saturation"])
    )


def write_report(path, rows, args):
    spatial_mode = "heatmap" if args.source_heatmap_cache else "landmark"
    lines = [
        "# AU attention training pilot matrix",
        "",
        f"Train: RAF-DB → ExpW; seed 42; spatial mode {spatial_mode}; {args.steps} steps; batch {args.batch_size}.",
        f"Calibration: scan at most {args.calibration_batches} train batches; strong pair means max AU ≥ 0.5 in both images.",
        "Gate: P90 ≥ 0.01 in both directions, strong-pair saturation ≤ 5%, at least 20 strong pairs, AMP skipped steps ≤ 5%.",
        "No ExpW validation/test labels or accuracy are used for selection. Passing this pilot does not establish full-epoch stability.",
        "",
        "| Run | g_scale | lr_λ | g_scale²·lr_λ | Train | Last loss | AMP skips | P90 t→s / s→t | Saturation t→s / s→t | Bias/logit t→s / s→t | Valid / strong pairs | Gate |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        train, calibration = row.get("train"), row.get("calibration")
        loss = f"{train['last_loss']:.4g}" if train else "—"
        skips = f"{train['skipped']}/{train['steps']}" if train else "—"
        p90 = f"{calibration['p90'][0]:.4f} / {calibration['p90'][1]:.4f}" if calibration else "—"
        saturation = (
            f"{calibration['saturation'][0]:.2%} / {calibration['saturation'][1]:.2%}"
            if calibration else "—"
        )
        ratio = (
            f"{calibration['bias_ratio'][0]:.3g} / {calibration['bias_ratio'][1]:.3g}"
            if calibration else "—"
        )
        pairs = (
            f"{calibration['valid_pairs']} / {calibration['strong_pairs']}"
            if calibration else "—"
        )
        gate = "pass" if eligible(row) else ("pending" if row["status"] == "pending" else "fail")
        lines.append(
            f"| {row['name']} | {row['scale']:g} | {row['lr']:g} | "
            f"{row['scale'] ** 2 * row['lr']:g} | {row['status']} | {loss} | {skips} | "
            f"{p90} | {saturation} | {ratio} | {pairs} | {gate} |"
        )
    passing = [row for row in rows if eligible(row)]
    # Prefer a lower effective rate; break ties by fewer AMP skips, then smaller scale.
    selected = min(
        passing,
        key=lambda row: (row["scale"] ** 2 * row["lr"], row["train"]["skipped"], row["scale"]),
    ) if passing else None
    lines += [
        "",
        f"Pilot selection: **{selected['name']}**" if selected else "Pilot selection: **none yet**.",
        "",
        "## Artifacts",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['name']}: [train log]({row['name']}.train.log), "
            f"[calibration log]({row['name']}.calibration.log), "
            f"checkpoint `{row['name']}/adaptation.pt` if training completed; "
            f"train {row.get('train_seconds', 0):g}s, calibration {row.get('calibration_seconds', 0):g}s."
        )
    lines += [
        "",
        "A passing candidate still needs a longer smoke run and fixed-batch attention check before E1–E5 full training.",
        "The four runs each refresh epoch-0 pseudo labels; no target labels enter adaptation.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/adaptation/au-attention-pilot-matrix"))
    parser.add_argument("--config", type=Path, default=Path("configs/training/ablation/g1_heatmap.yaml"))
    parser.add_argument("--source-heatmap-cache", type=Path)
    parser.add_argument("--target-heatmap-cache", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calibration-batches", type=int, default=768)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.calibration_batches) < 1:
        parser.error("steps, batch-size and calibration-batches must be positive")
    if (args.source_heatmap_cache is None) != (args.target_heatmap_cache is None):
        parser.error("source and target heatmap caches must be provided together")
    if args.output.exists():
        parser.error("output directory already exists; choose a new --output")
    args.output.mkdir(parents=True)
    report = args.output / "report.md"
    rows = [
        {"name": name, "scale": scale, "lr": lr, "status": "pending"}
        for name, scale, lr in CANDIDATES
    ]
    write_report(report, rows, args)
    scripts = Path(__file__).resolve().parent

    for row in rows:
        name, scale, lr = row["name"], row["scale"], row["lr"]
        print(f"\n=== {name}: g_scale={scale:g}, lr_lambda={lr:g} ===", flush=True)
        train_command = [
            sys.executable, str(scripts / "pilot_au_attention_training.py"),
            "--target-manifest", str(args.target_manifest),
            "--config", str(args.config),
            "--output", str(args.output / name),
            "--steps", str(args.steps), "--batch-size", str(args.batch_size),
            "--g-scale", str(scale), "--lambda-learning-rate", str(lr),
        ]
        if args.source_heatmap_cache is not None:
            train_command += [
                "--source-heatmap-cache", str(args.source_heatmap_cache),
                "--target-heatmap-cache", str(args.target_heatmap_cache),
            ]
        code, seconds = run_logged(train_command, args.output / f"{name}.train.log")
        row["train_seconds"] = seconds
        row["train"] = parse_train((args.output / f"{name}.train.log").read_text())
        if code != 0 or row["train"] is None:
            row["status"] = f"train failed ({code})"
            write_report(report, rows, args)
            continue
        calibration_command = [
            sys.executable, str(scripts / "calibrate_au_attention.py"),
            "--checkpoint", str(args.output / name / "adaptation.pt"),
            "--target-manifest", str(args.target_manifest),
            "--batches", str(args.calibration_batches),
            "--min-strong-pairs", str(MIN_STRONG_PAIRS),
            "--scales", str(scale),
        ]
        if args.source_heatmap_cache is not None:
            calibration_command += [
                "--source-heatmap-cache", str(args.source_heatmap_cache),
                "--target-heatmap-cache", str(args.target_heatmap_cache),
            ]
        code, seconds = run_logged(calibration_command, args.output / f"{name}.calibration.log")
        row["calibration_seconds"] = seconds
        row["calibration"] = parse_calibration(
            (args.output / f"{name}.calibration.log").read_text(), scale
        )
        row["status"] = "completed" if code == 0 and row["calibration"] else f"calibration failed ({code})"
        write_report(report, rows, args)
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
