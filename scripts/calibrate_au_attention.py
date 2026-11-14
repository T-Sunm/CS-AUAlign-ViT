"""Find an AU attention scale from train batches, without using labels or accuracy."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from cs_aualign_vit.data.au_guidance import compute_guidance
from cs_aualign_vit.data.datasets import build_loader
from cs_aualign_vit.models import POSTBCSModel


def attention_logits(block, query, key):
    return (query * query.shape[-1] ** -0.5) @ key.transpose(-2, -1) + block.attn.get_relative_position_bias()


def measure(logits, guidance, lam, scales, valid, strong, results):
    baseline = F.softmax(logits, dim=-1)
    for scale in scales:
        bias = (scale * lam[None, :, None, None] * guidance[:, None]).to(logits.dtype)
        guided = F.softmax(logits + bias, dim=-1)
        change = 0.5 * (guided - baseline).abs().sum(dim=-1)
        newly_saturated = (baseline.amax(-1) < 0.95) & (guided.amax(-1) >= 0.95)
        row = results[scale]
        row["change"].append(change[valid].detach().float().cpu().flatten())
        row["saturated"] += int(newly_saturated[strong].sum())
        row["strong_rows"] += int(newly_saturated[strong].numel())
        row["bias_std"].append(bias[valid].detach().float().std().item())
    return logits[valid].detach().float().std().item()


def enough_evidence(processed_batches, strong_pairs, *, min_batches, min_strong_pairs):
    return processed_batches >= min_batches and strong_pairs >= min_strong_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/adaptation/smoke-e3-b16/adaptation.pt"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/processed/raf-v5-deduplicated/processed.jsonl"))
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-au-cache", type=Path, default=Path("runs/au_cache/raf_db_train.csv"))
    parser.add_argument("--target-au-cache", type=Path, default=Path("runs/au_cache/expw_train.csv"))
    parser.add_argument("--source-heatmap-cache", type=Path)
    parser.add_argument("--target-heatmap-cache", type=Path)
    parser.add_argument("--batches", type=int, default=256, help="maximum batches to scan")
    parser.add_argument("--min-batches", type=int, default=8)
    parser.add_argument("--min-strong-pairs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--strong-au", type=float, default=0.5)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1e6, 3e6, 1e7, 3e7, 1e8],
    )
    args = parser.parse_args()
    if (
        args.batches < 1
        or args.min_batches < 1
        or args.min_batches > args.batches
        or args.min_strong_pairs < 1
        or args.batch_size < 1
        or not 0 < args.strong_au <= 1
    ):
        parser.error("invalid scan limits, batch size, or strong-au threshold")
    if any(scale <= 0 for scale in args.scales):
        parser.error("all scales must be positive")
    if (args.source_heatmap_cache is None) != (args.target_heatmap_cache is None):
        parser.error("source and target heatmap caches must be provided together")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if config.get("au_mode") != "dynamic":
        raise ValueError("Pilot checkpoint must have au_mode=dynamic")
    source_checkpoint = Path(state["source_checkpoint"])
    model = POSTBCSModel.from_source_checkpoint(source_checkpoint, au_mode="dynamic")
    model.load_state_dict(state["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    bcs = model.bcs_block1
    if not torch.any(bcs.lambda_target_to_source.detach()) or not torch.any(bcs.lambda_source_to_target.detach()):
        raise ValueError("Both lambda vectors must have moved from zero before this pilot")

    spatial_mode = "heatmap" if args.source_heatmap_cache is not None else "landmark"
    source_loader = build_loader(
        args.source_manifest,
        "source_train",
        batch_size=args.batch_size,
        seed=42,
        au_cache=args.source_au_cache,
        spatial_cache=args.source_heatmap_cache,
        spatial_mode=spatial_mode,
    )
    target_loader = build_loader(
        args.target_manifest,
        "target_adaptation",
        batch_size=args.batch_size,
        seed=42,
        au_cache=args.target_au_cache,
        spatial_cache=args.target_heatmap_cache,
        spatial_mode=spatial_mode,
    )
    source_iter = iter(source_loader)
    results = {
        direction: {scale: {"change": [], "saturated": 0, "strong_rows": 0, "bias_std": []} for scale in args.scales}
        for direction in ("t2s", "s2t")
    }
    logit_stds = {"t2s": [], "s2t": []}
    valid_pairs = strong_pairs = 0
    processed_batches = 0
    amp = device.type == "cuda" and bool(config.get("amp", True))

    with torch.inference_mode():
        for batch_index, (target_images, _, target_au) in enumerate(target_loader):
            if batch_index == args.batches:
                break
            try:
                source_images, _, source_au = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_images, _, source_au = next(source_iter)
            count = min(len(source_images), len(target_images))
            source_images = source_images[:count].to(device)
            target_images = target_images[:count].to(device)
            source_au = {key: value[:count].to(device) for key, value in source_au.items()}
            target_au = {key: value[:count].to(device) for key, value in target_au.items()}
            valid = (
                source_au["au_valid"]
                & source_au["spatial_valid"]
                & target_au["au_valid"]
                & target_au["spatial_valid"]
            )
            strong = valid & (source_au["au"].amax(-1) >= args.strong_au) & (target_au["au"].amax(-1) >= args.strong_au)
            valid_pairs += int(valid.sum())
            strong_pairs += int(strong.sum())
            processed_batches += 1
            if not valid.any():
                continue
            guidance = compute_guidance(target_au["au"], source_au["au"], target_au["mask"], source_au["mask"], valid)
            with torch.autocast(device_type=device.type, enabled=amp):
                tokens = model.shared_trunk(torch.cat((source_images, target_images)))
                source, target = tokens.split((count, count))
                source = model.source_block0(source)
                target = model.target_block0(target)
                qs, ks, _ = bcs._qkv(bcs.source, source)
                qt, kt, _ = bcs._qkv(bcs.target, target)
                for direction, block, query, key, matrix, lam in (
                    ("t2s", bcs.source, qt, ks, guidance, bcs.lambda_target_to_source),
                    ("s2t", bcs.target, qs, kt, guidance.transpose(-2, -1), bcs.lambda_source_to_target),
                ):
                    logits = attention_logits(block, query, key)
                    logit_stds[direction].append(measure(logits, matrix, lam, args.scales, valid, strong, results[direction]))
            if enough_evidence(
                processed_batches,
                strong_pairs,
                min_batches=args.min_batches,
                min_strong_pairs=args.min_strong_pairs,
            ):
                break

    if not valid_pairs or strong_pairs < args.min_strong_pairs:
        raise ValueError(
            f"Need at least {args.min_strong_pairs} strong AU pairs; scanned "
            f"{processed_batches} batches and found valid={valid_pairs}, strong={strong_pairs}. "
            "Increase --batches; do not lower --strong-au without defining a new protocol."
        )
    print(
        f"checkpoint={args.checkpoint} epoch={state['completed_epoch']} "
        f"batches={processed_batches} valid_pairs={valid_pairs} "
        f"strong_pairs={strong_pairs} amp={amp}"
    )
    print("scale      t2s_P90  s2t_P90  t2s_sat  s2t_sat  t2s_bias/logit  s2t_bias/logit  pass")
    passing = []
    for scale in args.scales:
        metrics = []
        for direction in ("t2s", "s2t"):
            row = results[direction][scale]
            p90 = torch.quantile(torch.cat(row["change"]), 0.9).item()
            saturation = row["saturated"] / row["strong_rows"]
            ratio = sum(row["bias_std"]) / sum(logit_stds[direction])
            metrics.append((p90, saturation, ratio))
        passed = all(p90 >= 0.01 and saturation <= 0.05 for p90, saturation, _ in metrics)
        if passed:
            passing.append(scale)
        print(f"{scale:<10g} {metrics[0][0]:8.4f} {metrics[1][0]:8.4f} {metrics[0][1]:8.2%} {metrics[1][1]:8.2%} {metrics[0][2]:15.3g} {metrics[1][2]:15.3g}  {passed}")
    print(f"smallest passing scale: {min(passing) if passing else 'none; extend --scales'}")
    print("Scale is conditional on lambda from this checkpoint; rerun the smoke test and pilot after changing g_scale or lambda learning rate.")


if __name__ == "__main__":
    main()
