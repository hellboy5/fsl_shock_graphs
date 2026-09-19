"""
scripts/compute_graph_stats.py
Computes dataset-wide normalization parameters for Shock Graphs,
inspects feature distributions, and prints the config YAML block.
Supports toggling between coarsened (*_coarse.pt) and uncoarsened (*.pt) files.
"""

import argparse
import glob
import math
import os
import torch
from tqdm import tqdm

EDGE_FEATURE_NAMES = [
    "s_length",      # 0
    "s_curve",       # 1
    "s_angle",       # 2
    "p_length",      # 3
    "p_curve",       # 4
    "p_angle",       # 5
    "m_length",      # 6
    "m_curve",       # 7
    "m_angle",       # 8
    "poly_area",     # 9
    "avg_thickness", # 10
    "max_thickness", # 11
    "taper_rate",    # 12
    "total_flare",   # 13
]


def print_distribution_summary(name, tensor):
    flat = tensor.flatten().float()
    n_elems = flat.numel()
    if n_elems == 0:
        print(f"{name:<16} | EMPTY TENSOR")
        return

    n_nans = torch.isnan(flat).sum().item()
    n_infs = torch.isinf(flat).sum().item()
    if n_nans > 0 or n_infs > 0:
        print(f"⚠️  {name:<14} | CONTAINS {n_nans} NaNs and {n_infs} Infs!")
        flat = flat[torch.isfinite(flat)]
        if flat.numel() == 0:
            return

    pct_zeros = (flat == 0.0).sum().item() / n_elems * 100.0
    q = torch.quantile(flat, torch.tensor([0.0, 0.01, 0.50, 0.99, 1.0], device=flat.device))
    min_v, p1, med, p99, max_v = q[0].item(), q[1].item(), q[2].item(), q[3].item(), q[4].item()

    warning_flag = ""
    if abs(max_v) > 100 * max(abs(p99), 1e-4) or abs(min_v) > 100 * max(abs(p1), 1e-4):
        warning_flag = "⚠️  [Extreme Outlier Tail]"
    elif flat.std().item() < 1e-5:
        warning_flag = "⚠️  [Near Zero Variance]"

    print(
        f"{name:<16} | Min: {min_v:9.3f} | 1%: {p1:8.3f} | Med: {med:8.3f} | "
        f"99%: {p99:8.3f} | Max: {max_v:9.3f} | Zeros: {pct_zeros:5.1f}% {warning_flag}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compute Graph Stats & Check Feature Distributions")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--split", type=str, default="train", help="Dataset subfolder to scan (e.g. 'train')")
    parser.add_argument("--image_size", type=float, default=84.0, help="Image canvas resolution")
    parser.add_argument(
        "--use_coarse",
        action="store_true",
        help="If set, only process files ending with '_coarse.pt'. If not set, only process uncoarsened '*.pt' files.",
    )
    args = parser.parse_args()

    target_dir = os.path.join(args.data_root, args.split)
    all_candidate_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.pt"), recursive=True))

    if not all_candidate_files:
        raise FileNotFoundError(f"No .pt files found in {target_dir}")

    # Disambiguate between coarse and uncoarsened files
    if args.use_coarse:
        pt_files = [f for f in all_candidate_files if f.endswith("_coarse.pt")]
        file_mode_str = "COARSENED (*_coarse.pt)"
    else:
        pt_files = [f for f in all_candidate_files if not f.endswith("_coarse.pt")]
        file_mode_str = "UNCOARSENED (*.pt, excluding *_coarse.pt)"

    if not pt_files:
        raise FileNotFoundError(
            f"No matching files found for mode {file_mode_str} under {target_dir}. "
            f"Total .pt files scanned: {len(all_candidate_files)}"
        )

    diag = math.sqrt(2.0) * args.image_size
    area = args.image_size * args.image_size

    print(f"\n==================================================")
    print(f"Target Directory: {target_dir}")
    print(f"Selection Mode  : {file_mode_str}")
    print(f"Matching Graphs : {len(pt_files)} (out of {len(all_candidate_files)} total .pt files)")
    print(f"Canvas Size     : {args.image_size}x{args.image_size} | Diagonal: {diag:.2f} | Area: {area:.2f}")
    print(f"==================================================\n")

    all_raw_node_t = []
    all_log_node_t = []
    all_transformed_edges = []
    all_raw_edges = []

    for path in tqdm(pt_files, desc="Processing Graphs"):
        data = torch.load(path, weights_only=False)

        # ------------------ Node Features ------------------
        if hasattr(data, "x") and data.x is not None and data.x.shape[0] > 0:
            raw_t = data.x[:, 2]
            all_raw_node_t.append(raw_t)
            all_log_node_t.append(torch.log(torch.clamp(raw_t / diag, min=0.0) + 1e-5))

        # ------------------ Edge Features ------------------
        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            e_raw = data.edge_attr.clone()
            all_raw_edges.append(e_raw)

            e_trans = e_raw.clone()
            # Lengths & Thicknesses: scale by diag, log
            e_trans[:, [0, 3, 6, 10, 11]] = torch.log(
                torch.clamp(e_trans[:, [0, 3, 6, 10, 11]] / diag, min=0.0) + 1e-5
            )
            # Area: scale by area, log
            e_trans[:, 9] = torch.log(torch.clamp(e_trans[:, 9] / area, min=0.0) + 1e-5)
            # Curves, Angles, Flare: log1p
            e_trans[:, [1, 2, 4, 5, 7, 8, 13]] = torch.log1p(
                torch.clamp(e_trans[:, [1, 2, 4, 5, 7, 8, 13]], min=0.0)
            )

            all_transformed_edges.append(e_trans)

    if len(all_raw_node_t) == 0:
        raise ValueError("No valid nodes found across any matching .pt file.")

    cat_raw_node_t = torch.cat(all_raw_node_t)
    cat_log_node_t = torch.cat(all_log_node_t)

    has_edges = len(all_raw_edges) > 0
    if has_edges:
        cat_raw_edges = torch.cat(all_raw_edges, dim=0)
        cat_trans_edges = torch.cat(all_transformed_edges, dim=0)

    # ------------------ 1. Distribution Health-Check ------------------
    print("\n" + "=" * 90)
    print(f"📊 FEATURE DISTRIBUTION HEALTH-CHECK [{file_mode_str}]")
    print("=" * 90)
    print("Feature          |      Min |       1% |   Median |      99% |       Max |   Zeros | Flags")
    print("-" * 90)

    print_distribution_summary("node_t (raw)", cat_raw_node_t)
    print_distribution_summary("node_t (log)", cat_log_node_t)
    print("-" * 90)

    if has_edges:
        for i, name in enumerate(EDGE_FEATURE_NAMES):
            print_distribution_summary(f"{name} (raw)", cat_raw_edges[:, i])
            print_distribution_summary(f"{name} (trans)", cat_trans_edges[:, i])
            print("-" * 90)

    # ------------------ 2. Normalization Parameters ------------------
    t_mean = cat_log_node_t.mean().item()
    t_std = max(cat_log_node_t.std().item(), 1e-6)

    if has_edges:
        edge_means = cat_trans_edges.mean(dim=0)
        edge_stds = cat_trans_edges.std(dim=0)

        # Enforce symmetric zero-mean for taper rate (index 12)
        edge_means[12] = 0.0
        edge_stds[12] = torch.sqrt(torch.mean(cat_trans_edges[:, 12] ** 2))
        edge_stds = torch.clamp(edge_stds, min=1e-6)
    else:
        edge_means = torch.zeros(14)
        edge_stds = torch.ones(14)

    # ------------------ 3. Formatted YAML Output ------------------
    print("\n" + "=" * 50)
    print(f"PASTE THIS INTO YOUR configs/dataset/mini_imagenet.yaml ({file_mode_str}):")
    print("=" * 50)
    print("graph:")
    print(f"  image_size: {int(args.image_size)}")
    print(f"  node_mean: [0.0, 0.0, {t_mean:.4f}]")
    print(f"  node_std:  [1.0, 1.0, {t_std:.4f}]")

    edge_mean_str = ", ".join([f"{x:.4f}" for x in edge_means.tolist()])
    edge_std_str = ", ".join([f"{x:.4f}" for x in edge_stds.tolist()])
    print(f"  edge_mean: [{edge_mean_str}]")
    print(f"  edge_std:  [{edge_std_str}]")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
