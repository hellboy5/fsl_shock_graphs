"""
scripts/compute_graph_stats.py
Computes dataset-wide normalization parameters for Shock Graphs using
batched Welford's streaming accumulation (flat ~50MB RAM footprint).
Uses absolute curvature magnitude (torch.abs), bilateral boundary pooling (p == m),
reservoir sampling for distribution inspection, and outputs the YAML config block.
"""

import argparse
import glob
import math
import os
import random
import torch
from tqdm import tqdm

EDGE_FEATURE_NAMES = [
    "s_length",      # 0 (Spine)
    "s_curve",       # 1 (Spine - Absolute Curvature)
    "s_angle",       # 2 (Spine - Angle Change)
    "p_length",      # 3 (Plus / Left Boundary)
    "p_curve",       # 4 (Plus / Left Boundary - Absolute Curvature)
    "p_angle",       # 5 (Plus / Left Boundary - Angle Change)
    "m_length",      # 6 (Minus / Right Boundary)
    "m_curve",       # 7 (Minus / Right Boundary - Absolute Curvature)
    "m_angle",       # 8 (Minus / Right Boundary - Angle Change)
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


def update_welford_batch(existing_count, existing_mean, existing_m2, batch):
    b_count = batch.size(0)
    if b_count == 0:
        return existing_count, existing_mean, existing_m2

    b_mean = batch.mean(dim=0)
    b_m2 = ((batch - b_mean) ** 2).sum(dim=0)

    if existing_count == 0:
        return b_count, b_mean, b_m2

    new_count = existing_count + b_count
    delta = b_mean - existing_mean
    new_mean = existing_mean + delta * (b_count / new_count)
    new_m2 = existing_m2 + b_m2 + (delta ** 2) * (existing_count * b_count / new_count)

    return new_count, new_mean, new_m2


def main():
    parser = argparse.ArgumentParser(description="Welford Streaming Graph Stats with Absolute Curvature")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--split", type=str, default="train", help="Dataset subfolder to scan (e.g. 'train')")
    parser.add_argument("--image_size", type=float, default=84.0, help="Image canvas resolution")
    parser.add_argument(
        "--use_coarse",
        action="store_true",
        help="If set, only process files ending with '_coarse.pt'. Otherwise process regular '*.pt' files.",
    )
    parser.add_argument(
        "--reservoir_size",
        type=int,
        default=50000,
        help="Size of reservoir sample for percentile inspection",
    )
    args = parser.parse_args()

    target_dir = os.path.join(args.data_root, args.split)
    all_candidate_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.pt"), recursive=True))

    if not all_candidate_files:
        raise FileNotFoundError(f"No .pt files found in {target_dir}")

    if args.use_coarse:
        pt_files = [f for f in all_candidate_files if f.endswith("_coarse.pt")]
        file_mode_str = "COARSENED (*_coarse.pt)"
    else:
        pt_files = [f for f in all_candidate_files if not f.endswith("_coarse.pt")]
        file_mode_str = "UNCOARSENED (*.pt, excluding *_coarse.pt)"

    if not pt_files:
        raise FileNotFoundError(f"No matching files found for {file_mode_str} in {target_dir}")

    diag = math.sqrt(2.0) * args.image_size
    area = args.image_size * args.image_size

    print(f"\n==================================================")
    print(f"Target Directory: {target_dir}")
    print(f"Selection Mode  : {file_mode_str}")
    print(f"Matching Graphs : {len(pt_files)} (out of {len(all_candidate_files)} total .pt files)")
    print(f"Canvas Size     : {args.image_size}x{args.image_size} | Diagonal: {diag:.2f} | Area: {area:.2f}")
    print(f"Streaming Mode  : Welford Accumulator (< 50MB RAM)")
    print(f"Curvature Mode  : Absolute Magnitude (log1p(|kappa|))")
    print(f"==================================================\n")

    # Welford accumulators
    node_count = 0
    node_mean = torch.zeros(1)
    node_m2 = torch.zeros(1)

    edge_count = 0
    edge_mean = torch.zeros(14)
    edge_m2 = torch.zeros(14)

    taper_sq_sum = 0.0

    # Reservoir buffers for distribution check
    res_node_raw, res_node_trans = [], []
    res_edge_raw, res_edge_trans = [], []
    total_nodes_seen = 0
    total_edges_seen = 0
    k_res = args.reservoir_size

    for path in tqdm(pt_files, desc="Streaming Graphs"):
        data = torch.load(path, weights_only=False)

        # ------------------ Process Node Features ------------------
        if hasattr(data, "x") and data.x is not None and data.x.shape[0] > 0:
            raw_t = data.x[:, 2:3]
            trans_t = torch.log(torch.clamp(raw_t / diag, min=0.0) + 1e-5)

            node_count, node_mean, node_m2 = update_welford_batch(
                node_count, node_mean, node_m2, trans_t
            )

            for r_val, t_val in zip(raw_t.flatten(), trans_t.flatten()):
                if len(res_node_raw) < k_res:
                    res_node_raw.append(r_val)
                    res_node_trans.append(t_val)
                else:
                    j = random.randint(0, total_nodes_seen)
                    if j < k_res:
                        res_node_raw[j] = r_val
                        res_node_trans[j] = t_val
                total_nodes_seen += 1

        # ------------------ Process Edge Features ------------------
        if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            e_raw = data.edge_attr.clone()
            e_trans = e_raw.clone()

            # (a) Lengths and Thicknesses: scale by diag, log
            e_trans[:, [0, 3, 6, 10, 11]] = torch.log(
                torch.clamp(e_trans[:, [0, 3, 6, 10, 11]] / diag, min=0.0) + 1e-5
            )
            # (b) Bounded Polygon Area: scale by area, log
            e_trans[:, 9] = torch.log(torch.clamp(e_trans[:, 9] / area, min=0.0) + 1e-5)

            # (c) Curvatures (1, 4, 7): Absolute magnitude + log1p
            curv_idx = [1, 4, 7]
            e_trans[:, curv_idx] = torch.log1p(torch.abs(e_trans[:, curv_idx]))

            # (d) Angles & Flare (2, 5, 8, 13): Non-negative + log1p
            angle_idx = [2, 5, 8, 13]
            e_trans[:, angle_idx] = torch.log1p(torch.clamp(e_trans[:, angle_idx], min=0.0))

            edge_count, edge_mean, edge_m2 = update_welford_batch(
                edge_count, edge_mean, edge_m2, e_trans
            )

            # Accumulate sum of squares for taper_rate (index 12)
            taper_sq_sum += (e_trans[:, 12] ** 2).sum().item()

            for r_row, t_row in zip(e_raw, e_trans):
                if len(res_edge_raw) < k_res:
                    res_edge_raw.append(r_row)
                    res_edge_trans.append(t_row)
                else:
                    j = random.randint(0, total_edges_seen)
                    if j < k_res:
                        res_edge_raw[j] = r_row
                        res_edge_trans[j] = t_row
                total_edges_seen += 1

        del data

    if node_count == 0:
        raise ValueError("No valid nodes found across any matching .pt file.")

    # ------------------ 1. Distribution Health-Check ------------------
    print("\n" + "=" * 90)
    print(f"📊 FEATURE DISTRIBUTION HEALTH-CHECK [{file_mode_str}] (Reservoir N={len(res_edge_raw):,})")
    print("=" * 90)
    print("Feature          |      Min |       1% |   Median |      99% |       Max |   Zeros | Flags")
    print("-" * 90)

    if res_node_raw:
        print_distribution_summary("node_t (raw)", torch.stack(res_node_raw))
        print_distribution_summary("node_t (log)", torch.stack(res_node_trans))
        print("-" * 90)

    if res_edge_raw:
        res_e_raw_cat = torch.stack(res_edge_raw)
        res_e_trans_cat = torch.stack(res_edge_trans)
        for i, name in enumerate(EDGE_FEATURE_NAMES):
            print_distribution_summary(f"{name} (raw)", res_e_raw_cat[:, i])
            print_distribution_summary(f"{name} (trans)", res_e_trans_cat[:, i])
            print("-" * 90)

    # ------------------ 2. Final Normalization Parameters ------------------
    final_node_t_mean = node_mean.item()
    final_node_t_std = max(math.sqrt(node_m2.item() / max(node_count - 1, 1)), 1e-6)

    final_edge_mean = edge_mean.clone()
    final_edge_std = torch.sqrt(edge_m2 / max(edge_count - 1, 1))

    # -------------------------------------------------------------
    # POOL BOUNDARY STATISTICS (Tie Plus 'p' and Minus 'm')
    # -------------------------------------------------------------
    # Lengths (3: p_length, 6: m_length)
    bdry_len_mean = 0.5 * (edge_mean[3].item() + edge_mean[6].item())
    bdry_len_m2 = 0.5 * (edge_m2[3].item() + edge_m2[6].item())
    bdry_len_std = max(math.sqrt(bdry_len_m2 / max(edge_count - 1, 1)), 1e-6)
    final_edge_mean[3] = final_edge_mean[6] = bdry_len_mean
    final_edge_std[3] = final_edge_std[6] = bdry_len_std

    # Curvatures (4: p_curve, 7: m_curve)
    bdry_curv_mean = 0.5 * (edge_mean[4].item() + edge_mean[7].item())
    bdry_curv_m2 = 0.5 * (edge_m2[4].item() + edge_m2[7].item())
    bdry_curv_std = max(math.sqrt(bdry_curv_m2 / max(edge_count - 1, 1)), 1e-6)
    final_edge_mean[4] = final_edge_mean[7] = bdry_curv_mean
    final_edge_std[4] = final_edge_std[7] = bdry_curv_std

    # Angles (5: p_angle, 8: m_angle)
    bdry_ang_mean = 0.5 * (edge_mean[5].item() + edge_mean[8].item())
    bdry_ang_m2 = 0.5 * (edge_m2[5].item() + edge_m2[8].item())
    bdry_ang_std = max(math.sqrt(bdry_ang_m2 / max(edge_count - 1, 1)), 1e-6)
    final_edge_mean[5] = final_edge_mean[8] = bdry_ang_mean
    final_edge_std[5] = final_edge_std[8] = bdry_ang_std

    # Enforce symmetric zero-mean for taper rate (index 12)
    final_edge_mean[12] = 0.0
    final_edge_std[12] = max(math.sqrt(taper_sq_sum / max(edge_count, 1)), 1e-6)
    final_edge_std = torch.clamp(final_edge_std, min=1e-6)

    # ------------------ 3. Formatted YAML Output ------------------
    print("\n" + "=" * 50)
    print(f"PASTE THIS INTO YOUR configs/dataset/mini_imagenet.yaml ({file_mode_str}):")
    print("=" * 50)
    print("graph:")
    print(f"  image_size: {int(args.image_size)}")
    print(f"  node_mean: [0.0, 0.0, {final_node_t_mean:.4f}]")
    print(f"  node_std:  [1.0, 1.0, {final_node_t_std:.4f}]")

    edge_mean_str = ", ".join([f"{x:.4f}" for x in final_edge_mean.tolist()])
    edge_std_str = ", ".join([f"{x:.4f}" for x in final_edge_std.tolist()])
    print(f"  edge_mean: [{edge_mean_str}]")
    print(f"  edge_std:  [{edge_std_str}]")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
