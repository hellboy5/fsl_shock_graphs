# scripts/compute_graph_stats.py
import os
import glob
import torch
import argparse

def compute_train_stats(train_dir):
    node_feats = []
    edge_feats = []

    pth_files = glob.glob(os.path.join(train_dir, "**/*.pth"), recursive=True)
    print(f"Found {len(pth_files)} .pth files. Computing stats...")
    
    for pth in pth_files:
        data = torch.load(pth)
        if data.x is not None and data.x.shape[0] > 0:
            node_feats.append(data.x[:, 0:3])
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            edge_feats.append(data.edge_attr)

    all_nodes = torch.cat(node_feats, dim=0)
    all_edges = torch.cat(edge_feats, dim=0)

    print("\n--- YAML OUTPUT ---")
    print("normalize_graph:")
    print(f"  node_mean: {all_nodes.mean(dim=0).tolist()}")
    print(f"  node_std: {all_nodes.std(dim=0).tolist()}")
    print(f"  edge_mean: {all_edges.mean(dim=0).tolist()}")
    print(f"  edge_std: {all_edges.std(dim=0).tolist()}")
    print("-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the train dataset directory")
    args = parser.parse_args()
    compute_train_stats(args.train_dir)
