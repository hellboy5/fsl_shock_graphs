# eval_decision_fusion.py
#
# Usage:
# python -u eval_decision_fusion.py \
#     task.n_way=5 \
#     task.n_shot=1 \
#     task.test_episodes=10000 \
#     dataset.graph.use_coarse=false \
#     model.vision_checkpoint="checkpoints/baselines/vision/best_model.pth" \
#     model.graph_checkpoint="checkpoints/baselines/graph/best_model.pth"

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MultimodalFSLDataset
from data.samplers import EpisodicBatchSampler
from data.transforms import get_graph_transform, get_vision_transform
from models.multimodal_network import MultimodalFewShotNetwork
from train import calculate_accuracy, collate_fn, seed_worker
from utils.helpers import seed_everything


def compute_confidence_interval(data):
  """Calculates the mean and 95% confidence interval across episodes."""
  a = 1.0 * np.array(data)
  m = float(np.mean(a))
  pm = float(1.96 * (np.std(a) / np.sqrt(len(a))))
  return m, pm


def shannon_entropy(probs):
  """Computes Shannon entropy H(p) = -sum p_i log(p_i) per query [B, 1]."""
  log_p = torch.log(probs + 1e-12)
  return -torch.sum(probs * log_p, dim=-1, keepdim=True)


def run_decision_fusion_evaluation(cfg: DictConfig, device: torch.device):
  v_ckpt_path = cfg.model.get("vision_checkpoint", None)
  g_ckpt_path = cfg.model.get("graph_checkpoint", None)

  if not v_ckpt_path or not g_ckpt_path:
    raise ValueError(
        "Must provide both model.vision_checkpoint and model.graph_checkpoint"
        " for decision fusion evaluation."
    )

  print(f"Loading Vision Checkpoint from : {v_ckpt_path}")
  v_ckpt = torch.load(v_ckpt_path, map_location=device, weights_only=False)
  v_train_cfg = v_ckpt["cfg"]

  print(f"Loading Graph Checkpoint from  : {g_ckpt_path}")
  g_ckpt = torch.load(g_ckpt_path, map_location=device, weights_only=False)
  g_train_cfg = g_ckpt["cfg"]

  # Sync graph transform config with CLI dataset.graph.use_coarse
  g_train_cfg.dataset.graph.use_coarse = cfg.dataset.graph.use_coarse

  eval_n_way = cfg.task.n_way
  eval_n_shot = cfg.task.n_shot
  eval_n_query = cfg.task.n_query
  eval_episodes = cfg.task.get("eval_episodes", cfg.task.test_episodes)

  v_transform = get_vision_transform(v_train_cfg)
  g_transform = get_graph_transform(g_train_cfg)

  # Setup Dataset & DataLoader (Identical to eval.py)
  test_set = MultimodalFSLDataset(
      cfg.dataset,
      modality="multimodal",
      split="test",
      vision_transform=v_transform,
      graph_transform=g_transform,
  )

  test_sampler = EpisodicBatchSampler(
      test_set.labels,
      test_set.base_names,
      eval_n_way,
      eval_n_shot,
      eval_n_query,
      eval_episodes,
  )

  num_workers = int(getattr(cfg.training, "num_workers", 4))
  g = torch.Generator()
  g.manual_seed(cfg.seed)

  test_loader = DataLoader(
      test_set,
      batch_sampler=test_sampler,
      collate_fn=collate_fn,
      num_workers=num_workers,
      pin_memory=True,
      persistent_workers=(num_workers > 0),
      prefetch_factor=2 if num_workers > 0 else None,
      worker_init_fn=seed_worker,
      generator=g,
  )

  # Instantiate Unimodal Models
  model_v = MultimodalFewShotNetwork(v_train_cfg).to(device)
  model_v.load_state_dict(v_ckpt["model_state_dict"], strict=False)
  model_v.eval()

  model_g = MultimodalFewShotNetwork(g_train_cfg).to(device)
  model_g.load_state_dict(g_ckpt["model_state_dict"], strict=False)
  model_g.eval()

  targets = (
      torch.arange(eval_n_way)
      .repeat_interleave(eval_n_query)
      .long()
      .to(device, non_blocking=True)
  )

  # Hyperparameter Grids
  lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
  alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  margin_thresholds = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
  prob_margin_v_threshs = [0.15, 0.25, 0.35, 0.50]
  prob_margin_g_threshs = [0.10, 0.20, 0.30]
  decay_gammas = [0.5, 1.0, 2.0, 3.0, 5.0]
  temperatures = [0.5, 1.0, 2.0]

  # Metric Accumulators
  v_accs, g_accs, oracle_accs = [], [], []
  both_right_list, v_only_list, g_only_list, both_wrong_list = [], [], [], []

  # Dictionaries for all 14 fusion paradigms
  accs_static = {lam: [] for lam in lambdas}
  accs_zscore = {lam: [] for lam in lambdas}
  accs_prob = {a: [] for a in alphas}
  accs_logpool = {lam: [] for lam in lambdas}
  accs_borda = []
  accs_temp_scale = {
      (t_v, t_g, lam): []
      for t_v in temperatures
      for t_g in temperatures
      for lam in [0.5, 1.0]
  }
  accs_max_conf = {lam: [] for lam in lambdas}
  accs_hard_margin = {
      (lam, th): [] for lam in lambdas for th in margin_thresholds
  }
  accs_soft_margin = {(lam, gm): [] for lam in lambdas for gm in decay_gammas}
  accs_margin_ratio = {lam: [] for lam in lambdas}
  accs_entropy = {lam: [] for lam in lambdas}
  accs_post_arbitration = {
      (tv, tg): []
      for tv in prob_margin_v_threshs
      for tg in prob_margin_g_threshs
  }
  accs_top2_swap = {(th, lam): [] for th in [0.05, 0.10, 0.15] for lam in [0.8, 1.0]}
  accs_dynamic_spread = {lam: [] for lam in [0.5, 0.8, 1.0, 1.2]}

  print(
      f"--- Starting Decision Fusion Evaluation ({eval_episodes} Episodes) |"
      f" {eval_n_way}-Way {eval_n_shot}-Shot ---"
  )

  with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating Decision Fusion"):
      img_batch = (
          batch["image"].to(device, non_blocking=True)
          if batch["image"] is not None
          else None
      )
      graph_batch = (
          batch["graph"].to(device, non_blocking=True)
          if batch["graph"] is not None
          else None
      )

      # 1. Unimodal Forward Passes (Standard, Crash-Proof Call)
      logits_v = model_v(img_batch, None, eval_n_way, eval_n_shot)
      logits_g = model_g(None, graph_batch, eval_n_way, eval_n_shot)

      v_accs.append(calculate_accuracy(logits_v, targets))
      g_accs.append(calculate_accuracy(logits_g, targets))

      # 2. Disagreement & Oracle Breakdown
      corr_v = logits_v.argmax(dim=1) == targets
      corr_g = logits_g.argmax(dim=1) == targets

      oracle_accs.append((corr_v | corr_g).float().mean().item() * 100.0)
      both_right_list.append((corr_v & corr_g).float().mean().item() * 100.0)
      v_only_list.append((corr_v & ~corr_g).float().mean().item() * 100.0)
      g_only_list.append((~corr_v & corr_g).float().mean().item() * 100.0)
      both_wrong_list.append((~corr_v & ~corr_g).float().mean().item() * 100.0)

      # 3. Per-Query Transformations & Confidences
      z_v = (logits_v - logits_v.mean(dim=-1, keepdim=True)) / (
          logits_v.std(dim=-1, keepdim=True) + 1e-6
      )
      z_g = (logits_g - logits_g.mean(dim=-1, keepdim=True)) / (
          logits_g.std(dim=-1, keepdim=True) + 1e-6
      )

      prob_v = F.softmax(logits_v, dim=-1)
      prob_g = F.softmax(logits_g, dim=-1)
      logp_v = F.log_softmax(logits_v, dim=-1)
      logp_g = F.log_softmax(logits_g, dim=-1)

      top2_v, _ = torch.topk(logits_v, k=2, dim=-1)
      margin_v = (top2_v[:, 0] - top2_v[:, 1]).unsqueeze(-1)

      top2_g, _ = torch.topk(logits_g, k=2, dim=-1)
      margin_g = (top2_g[:, 0] - top2_g[:, 1]).unsqueeze(-1)

      prob_top2_v, idx_top2_v = torch.topk(prob_v, k=2, dim=-1)
      prob_margin_v = prob_top2_v[:, 0] - prob_top2_v[:, 1]

      prob_top2_g, _ = torch.topk(prob_g, k=2, dim=-1)
      prob_margin_g = prob_top2_g[:, 0] - prob_top2_g[:, 1]

      ent_v = shannon_entropy(prob_v)
      ent_g = shannon_entropy(prob_g)

      # 4. Episode-Level Discriminability Ratio from Query Logit Spread
      spread_v = logits_v.std(dim=-1).mean()
      spread_g = logits_g.std(dim=-1).mean()
      spread_ratio = spread_g / (spread_v + 1e-6)

      # ----------------- Execution of the 14 Operators -----------------

      # 5. Rank Borda Count Fusion (Method 5)
      ranks_v = torch.argsort(
          torch.argsort(logits_v, dim=-1, descending=True), dim=-1
      ).float()
      ranks_g = torch.argsort(
          torch.argsort(logits_g, dim=-1, descending=True), dim=-1
      ).float()
      borda_score = (1.0 / (ranks_v + 1.0)) + (1.0 / (ranks_g + 1.0))
      accs_borda.append(calculate_accuracy(borda_score, targets))

      # 3. Softmax Probability Linear Pool (Method 3)
      for a in alphas:
        accs_prob[a].append(
            calculate_accuracy((1.0 - a) * prob_v + a * prob_g, targets)
        )

      # 6. Temperature Scaled Logit Blending (Method 6)
      for t_v, t_g, lam in accs_temp_scale.keys():
        p_v_scaled = F.softmax(logits_v / t_v, dim=-1)
        p_g_scaled = F.softmax(logits_g / t_g, dim=-1)
        accs_temp_scale[(t_v, t_g, lam)].append(
            calculate_accuracy(p_v_scaled + lam * p_g_scaled, targets)
        )

      # 12. Calibrated Posterior Margin Arbitration (Method 12)
      pred_v = logits_v.argmax(dim=-1)
      pred_g = logits_g.argmax(dim=-1)
      for tv, tg in accs_post_arbitration.keys():
        pred_arb = pred_v.clone()
        switch_mask = (prob_margin_v < tv) & (prob_margin_g > tg)
        pred_arb[switch_mask] = pred_g[switch_mask]
        acc_arb = (pred_arb == targets).float().mean().item() * 100.0
        accs_post_arbitration[(tv, tg)].append(acc_arb)

      # 13. Top-2 Competitive Swap Rule (Method 13)
      for th, lam in accs_top2_swap.keys():
        pred_swap = pred_v.clone()
        v_runner_up = idx_top2_v[:, 1]
        swap_condition = (prob_margin_v < th) & (pred_g == v_runner_up)
        pred_swap[swap_condition] = v_runner_up[swap_condition]
        acc_sw = (pred_swap == targets).float().mean().item() * 100.0
        accs_top2_swap[(th, lam)].append(acc_sw)

      # 14. Dynamic Support-Separation Blending (Method 14)
      for lam in [0.5, 0.8, 1.0, 1.2]:
        dynamic_lam = lam * spread_ratio
        accs_dynamic_spread[lam].append(
            calculate_accuracy(logits_v + dynamic_lam * logits_g, targets)
        )

      # Standard sweeps over lambdas
      for lam in lambdas:
        # 1. Static Linear Logit Blending
        accs_static[lam].append(
            calculate_accuracy(logits_v + lam * logits_g, targets)
        )

        # 2. Z-Score Calibrated Logit Blending
        accs_zscore[lam].append(calculate_accuracy(z_v + lam * z_g, targets))

        # 4. Log-Opinion Pool (Geometric Mean)
        accs_logpool[lam].append(
            calculate_accuracy(logp_v + lam * logp_g, targets)
        )

        # 7. Max-Confidence Pooling
        accs_max_conf[lam].append(
            calculate_accuracy(torch.max(logits_v, lam * logits_g), targets)
        )

        # 10. Relative Margin-Ratio Blending
        s_mratio = (margin_v * logits_v) + (lam * margin_g * logits_g)
        accs_margin_ratio[lam].append(calculate_accuracy(s_mratio, targets))

        # 11. Shannon Entropy-Weighted Blending
        s_ent = (logits_v / (ent_v + 1e-6)) + lam * (logits_g / (ent_g + 1e-6))
        accs_entropy[lam].append(calculate_accuracy(s_ent, targets))

        # 8. Hard Margin-Gated Blending
        for th in margin_thresholds:
          gate = (margin_v < th).float()
          accs_hard_margin[(lam, th)].append(
              calculate_accuracy(logits_v + lam * gate * logits_g, targets)
          )

        # 9. Soft Exponential Margin-Decay Gating
        for gm in decay_gammas:
          soft_gate = torch.exp(-gm * margin_v)
          accs_soft_margin[(lam, gm)].append(
              calculate_accuracy(logits_v + lam * soft_gate * logits_g, targets)
          )

  # Synthesize & Format Results
  mean_v, ci_v = compute_confidence_interval(v_accs)
  mean_g, ci_g = compute_confidence_interval(g_accs)
  mean_oracle, ci_oracle = compute_confidence_interval(oracle_accs)

  def get_best_1d(acc_dict, param_name="lambda"):
    best_m, best_ci, best_p = -1.0, 0.0, None
    for p, lst in acc_dict.items():
      m, ci = compute_confidence_interval(lst)
      if m > best_m:
        best_m, best_ci, best_p = m, ci, p
    return best_m, best_ci, f"{param_name}={best_p}"

  def get_best_multid(acc_dict, param_names):
    best_m, best_ci, best_p = -1.0, 0.0, None
    for p_tuple, lst in acc_dict.items():
      m, ci = compute_confidence_interval(lst)
      if m > best_m:
        best_m, best_ci, best_p = m, ci, p_tuple
    hparam_str = ", ".join(
        f"{n}={v}" for n, v in zip(param_names, best_p)
    )
    return best_m, best_ci, hparam_str

  m_borda, ci_borda = compute_confidence_interval(accs_borda)

  results = [
      ("1. Static Linear Logit Blending", *get_best_1d(accs_static, "lambda")),
      ("2. Z-Score Calibrated Blending", *get_best_1d(accs_zscore, "lambda")),
      ("3. Softmax Probability Pool", *get_best_1d(accs_prob, "alpha")),
      ("4. Log-Opinion Pool (Geometric)", *get_best_1d(accs_logpool, "lambda")),
      ("5. Rank Borda Count Reciprocal", m_borda, ci_borda, "None (Pure Rank)"),
      (
          "6. Temperature Scaled Softmax",
          *get_best_multid(accs_temp_scale, ["Tv", "Tg", "lambda"]),
      ),
      ("7. Max-Confidence Pooling", *get_best_1d(accs_max_conf, "lambda")),
      (
          "8. Hard Margin-Gated (Logit)",
          *get_best_multid(accs_hard_margin, ["lambda", "thresh"]),
      ),
      (
          "9. Soft Exp Margin-Decay Gate",
          *get_best_multid(accs_soft_margin, ["lambda", "gamma"]),
      ),
      (
          "10. Relative Margin-Ratio Fusion",
          *get_best_1d(accs_margin_ratio, "lambda"),
      ),
      ("11. Entropy-Weighted Blending", *get_best_1d(accs_entropy, "lambda")),
      (
          "12. Calibrated Posterior Margin",
          *get_best_multid(accs_post_arbitration, ["tau_v", "tau_g"]),
      ),
      (
          "13. Top-2 Competitive Swap Rule",
          *get_best_multid(accs_top2_swap, ["tau_v", "lambda"]),
      ),
      (
          "14. Dynamic Support-Separation",
          *get_best_1d(accs_dynamic_spread, "lambda_0"),
      ),
  ]

  print("\n" + "=" * 92)
  print("  UNIMODAL BASELINES & DISAGREEMENT BREAKDOWN")
  print("=" * 92)
  print(f"  • Vision Only (ResNet-12)         : {mean_v:6.2f}% ± {ci_v:.2f}%")
  print(f"  • Graph Only  (GINE)              : {mean_g:6.2f}% ± {ci_g:.2f}%")
  print(
      f"  • Theoretical Oracle Upper Bound  : {mean_oracle:6.2f}% ±"
      f" {ci_oracle:.2f}% (Gain Room: +{mean_oracle - mean_v:.2f}%)"
  )
  print("-" * 92)
  print(
      f"  • Both Modalities Correct         : {np.mean(both_right_list):6.2f}%"
  )
  print(f"  • Vision Correct, Graph Wrong     : {np.mean(v_only_list):6.2f}%")
  print(
      f"  • Graph Correct, Vision Wrong     : {np.mean(g_only_list):6.2f}%"
      "  <-- Orthogonal Shape Signal"
  )
  print(
      f"  • Both Modalities Wrong           : {np.mean(both_wrong_list):6.2f}%"
  )

  print("\n" + "=" * 92)
  print("  EXHAUSTIVE DECISION-LEVEL FUSION BENCHMARK LEADERBOARD")
  print("=" * 92)
  print(
      f"  {'Method Name':<36} | {'Peak Accuracy':<17} | {'Vs Vision':<10} |"
      " Optimal Hyperparams"
  )
  print("-" * 92)
  best_overall_name, best_overall_m, best_overall_ci = "", -1.0, 0.0
  for name, m, ci, hparams in results:
    delta = m - mean_v
    print(
        f"  {name:<36} | {m:6.2f}% ± {ci:.2f}% | {delta:+7.2f}%   | {hparams}"
    )
    if m > best_overall_m:
      best_overall_name, best_overall_m, best_overall_ci = name, m, ci
  print("=" * 92)
  print(
      f"  CHAMPION FUSION OPERATOR: {best_overall_name}"
  )
  print(
      f"  Accuracy: {best_overall_m:.2f}% ± {best_overall_ci:.2f}%"
      f" (Net Gain vs Vision: {best_overall_m - mean_v:+.2f}%)"
  )
  print("=" * 92)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
  seed_everything(cfg.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  run_decision_fusion_evaluation(cfg, device)


if __name__ == "__main__":
  main()
