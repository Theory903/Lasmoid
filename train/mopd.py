"""
Lasmoid — mopd.py
================
Multi-teacher on-policy distillation (MOPD) runner and KL divergence loss.
Distills knowledge from two teacher models (e.g., DeepSeek and Gemma-4) into
the Lasmoid student model.
"""

import os
import sys
import argparse
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference"))

from inference.model import Lasmoid, ModelArgs, Linear
try:
    from .optimizer import Muon
except ImportError:
    from optimizer import Muon


def compute_mopd_loss(
    student_logits: torch.Tensor,
    teacher1_logits: torch.Tensor,
    teacher2_logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    temp: float = 2.0,
) -> torch.Tensor:
    """
    Computes MOPD loss:
      Loss = (1 - alpha) * CE(Student, Targets) +
             alpha * (Temp^2) * [0.5 * KL(T1 || S) + 0.5 * KL(T2 || S)]
    """
    vocab_size = student_logits.size(-1)
    
    # 1. Standard Cross-Entropy Loss
    loss_mask_flat = loss_mask.view(-1) if loss_mask is not None else None
    
    ce_loss_flat = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        targets.view(-1),
        reduction="none"
    )
    if loss_mask_flat is not None:
        ce_loss = (ce_loss_flat * loss_mask_flat).sum() / (loss_mask_flat.sum() + 1e-8)
    else:
        ce_loss = ce_loss_flat.mean()
        
    # 2. Soft Distillation KL Loss
    log_probs_s = F.log_softmax(student_logits / temp, dim=-1)
    probs_t1 = F.softmax(teacher1_logits / temp, dim=-1)
    probs_t2 = F.softmax(teacher2_logits / temp, dim=-1)
    
    # kl_div computes: target * (log(target) - input). To compute KL(T || S):
    kl_t1_flat = F.kl_div(log_probs_s, probs_t1, reduction="none").sum(dim=-1).view(-1)
    kl_t2_flat = F.kl_div(log_probs_s, probs_t2, reduction="none").sum(dim=-1).view(-1)
    
    kl_loss_flat = 0.5 * (kl_t1_flat + kl_t2_flat)
    
    if loss_mask_flat is not None:
        kl_loss = (kl_loss_flat * loss_mask_flat).sum() / (loss_mask_flat.sum() + 1e-8)
    else:
        kl_loss = kl_loss_flat.mean()
        
    # Combine losses (scale KL by temp^2)
    total_loss = (1.0 - alpha) * ce_loss + alpha * (temp ** 2) * kl_loss
    return total_loss


def load_teacher(config_path: str, ckpt_path: str, device: str) -> nn.Module:
    """Helper to load a teacher model checkpoint or return a dummy if missing."""
    import json
    from dataclasses import fields
    
    if os.path.exists(config_path) and os.path.exists(ckpt_path):
        with open(config_path) as f:
            cfg_dict = json.load(f)
        valid_fields = {f.name for f in fields(ModelArgs)}
        filtered_config = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        args = ModelArgs(**filtered_config)
        model = Lasmoid(args).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        return model
        
    # Fallback/Dummy teacher for testing/local verification
    args = ModelArgs()
    args.vocab_size = 1000
    args.dim = 64
    args.n_layers = 1
    args.max_seq_len = 32
    args.max_batch_size = 2
    args.n_heads = 4
    args.head_dim = 16
    args.rope_head_dim = 8
    args.q_lora_rank = 16
    args.o_lora_rank = 16
    
    model = Lasmoid(args).to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument("--student_config", type=str, required=True)
    parser.add_argument("--teacher1_ckpt", type=str, default="")
    parser.add_argument("--teacher1_config", type=str, default="")
    parser.add_argument("--teacher2_ckpt", type=str, default="")
    parser.add_argument("--teacher2_config", type=str, default="")
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5, help="Distillation weight")
    parser.add_argument("--temp", type=float, default=2.0, help="KL temperature")
    args_cli = parser.parse_args()
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MOPD] Running on device: {device.upper()}")
    
    # 1. Load Student Model
    import json
    from dataclasses import fields
    with open(args_cli.student_config) as f:
        cfg_dict = json.load(f)
    valid_fields = {f.name for f in fields(ModelArgs)}
    filtered_config = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    student_args = ModelArgs(**filtered_config)
    student = Lasmoid(student_args).to(device)
    
    if os.path.exists(args_cli.student_ckpt):
        student.load_state_dict(torch.load(args_cli.student_ckpt, map_location=device), strict=False)
        print("[MOPD] Student checkpoint loaded.")
    student.train()
    
    # 2. Load Teachers
    print("[MOPD] Loading teacher models...")
    teacher1 = load_teacher(args_cli.teacher1_config, args_cli.teacher1_ckpt, device)
    teacher2 = load_teacher(args_cli.teacher2_config, args_cli.teacher2_ckpt, device)
    
    # 3. Setup Optimizers
    muon_params, adamw_params = [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if len(p.shape) == 2 and "emb" not in name and "head" not in name and "adj" not in name and "gate" not in name and "hc" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
            
    opt_muon = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args_cli.learning_rate)
    
    # 4. Dummy data for demonstration/run checking
    vocab_size = student_args.vocab_size
    seq_len = student_args.max_seq_len
    
    print("[MOPD] Starting distillation loop...")
    for step in range(args_cli.max_iters):
        # Generate mock input
        x = torch.randint(0, vocab_size, (args_cli.batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (args_cli.batch_size, seq_len), device=device)
        loss_mask = torch.ones((args_cli.batch_size, seq_len), device=device)
        
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        
        # On-policy generation of logits
        with torch.no_grad():
            t1_logits, *_ = teacher1(x, x)
            t2_logits, *_ = teacher2(x, x)
            
        student_logits, *_ = student(x, x)
        
        loss = compute_mopd_loss(
            student_logits,
            t1_logits,
            t2_logits,
            targets,
            loss_mask=loss_mask,
            alpha=args_cli.alpha,
            temp=args_cli.temp,
        )
        
        loss.backward()
        
        # Gradient clip
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        opt_muon.step()
        opt_adamw.step()
        
        if step % 2 == 0:
            print(f"Step {step:3d} | Distillation Loss: {loss.item():.4f}")
            
    print("[MOPD] Distillation completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
