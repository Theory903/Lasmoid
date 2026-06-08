"""
Lasmoid — pretrain.py
=====================
NVFP4-aware pretraining script for Lasmoid.
Integrates simulated per-operator precision (QAT with STE) and WSD LR scheduling.
"""

import os
import sys
import json
import argparse
import urllib.request
import torch
import warnings

# Monkey-patch torch.bf16 for compatibility with older/custom PyTorch versions in transformers
if not hasattr(torch, "bf16"):
    torch.bf16 = torch.bfloat16

# Suppress incorrect regex warnings from tokenizers at runtime
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

import torch.nn as nn
import torch.nn.functional as F
import transformers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference"))

from inference.model import Lasmoid, ModelArgs, Linear, compute_loss
from inference.kernel import act_quant, fp4_act_quant, weight_dequant
try:
    from .optimizer import Muon
    from .scheduler import WSDScheduler
except ImportError:
    from optimizer import Muon
    from scheduler import WSDScheduler


class SimulatedQuantLinear(nn.Module):
    """
    QAT wrapper with Straight-Through Estimators (STE) to simulate
    NVFP4 / FP8 operations during training.
    """

    def __init__(self, original_linear: nn.Module, mode: str = "fp8", block_size: int = 128):
        super().__init__()
        self.original_linear = original_linear
        self.mode = mode
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.original_linear.weight
        bias = self.original_linear.bias
        
        if self.mode == "fp8":
            # Quantize weight
            q_w, s_w = act_quant(weight.contiguous(), self.block_size)
            dq_w = weight_dequant(q_w, s_w, self.block_size)
            w_sim = weight + (dq_w - weight).detach()
            
            # Quantize input
            q_x, s_x = act_quant(x.contiguous(), self.block_size)
            dq_x = weight_dequant(q_x, s_x, self.block_size)
            x_sim = x + (dq_x - x).detach()
            
        elif self.mode == "nvfp4":
            # Quantize weight to FP4
            q_w, s_w = fp4_act_quant(weight.contiguous(), block_size=32)
            dq_w = q_w
            if s_w is not None and s_w.ndim == 2:
                N, K = q_w.shape
                s_exp = s_w.float().repeat_interleave(32, dim=1)[:, :K]
                dq_w = q_w * s_exp
            w_sim = weight + (dq_w - weight).detach()
            
            # Quantize input to FP4
            q_x, s_x = fp4_act_quant(x.contiguous(), block_size=32)
            dq_x = q_x
            if s_x is not None:
                K = q_x.shape[-1]
                s_exp = s_x.float().repeat_interleave(32, dim=-1)[..., :K]
                dq_x = q_x * s_exp
            x_sim = x + (dq_x - x).detach()
            
        else:
            w_sim = weight
            x_sim = x
            
        return F.linear(x_sim.to(weight.dtype), w_sim, bias)


def apply_per_operator_precision(model: nn.Module, expert_dtype: str = "nvfp4") -> None:
    """
    Traverses the model and wraps MoE experts in SimulatedQuantLinear:
      - Routed experts -> NVFP4 (if configured)
      - Shared expert  -> FP8
    """
    for module in model.modules():
        # Check if this is a DeepSeekMoE block
        if hasattr(module, "experts") and isinstance(module.experts, nn.ModuleList):
            # Routed experts
            for expert in module.experts:
                if expert_dtype == "nvfp4":
                    expert.w1 = SimulatedQuantLinear(expert.w1, mode="nvfp4")
                    expert.w2 = SimulatedQuantLinear(expert.w2, mode="nvfp4")
                    expert.w3 = SimulatedQuantLinear(expert.w3, mode="nvfp4")
                elif expert_dtype == "fp8":
                    expert.w1 = SimulatedQuantLinear(expert.w1, mode="fp8")
                    expert.w2 = SimulatedQuantLinear(expert.w2, mode="fp8")
                    expert.w3 = SimulatedQuantLinear(expert.w3, mode="fp8")
            
            # Shared expert
            shared = module.shared
            shared.w1 = SimulatedQuantLinear(shared.w1, mode="fp8")
            shared.w2 = SimulatedQuantLinear(shared.w2, mode="fp8")
            shared.w3 = SimulatedQuantLinear(shared.w3, mode="fp8")


def pretrain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=10, help="Typically 2%% of pretraining steps")
    parser.add_argument("--stable_steps", type=int, default=80, help="Typically 90%% of pretraining steps")
    parser.add_argument("--decay_steps", type=int, default=10, help="Typically 8%% of pretraining steps")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--expert_dtype", type=str, default="nvfp4", choices=["bf16", "fp8", "nvfp4"])
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face repo ID to upload checkpoints")
    args_cli = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pretrain] Running on device: {device.upper()}")
    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    # Initialize Hugging Face Uploader
    hf_uploader = None
    if args_cli.hf_repo:
        from hf_uploader import HFAnyUploader
        hf_uploader = HFAnyUploader(repo_id=args_cli.hf_repo)

    # 1. Load config
    lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(lasmoid_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    from dataclasses import fields
    valid_fields = {f.name for f in fields(ModelArgs)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    model_args = ModelArgs(**filtered_config)

    # Load tokenizer and fallback dataset
    try:
        enc = transformers.PreTrainedTokenizerFast.from_pretrained(lasmoid_dir, fix_mistral_regex=True)
    except Exception:
        enc = transformers.PreTrainedTokenizerFast.from_pretrained(lasmoid_dir)
    model_args.vocab_size = len(enc)

    dataset_path = "input.txt"
    if not os.path.exists(dataset_path):
        parent_dataset = os.path.join(lasmoid_dir, "input.txt")
        if os.path.exists(parent_dataset):
            dataset_path = parent_dataset
        else:
            print("[Pretrain] Downloading Tiny Shakespeare dataset...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                dataset_path,
            )

    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    train_data = data[:int(0.9 * len(data))]
    val_data = data[int(0.9 * len(data)):]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - model_args.max_seq_len - 1, (args_cli.batch_size,))
        x = torch.stack([d[i : i + model_args.max_seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + model_args.max_seq_len + 1] for i in ix]).to(device)
        loss_mask = torch.ones_like(x, dtype=torch.float32)
        return x, y, loss_mask

    # Initialize model
    model = Lasmoid(model_args).to(device)
    
    # 2. Apply simulated per-operator precision (QAT with STE)
    print(f"[Pretrain] Applying per-operator precision: MoE experts -> {args_cli.expert_dtype.upper()}, shared expert -> FP8")
    apply_per_operator_precision(model, args_cli.expert_dtype)
    
    model.train()

    # 3. Partition parameters for Muon/AdamW
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if len(p.shape) == 2 and "emb" not in name and "head" not in name and "adj" not in name and "gate" not in name and "hc" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=2e-3)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=args_cli.learning_rate)

    # 4. Initialize WSD Scheduler
    scheduler = WSDScheduler(
        optimizers=[opt_muon, opt_adamw],
        warmup_steps=args_cli.warmup_steps,
        stable_steps=args_cli.stable_steps,
        decay_steps=args_cli.decay_steps,
        base_lrs=[[2e-3], [args_cli.learning_rate]],
        min_lr_ratio=0.1,
    )

    print("[Pretrain] Starting pretraining loop...")
    for step in range(args_cli.max_iters):
        # Update LR via WSD scheduler
        scheduler.step(step)

        opt_muon.zero_grad()
        opt_adamw.zero_grad()

        total_step_loss = 0.0
        for micro in range(args_cli.grad_accum):
            xb, yb, loss_mask = get_batch("train")
            
            logits, mtp_logits, _, _, routing_maps, _, adjs, event_probs = model(xb, xb)
            
            main_loss = compute_loss(
                logits,
                yb,
                routing_maps,
                [model.last_vq_loss],
                adjs,
                event_probs,
                loss_mask=loss_mask,
                moe_aux_loss=model.last_moe_loss,
                moe_aux_coeff=getattr(model_args, "moe_aux_coeff", 1.0),
                token_concept_loss=model.last_token_concept_loss,
                token_concept_coeff=getattr(model_args, "token_concept_loss_coeff", 0.05),
                ignore_index=getattr(model_args, "loss_ignore_index", -100),
            )

            # MTP loss
            ce_loss_mtp = torch.tensor(0.0, device=device)
            if mtp_logits is not None:
                ce_loss_mtp = F.cross_entropy(
                    mtp_logits.view(-1, model_args.vocab_size),
                    yb[:, 1:].contiguous().view(-1),
                    ignore_index=getattr(model_args, "loss_ignore_index", -100),
                )
                # MTP loss is now folded into main_loss via mtp_loss param
                main_loss = main_loss + getattr(model_args, "mtp_loss_coeff", 0.3) * ce_loss_mtp

            pred_coeff = getattr(model_args, "predictive_coding_coeff", 0.01)
            loss = main_loss + pred_coeff * model.last_pred_loss
            loss = loss / args_cli.grad_accum
            loss.backward()
            total_step_loss += loss.item() * args_cli.grad_accum

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt_muon.step()
        opt_adamw.step()

        if step % 10 == 0:
            print(f"Step {step:4d} | Pretrain Loss: {total_step_loss:.4f} | LR Multiplier: {scheduler.step(step):.4f}")

        # Checkpoint Saving
        if step > 0 and step % args_cli.save_interval == 0:
            ckpt_path = os.path.join(args_cli.checkpoint_dir, f"lasmoid_pretrain_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Pretrain] Checkpoint saved: {ckpt_path}")
            if hf_uploader:
                hf_uploader.upload_file_async(
                    file_path=ckpt_path,
                    path_in_repo=f"checkpoints/lasmoid_pretrain_{step}.pt",
                    commit_message=f"Pretrain checkpoint at step {step}"
                )

    # Save final model
    final_path = os.path.join(args_cli.checkpoint_dir, "lasmoid_pretrain_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[Pretrain] Pretraining completed successfully. Weights saved: {final_path}")
    if hf_uploader:
        hf_uploader.upload_file_async(
            file_path=final_path,
            path_in_repo="lasmoid_pretrain_final.pt",
            commit_message="Final pretrained weights"
        )
        hf_uploader.wait_for_uploads()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pretrain()
