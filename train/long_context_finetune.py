"""
Lasmoid — long_context_finetune.py
==================================
Progressive length extension pipeline.
Finetunes the model by progressively expanding context length:
4K → 32K → 256K → 1M → 2M, adjusting YaRN RoPE factor and resizing KV caches.
"""

import os
import sys
import json
import argparse
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference"))

from inference.model import Lasmoid, ModelArgs, compute_loss
from inference.attention import precompute_freqs_cis
from inference.kv_cache import AdaptiveQuantizedKVCache
from optimizer import Muon


def safe_register_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        del module._buffers[name]
    if hasattr(module, name):
        delattr(module, name)
    module.register_buffer(name, tensor, persistent=False)


def extend_model_context_length(model: Lasmoid, new_seq_len: int, base_seq_len: int = 4096) -> None:
    """
    Dynamically extends the model's sequence length capacity:
      1. Updates max_seq_len and rope_factor in ModelArgs.
      2. Re-precomputes RoPE frequencies with new scaling.
      3. Re-allocates/updates KV caches and RoPE frequency caches in all attention layers.
    """
    print(f"[Extension] Extending context length: {model.args.max_seq_len} -> {new_seq_len}")
    
    # 1. Update arguments
    model.args.max_seq_len = new_seq_len
    model.args.rope_factor = float(new_seq_len) / float(base_seq_len)
    
    # 2. Re-precompute model-level RoPE frequencies
    freqs_seqlen = min(new_seq_len + 1024, 2097152 + 1024)
    new_freqs = precompute_freqs_cis(
        model.args.rope_head_dim,  # Fix: use rope_head_dim, not head_dim
        freqs_seqlen,
        original_seq_len=base_seq_len,
        base=model.args.rope_theta,
        factor=model.args.rope_factor,
        beta_fast=getattr(model.args, "beta_fast", 32),
        beta_slow=getattr(model.args, "beta_slow", 1),
    )
    
    # Update freqs_cis buffer on model
    device = model.freqs_cis.device
    safe_register_buffer(model, "freqs_cis", new_freqs.to(device))
    
    # 3. Update KV Caches and RoPE caches across all attention modules
    count = 0
    for name, module in model.named_modules():
        m_classname = module.__class__.__name__
        
        # A. Update layer-level RoPE frequency caches
        if hasattr(module, "freqs_cis") and getattr(module, "freqs_cis", None) is not None and hasattr(module, "rope_head_dim"):
            compress_ratio = getattr(module, "compress_ratio", None)
            if compress_ratio:
                layer_orig_seq_len = getattr(model.args, "original_seq_len", base_seq_len)
                layer_rope_theta = getattr(model.args, "compress_rope_theta", 40000.0)
            else:
                layer_orig_seq_len = 0
                layer_rope_theta = model.args.rope_theta
            
            new_layer_freqs = precompute_freqs_cis(
                module.rope_head_dim,
                freqs_seqlen,
                original_seq_len=layer_orig_seq_len,
                base=layer_rope_theta,
                factor=model.args.rope_factor,
                beta_fast=getattr(model.args, "beta_fast", 32),
                beta_slow=getattr(model.args, "beta_slow", 1),
            )
            safe_register_buffer(module, "freqs_cis", new_layer_freqs.to(device))
            
        elif m_classname == "HybridSlidingGlobal" and hasattr(module, "rope_cache"):
            r_cache = module.rope_cache
            dim = module.rope_head_dim
            
            new_local = precompute_freqs_cis(
                dim,
                freqs_seqlen,
                original_seq_len=0,
                base=module.local_base,
                factor=model.args.rope_factor,
                beta_fast=0,
                beta_slow=0,
            )
            safe_register_buffer(r_cache, "local_freqs", new_local.to(device))
            
            new_global = precompute_freqs_cis(
                dim,
                freqs_seqlen,
                original_seq_len=0,
                base=module.global_base,
                factor=model.args.rope_factor,
                beta_fast=0,
                beta_slow=0,
            )
            safe_register_buffer(r_cache, "global_freqs", new_global.to(device))
            
        # B. Update Layer-level KV Caches
        if m_classname == "HybridSlidingGlobal":
            # Resize global K/V caches
            max_batch = module.global_k_cache.shape[0]
            gcache = min(new_seq_len, module.global_key_size)
            ghdim = module.global_heads * module.head_dim
            
            new_gk = torch.zeros(max_batch, gcache, ghdim, device=module.global_k_cache.device, dtype=module.global_k_cache.dtype)
            new_gv = torch.zeros(max_batch, gcache, ghdim, device=module.global_v_cache.device, dtype=module.global_v_cache.dtype)
            
            safe_register_buffer(module, "global_k_cache", new_gk)
            safe_register_buffer(module, "global_v_cache", new_gv)
            module.global_write_ptr.zero_()
            count += 1
            
        elif hasattr(module, "kv_cache"):
            if isinstance(module.kv_cache, AdaptiveQuantizedKVCache):
                old_cache = module.kv_cache
                compress_ratio = getattr(module, "compress_ratio", None)
                window_size = getattr(module, "window_size", 0)
                if compress_ratio:
                    new_cache_size = window_size + max(1, new_seq_len // compress_ratio)
                else:
                    new_cache_size = new_seq_len
                
                new_cache = AdaptiveQuantizedKVCache(
                    max_batch=old_cache.max_batch,
                    max_seq=new_cache_size,
                    head_dim=old_cache.head_dim,
                    args=model.args,
                    dtype=old_cache.dtype,
                ).to(old_cache.device)
                module.kv_cache = new_cache
                count += 1
                
            elif isinstance(module.kv_cache, torch.Tensor):
                old_cache = module.kv_cache
                compress_ratio = getattr(module, "compress_ratio", None)
                window_size = getattr(module, "window_size", 0)
                if compress_ratio:
                    new_cache_size = window_size + max(1, new_seq_len // compress_ratio)
                else:
                    # If it has window_size but no compress_ratio, it is MLA attention sliding window.
                    # Standard MLA window size is fixed (args.window_size). So we don't resize it.
                    continue
                
                new_shape = (old_cache.shape[0], new_cache_size, old_cache.shape[2])
                new_cache = torch.zeros(new_shape, device=old_cache.device, dtype=old_cache.dtype)
                safe_register_buffer(module, "kv_cache", new_cache)
                
                if hasattr(module, "compressor") and module.compressor is not None:
                    module.compressor.kv_cache = new_cache[:, window_size:]
                count += 1
            
    print(f"[Extension] Successfully updated {count} attention KV caches.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stages", type=str, default="512,1024,2048", help="Comma-separated list of stage sequence lengths")
    parser.add_argument("--base_seq_len", type=int, default=512, help="Base sequence length (original RoPE scale)")
    parser.add_argument("--iters_per_stage", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/extension")
    args_cli = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Extension] Running on device: {device.upper()}")
    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    # Load Model
    with open(args_cli.config) as f:
        config_dict = json.load(f)
    from dataclasses import fields
    valid_fields = {f.name for f in fields(ModelArgs)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    model_args = ModelArgs(**filtered_config)

    lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enc = transformers.PreTrainedTokenizerFast.from_pretrained(lasmoid_dir, fix_mistral_regex=True)
    model_args.vocab_size = len(enc)

    model = Lasmoid(model_args).to(device)
    if os.path.exists(args_cli.ckpt_path):
        model.load_state_dict(torch.load(args_cli.ckpt_path, map_location=device), strict=False)
        print("[Extension] Initial checkpoint loaded.")

    # Parse stages
    stages = [int(s) for s in args_cli.stages.split(",")]
    print(f"[Extension] Target stages: {stages}")

    # Dummy training dataset fallback
    dataset_path = "input.txt"
    if not os.path.exists(dataset_path):
        parent_dataset = os.path.join(lasmoid_dir, "input.txt")
        if os.path.exists(parent_dataset):
            dataset_path = parent_dataset
        else:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                dataset_path,
            )
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    train_data = data[:int(0.9 * len(data))]

    def get_batch(split, seq_len):
        d = train_data
        ix = torch.randint(len(d) - seq_len - 1, (args_cli.batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        loss_mask = torch.ones_like(x, dtype=torch.float32)
        return x, y, loss_mask

    # Run stages
    for stage_idx, seq_len in enumerate(stages):
        print(f"\n==========================================")
        print(f" STAGE {stage_idx + 1}/{len(stages)}: seq_len={seq_len}")
        print(f"==========================================")
        
        # Extend context length and update weights/ RoPE freqs
        extend_model_context_length(model, seq_len, args_cli.base_seq_len)
        
        # Setup Optimizers for this stage
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if len(p.shape) == 2 and "emb" not in name and "head" not in name and "adj" not in name and "gate" not in name and "hc" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
                
        opt_muon = Muon(muon_params, lr=1e-3)
        opt_adamw = torch.optim.AdamW(adamw_params, lr=args_cli.learning_rate)

        model.train()
        
        # Train for specified iterations
        for step in range(args_cli.iters_per_stage):
            opt_muon.zero_grad()
            opt_adamw.zero_grad()
            
            xb, yb, loss_mask = get_batch("train", seq_len)
            
            logits, mtp_logits, _, _, routing_maps, _, adjs, event_probs = model(xb, xb)
            
            loss = compute_loss(
                logits,
                yb,
                routing_maps,
                [model.last_vq_loss],
                adjs,
                event_probs,
                loss_mask=loss_mask,
                moe_aux_loss=model.last_moe_loss,
                moe_aux_coeff=getattr(model.args, "moe_aux_coeff", 1.0),
                token_concept_loss=model.last_token_concept_loss,
                token_concept_coeff=getattr(model.args, "token_concept_loss_coeff", 0.05),
                ignore_index=getattr(model.args, "loss_ignore_index", -100),
            )
            
            if mtp_logits is not None:
                ce_loss_mtp = F.cross_entropy(
                    mtp_logits.view(-1, model.args.vocab_size),
                    yb[:, 1:].contiguous().view(-1),
                    ignore_index=getattr(model.args, "loss_ignore_index", -100),
                )
                loss = loss + getattr(model.args, "mtp_loss_coeff", 0.3) * ce_loss_mtp
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt_muon.step()
            opt_adamw.step()
            
            print(f"  Step {step:2d}/{args_cli.iters_per_stage} | Loss: {loss.item():.4f}")
            
        # Save checkpoint for this stage
        ckpt_path = os.path.join(args_cli.checkpoint_dir, f"lasmoid_stage_{seq_len}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Extension] Saved Stage {seq_len} checkpoint: {ckpt_path}")

    print("\n[Extension] All progressive length extension stages completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
