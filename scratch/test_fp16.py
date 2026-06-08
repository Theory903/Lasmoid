import torch
import torch.nn as nn
import json
import sys
from pathlib import Path

# Add Lasmoid repo paths
sys.path.insert(0, "/Users/abhishekjha/CODE/NEXUS/Lasmoid")
sys.path.insert(0, "/Users/abhishekjha/CODE/NEXUS/Lasmoid/inference")

from config import ModelArgs
from lasmoid import Lasmoid

def test_fp16():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
        
    cfg_path = Path("/Users/abhishekjha/CODE/NEXUS/Lasmoid/configs/model/config_gemma4_tiny_frontier.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    cfg["vocab_size"] = 32768
    cfg["max_seq_len"] = 1024
    cfg["max_batch_size"] = 4
    
    args = ModelArgs(**cfg)
    device = torch.device("cuda:0")
    
    print("Initializing model...")
    model = Lasmoid(args).to(device=device, dtype=torch.float16)
    model.train()
    model.gradient_checkpointing = True
    
    x = torch.randint(0, 32768, (2, 1024), device=device)
    
    print("Testing forward pass in fp16...")
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x, x)
    
    print("Testing backward pass...")
    logits = out[0]
    loss = logits.float().mean()
    loss.backward()
    
    print("SUCCESS: Model forward & backward completed successfully in fp16!")

if __name__ == "__main__":
    test_fp16()
