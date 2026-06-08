import torch
import torch.nn as nn
import json
import sys
from pathlib import Path

# Add Lasmoid repo path
sys.path.insert(0, "/Users/abhishekjha/CODE/NEXUS/Lasmoid")
sys.path.insert(0, "/Users/abhishekjha/CODE/NEXUS/Lasmoid/inference")

from config import ModelArgs
from lasmoid import Lasmoid

def test_checkpointing():
    # Load config
    cfg_path = Path("/Users/abhishekjha/CODE/NEXUS/Lasmoid/configs/model/config_gemma4_tiny_frontier.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    cfg["vocab_size"] = 32768
    cfg["max_seq_len"] = 1024
    cfg["max_batch_size"] = 4
    
    args = ModelArgs(**cfg)
    # Use float32 on CPU to avoid device-specific issues
    model = Lasmoid(args).to(torch.float32)
    model.train()
    model.gradient_checkpointing = True
    
    # Generate dummy input
    x = torch.randint(0, 32768, (2, 64))
    
    print("Running forward pass...")
    out = model(x, x)
    logits = out[0]
    loss = logits.float().mean()
    
    print("Running backward pass...")
    loss.backward()
    print("Backward pass completed successfully!")

if __name__ == "__main__":
    test_checkpointing()
