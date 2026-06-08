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

def test_compilation():
    cfg_path = Path("/Users/abhishekjha/CODE/NEXUS/Lasmoid/configs/model/config_gemma4_tiny_frontier.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    cfg["vocab_size"] = 32768
    cfg["max_seq_len"] = 128
    cfg["max_batch_size"] = 2
    # Use fp16 in model args
    cfg["dtype"] = "fp16"
    
    args = ModelArgs(**cfg)
    print("Model config successfully validated!")
    
    print("Initializing model on CPU...")
    # Initialize in float32 for CPU compatibility of standard operators, or float16 if supported
    model = Lasmoid(args).to(dtype=torch.float32)
    model.train()
    
    x = torch.randint(0, 32768, (2, 128))
    
    print("Running forward pass...")
    # Perform forward pass in float32 to verify shape consistency and logic on CPU
    out = model(x, x)
    print("Forward pass outputs shape:", out[0].shape)
    
    print("SUCCESS: Model compiles and runs a forward step on CPU without issue!")

if __name__ == "__main__":
    test_compilation()
