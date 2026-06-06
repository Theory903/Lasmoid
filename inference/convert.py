"""
Lasmoid — convert.py
====================
Weight converter and dtype conversion utility.
Casts weights between fp32, bf16, fp16, and FP8 formats.
"""

import os
import sys
import torch
from argparse import ArgumentParser

def convert_checkpoint(input_path: str, output_path: str, target_dtype: str):
    print(f"Loading checkpoint from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: input path {input_path} does not exist.")
        sys.exit(1)
        
    sd = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Extract state dict if it's nested
    is_nested = False
    if isinstance(sd, dict) and "model_state_dict" in sd:
        state_dict = sd["model_state_dict"]
        is_nested = True
    else:
        state_dict = sd
        
    print(f"Casting weights to target dtype: {target_dtype}")
    
    # Select PyTorch dtype
    if target_dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif target_dtype == "fp16":
        torch_dtype = torch.float16
    elif target_dtype == "fp32":
        torch_dtype = torch.float32
    elif target_dtype == "fp8":
        # FP8 weights are stored as torch.float8_e4m3fn in PyTorch >= 2.1
        if hasattr(torch, "float8_e4m3fn"):
            torch_dtype = torch.float8_e4m3fn
        else:
            print("WARNING: torch.float8_e4m3fn not available. Casting to bfloat16 instead.")
            torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown target dtype: {target_dtype}")
        
    new_state_dict = {}
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            # Only cast floating point tensors
            if param.is_floating_point():
                new_state_dict[key] = param.to(torch_dtype)
            else:
                new_state_dict[key] = param
        else:
            new_state_dict[key] = param
            
    if is_nested:
        sd["model_state_dict"] = new_state_dict
        if "model_args" in sd:
            sd["model_args"].dtype = target_dtype
        torch.save(sd, output_path)
    else:
        torch.save(new_state_dict, output_path)
        
    print(f"Successfully converted checkpoint saved to: {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Lasmoid Weight Converter")
    parser.add_argument("--input", type=str, required=True, help="Path to input checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Path to save converted checkpoint")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32", "fp8"], default="bf16", help="Target data type")
    args = parser.parse_args()
    
    convert_checkpoint(args.input, args.output, args.dtype)
