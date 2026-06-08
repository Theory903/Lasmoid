"""
Lasmoid — prepare_data.py
========================================================================
Utility script to download, format, tokenize, and pack datasets for
pretraining, SteerLM SFT, and GRPO Reinforcement Learning.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

# Add root folder to sys.path
lasmoid_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lasmoid_dir)

try:
    import datasets
except ImportError:
    print("Error: The 'datasets' library is required to run this script.")
    print("Please install it using: pip install datasets tqdm")
    sys.exit(1)

import transformers

def setup_tokenizer(tokenizer_path):
    print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Standard AutoTokenizer failed to load ({e}). Falling back to PreTrainedTokenizerFast...")
        try:
            tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path, fix_mistral_regex=True)
        except Exception as e_fast:
            try:
                tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            except Exception as e_fast2:
                print(f"Error: Failed to load tokenizer from '{tokenizer_path}'.")
                print(f"Detailed error: {e_fast2}")
                print("\nIf you are loading a gated Hugging Face model (such as Gemma), make sure:")
                print("1. You have accepted the license terms on Hugging Face model page.")
                print("2. You are logged in using 'huggingface-cli login' or have set 'HF_TOKEN' environment variable.")
                sys.exit(1)

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def format_pretrain(example, tokenizer, text_field="text"):
    text = example[text_field] if text_field in example else ""
    if not text:
        return []
    # Tokenize and append EOS
    tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
    return tokens

def format_steerlm(example, tokenizer):
    """
    Formats HelpSteer prompt with attributes:
    helpfulness, correctness (rigor), complexity, creativity.
    """
    prompt = example.get("prompt", "").strip()
    response = example.get("response", "").strip()
    
    # Map HelpSteer attributes (HelpSteer scores are 0-4 or 0-5)
    helpfulness = int(example.get("helpfulness", 3))
    correctness = int(example.get("correctness", 3))  # maps to scientific_rigor
    complexity = int(example.get("complexity", 2))
    # Creativity isn't always marked in HelpSteer, default to 2
    creativity = 2
    
    # SteerLM formatting template
    steer_prefix = (
        f"<|im_start|>system\n"
        f"You are a helpful assistant. Rating - Helpfulness: {helpfulness}/4 Complexity: {complexity}/4 Scientific Rigor: {correctness}/4 Creativity: {creativity}/4\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    prefix_tokens = tokenizer.encode(steer_prefix)
    response_tokens = tokenizer.encode(response) + [tokenizer.eos_token_id]
    
    # Full sequence
    full_tokens = prefix_tokens + response_tokens
    
    # We return the tokenized sequence and the prompt length so the training loop can mask the loss
    return {
        "tokens": full_tokens,
        "split_idx": len(prefix_tokens)
    }

def format_grpo(example, tokenizer):
    """
    Formats GSM8K or NuminaMath reasoning prompts for GRPO alignment.
    Requires `<think>...</think>` structural formatting.
    """
    question = example.get("question", "").strip()
    answer = example.get("answer", "").strip()
    
    prompt = (
        f"<|im_start|>user\n{question}\n"
        f"Show your step-by-step thinking inside <think>...</think> and provide the final answer inside <answer>...</answer>.\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    
    prompt_tokens = tokenizer.encode(prompt)
    
    # If we are doing SFT training of thinking traces first, we construct the reference response
    # For GRPO reinforcement learning, the model is fed the prompt_tokens, and completions are generated.
    # Here we save both prompt and full formatted reference.
    reference = f"{answer}"
    if "<think>" not in answer:
        # Wrap ground truth answer for SFT initialization if needed
        reference = f"Reasoning steps verified.\n</think>\n<answer>\n{answer}\n</answer>\n"
        
    reference_tokens = tokenizer.encode(reference) + [tokenizer.eos_token_id]
    
    return {
        "tokens": prompt_tokens + reference_tokens,
        "split_idx": len(prompt_tokens)
    }

def process_dataset():
    parser = argparse.ArgumentParser(description="Prepare datasets for training Lasmoid.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="tinystories",
        choices=["tinystories", "fineweb-edu", "helpsteer", "gsm8k"],
        help="Dataset name to prepare."
    )
    parser.add_argument(
        "--max_seq_len", 
        type=int, 
        default=512,
        help="Maximum sequence length of the model (must match config max_seq_len)."
    )
    parser.add_argument(
        "--split_ratio", 
        type=float, 
        default=0.9,
        help="Ratio of training to validation split."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process (useful for testing)."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path or HuggingFace identifier for the tokenizer (e.g. google/gemma-4-12B)."
    )
    args = parser.parse_args()
    
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else lasmoid_dir
    tokenizer = setup_tokenizer(tokenizer_path)
    
    print(f"\nPreparing dataset: {args.dataset.upper()}")
    
    # 1. Load Dataset from HF
    if args.dataset == "tinystories":
        print("Loading roneneldan/TinyStories...")
        dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
        text_field = "text"
        mode = "pretrain"
    elif args.dataset == "fineweb-edu":
        print("Loading HuggingFaceTB/fineweb-edu (10BT sample subset)...")
        dataset = datasets.load_dataset("HuggingFaceTB/fineweb-edu", name="sample-10BT", split="train")
        text_field = "text"
        mode = "pretrain"
    elif args.dataset == "helpsteer":
        print("Loading nvidia/HelpSteer...")
        # HelpSteer has train/validation splits
        dataset = datasets.load_dataset("nvidia/HelpSteer", split="train")
        mode = "steerlm"
    elif args.dataset == "gsm8k":
        print("Loading openai/gsm8k...")
        dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
        mode = "grpo"
        
    if args.limit:
        print(f"Limiting dataset to {args.limit} examples...")
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        
    print(f"Total downloaded examples: {len(dataset):,}")
    
    # Split train/val
    num_train = int(len(dataset) * args.split_ratio)
    train_slice = dataset.select(range(num_train))
    val_slice = dataset.select(range(num_train, len(dataset)))
    
    print(f"Train split size: {len(train_slice):,}")
    print(f"Val split size: {len(val_slice):,}")
    
    for split_name, dataset_split in [("train", train_slice), ("val", val_slice)]:
        print(f"\nProcessing {split_name} split...")
        
        all_tokens = []
        metadata = [] # List of dicts storing sequence bounds and split indices for SFT/GRPO loss masking
        
        if mode == "pretrain":
            # For pretraining, we pack tokens sequentially to reduce padding overhead
            buffer = []
            for item in tqdm(dataset_split, desc=f"Tokenizing {split_name}"):
                tokens = format_pretrain(item, tokenizer, text_field)
                buffer.extend(tokens)
                
                # Yield max_seq_len blocks
                while len(buffer) >= args.max_seq_len:
                    chunk = buffer[:args.max_seq_len]
                    all_tokens.extend(chunk)
                    # For pretraining, loss is calculated on all tokens, split_idx is 0
                    metadata.append({"length": args.max_seq_len, "split_idx": 0})
                    buffer = buffer[args.max_seq_len:]
        else:
            # For SFT and GRPO, we process example-by-example and pad/clip to max_seq_len
            for item in tqdm(dataset_split, desc=f"Tokenizing {split_name}"):
                if mode == "steerlm":
                    res = format_steerlm(item, tokenizer)
                else: # grpo
                    res = format_grpo(item, tokenizer)
                    
                tokens = res["tokens"]
                split_idx = res["split_idx"]
                
                # Clip to max_seq_len if too long
                if len(tokens) > args.max_seq_len:
                    tokens = tokens[:args.max_seq_len]
                    split_idx = min(split_idx, args.max_seq_len - 1)
                else:
                    # Pad with EOS tokens
                    padding_len = args.max_seq_len - len(tokens)
                    tokens = tokens + [tokenizer.eos_token_id] * padding_len
                    
                all_tokens.extend(tokens)
                metadata.append({"length": args.max_seq_len, "split_idx": split_idx})
                
        # Convert to numpy uint32 array for compact storage
        tokens_np = np.array(all_tokens, dtype=np.uint32)
        
        # ── Validate batch shapes match configured max_seq_len ──
        num_seqs = len(metadata)
        expected_total_tokens = num_seqs * args.max_seq_len
        if len(all_tokens) != expected_total_tokens:
            print(
                f"WARNING: Token count mismatch! Expected {expected_total_tokens} "
                f"({num_seqs} sequences × {args.max_seq_len} seq_len), "
                f"got {len(all_tokens)}. Data may be truncated or misaligned."
            )
        for i, entry in enumerate(metadata):
            if entry["length"] != args.max_seq_len:
                raise ValueError(
                    f"Batch shape error: metadata entry {i} has length "
                    f"{entry['length']}, expected {args.max_seq_len} (max_seq_len)."
                )
        
        # Save files
        out_bin_path = os.path.join(lasmoid_dir, "train", f"{args.dataset}_{split_name}.bin")
        out_meta_path = os.path.join(lasmoid_dir, "train", f"{args.dataset}_{split_name}_meta.json")
        
        print(f"Saving {len(tokens_np):,} tokens to {out_bin_path}...")
        with open(out_bin_path, "wb") as f:
            f.write(tokens_np.tobytes())
            
        print(f"Saving segment metadata to {out_meta_path}...")
        with open(out_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    print(f"\nSuccess! Dataset '{args.dataset}' prepared and saved inside train/ folder.")
    print("Files ready for direct binary reading in the training loop.")

if __name__ == "__main__":
    process_dataset()
