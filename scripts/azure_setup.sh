#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Azure H100 Setup Script for Lasmoid Distillation
# Run this ONCE after SSH-ing into your Azure VM.
# ═══════════════════════════════════════════════════════════════
set -e

echo "╔═══════════════════════════════════════════════╗"
echo "║  Lasmoid Distillation — Azure H100 Setup     ║"
echo "╚═══════════════════════════════════════════════╝"

# 1. System deps
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-pip screen tmux

# 2. Clone repo
cd /home/$USER
if [ ! -d "Lasmoid" ]; then
    git clone https://github.com/Theory903/Lasmoid.git
fi
cd Lasmoid

# 3. Install Python deps
pip install -q torch transformers safetensors tiktoken numpy tqdm \
    datasets huggingface_hub accelerate

# 4. Login to HuggingFace (needed for Gemma-4 gated model)
echo ""
echo "⚠️  You need HuggingFace access to google/gemma-4-26B-A4B"
echo "   1. Accept the license at: https://huggingface.co/google/gemma-4-26B-A4B"
echo "   2. Run: huggingface-cli login"
echo ""
huggingface-cli login

# 5. Verify GPU
python3 -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
print(f'PyTorch: {torch.__version__}')
assert torch.cuda.is_available(), 'NO GPU!'
print('✅ Ready to train')
"

echo ""
echo "✅ Setup complete. Run distillation with:"
echo "   python3 scripts/distill_gemma4.py"
echo ""
echo "💡 Use 'screen' or 'tmux' so training survives SSH disconnect:"
echo "   screen -S train"
echo "   python3 scripts/distill_gemma4.py"
echo "   (Ctrl+A, D to detach)"
