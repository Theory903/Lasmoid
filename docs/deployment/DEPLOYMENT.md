# Lasmoid Deployment Guide

Cloud and local deployment reference.

---

## Table of Contents

1. [Kaggle (Free)](#kaggle-free)
2. [Azure VM](#azure-vm)
3. [Local GPU](#local-gpu)
4. [Model Export](#model-export)
5. [HuggingFace Hub](#huggingface-hub)

---

## Kaggle (Free)

**Best option for training/distillation with zero cost.**

| Resource | Spec |
|----------|------|
| GPU | 2× Tesla T4 (16 GB each = 32 GB total) |
| RAM | 29 GB |
| Session limit | 12 hrs / week (interactive) or 9 hrs (notebook) |
| Cost | **Free** |

### Steps

1. Import notebook from GitHub:
   - `notebooks/distillation/lasmoid_distill_gemma4_12B_kaggle.ipynb`
2. Settings → Accelerator → **GPU T4 x2**
3. Add-ons → Secrets → `HF_TOKEN`
4. Run All

Outputs saved to `/kaggle/working/checkpoints/`. Download before session ends!

---

## Azure VM

### One-command deploy (auto-selects A100 → T4 fallback)

```bash
export HF_TOKEN=hf_...
chmod +x scripts/azure_auto_deploy.sh
./scripts/azure_auto_deploy.sh
```

What it does:
1. Logs into Azure CLI
2. Creates resource group `lasmoid-train-rg` in East US
3. Tries A100 spot ($0.82/hr) → falls back to T4 ($0.22/hr)
4. SSHs in, installs NVIDIA + PyTorch
5. Clones repo, logs into HuggingFace
6. Starts training in `tmux` session (survives SSH disconnect)

### Manual VM Setup

```bash
# Recommended: Standard_NC24ads_A100_v4 (A100 80GB)
# Budget:      Standard_NC4as_T4_v3     (T4 16GB)

az vm create \
    --resource-group lasmoid-train-rg \
    --name lasmoid-gpu \
    --image Ubuntu2204 \
    --size Standard_NC4as_T4_v3 \
    --admin-username azureuser \
    --generate-ssh-keys \
    --priority Spot \
    --eviction-policy Deallocate

# SSH in
ssh azureuser@<VM_IP>

# Setup
curl -fsSL https://raw.githubusercontent.com/Theory903/Lasmoid/main/scripts/azure_setup.sh | bash
```

### Monitor training

```bash
ssh azureuser@<VM_IP> 'tmux attach -t train'   # live view
ssh azureuser@<VM_IP> 'tail -20 ~/train.log'   # quick check
```

### ⚠️ STOP VM WHEN DONE (saves money)

```bash
az group delete --name lasmoid-train-rg --yes --no-wait
```

---

## Local GPU

```bash
# Clone
git clone https://github.com/Theory903/Lasmoid
cd Lasmoid

# Install
pip install -r inference/requirements.txt

# Train 100M
python train/train.py \
    --config configs/model/config_100m.json \
    --data   datasets/input.txt \
    --out    checkpoints/
```

---

## Model Export

### PyTorch → SafeTensors

```python
from safetensors.torch import save_file
import torch

ckpt = torch.load("checkpoints/current/lasmoid_final.pt")
save_file(ckpt["model_state_dict"], "lasmoid_100m.safetensors")
```

### Convert to HuggingFace format

```bash
python inference/convert.py \
    --checkpoint checkpoints/current/ \
    --config     configs/model/config_100m.json \
    --output     hf_export/
```

---

## HuggingFace Hub

### Upload

```python
from huggingface_hub import HfApi, create_repo
from safetensors.torch import save_file
import shutil, torch

REPO = "Theory903/lasmoid-100m"

api = HfApi()
create_repo(REPO, exist_ok=True)

# Save weights
ckpt = torch.load("checkpoints/current/lasmoid_final.pt")
save_file(ckpt["model_state_dict"], "upload/model.safetensors")
shutil.copy("configs/model/config_100m.json", "upload/config.json")
shutil.copy("tokenizer.json", "upload/")
shutil.copy("configs/tokenizer/tokenizer_config.json", "upload/")
shutil.copy("configs/generation_config.json", "upload/")

api.upload_folder(
    folder_path="upload/",
    repo_id=REPO,
    commit_message="Lasmoid-100M checkpoint",
)
```

### Download

```python
from huggingface_hub import snapshot_download
snapshot_download("Theory903/lasmoid-100m", local_dir="checkpoints/hf/")
```

---

## Config Path Updates

> After the file system reorganization, update any hardcoded paths:

| Old path | New path |
|----------|----------|
| `config.json` | `configs/model/config_default.json` |
| `config_100m.json` | `configs/model/config_100m.json` |
| `generation_config.json` | `configs/generation_config.json` |
| `tokenizer_config.json` | `configs/tokenizer/tokenizer_config.json` |
| `lasmoid_distill_gemma4_12B_kaggle.ipynb` | `notebooks/distillation/lasmoid_distill_gemma4_12B_kaggle.ipynb` |
| `input.txt` | `datasets/input.txt` |
