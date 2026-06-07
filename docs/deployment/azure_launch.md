# Azure H100 — Lasmoid Distillation Launch Guide

## Budget: $200 → ~90 hours of H100 NVL at $2.19/hr

## Step 1: Get Azure GPU Access (5 min)

1. Go to https://portal.azure.com
2. **IMPORTANT**: Upgrade from "Free Trial" to **"Pay-As-You-Go"**
   - This does NOT charge you extra — it just removes GPU restrictions
   - Your $200 credit still applies
   - Settings → Subscription → Upgrade
3. Search "Virtual Machines" → Create → Azure Virtual Machine

## Step 2: Create the H100 VM

- **Region**: East US or West US 2 (best H100 availability)
- **Image**: Ubuntu 22.04 LTS (Gen2)
- **Size**: `Standard_NC40ads_H100_v5` ($2.19/hr spot)
  - If not available, try `Standard_NC80adis_H200_v5` or `Standard_NC24ads_A100_v4`
- **Spot instance**: YES (saves 60-70% — this is how you get $2.19/hr)
  - Set eviction policy to "Stop / Deallocate"
- **Disk**: 256GB Premium SSD (for model weights)
- **Networking**: Allow SSH (port 22)

## Step 3: SSH In & Run Setup

```bash
ssh azureuser@YOUR_VM_IP

# Run the setup script
git clone https://github.com/Theory903/Lasmoid.git
cd Lasmoid
bash scripts/azure_setup.sh
```

## Step 4: Start Distillation (in tmux/screen!)

```bash
# Use screen so training survives SSH disconnect
screen -S distill

# Run distillation
python3 scripts/distill_gemma4.py

# Detach: Ctrl+A, then D
# Reattach later: screen -r distill
```

## Step 5: Monitor

```bash
# Check GPU usage
nvidia-smi

# Watch training logs (from another SSH session)
tail -f /path/to/output.log

# Check cost
# Azure Portal → Cost Management → Cost analysis
```

## Expected Timeline

| Phase | Steps | Time | Cost |
|-------|-------|------|------|
| Distillation | 15,000 | ~45h | $99 |
| SFT | 3,000 | ~5h | $11 |
| GRPO | 1,500 | ~3h | $7 |
| **Total** | | **~53h** | **$116** |

**Remaining budget after training: ~$84**

## Checkpoints

Saved every 1000 steps to `checkpoints/distill/`.
If the spot VM is evicted, just restart from the latest checkpoint:
```bash
# Edit distill_gemma4.py to set RESUME_FROM
python3 scripts/distill_gemma4.py
```

## Final Output

Model uploaded to: https://huggingface.co/Theory903/lasmoid-500m-distilled
- `model.safetensors` — student weights
- `config.json` — architecture config
- `README.md` — model card

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Quota exceeded" | Request quota increase via Azure Portal → Help + Support |
| VM evicted (spot) | Restart VM, run `screen -r distill` or re-run script (auto-resumes from checkpoint) |
| "Gemma-4 access denied" | Accept license at https://huggingface.co/google/gemma-4-26B-A4B then re-login |
| OOM on H100 | Reduce BATCH_SIZE from 8 to 4 in distill_gemma4.py |
