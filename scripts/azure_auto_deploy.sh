#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# LASMOID — Fully Automated Azure GPU Deploy + Train
# ═══════════════════════════════════════════════════════════════
# 
# One command to:
#   1. Create Azure Resource Group
#   2. Launch GPU VM (tries A100 spot → falls back to T4)
#   3. Install deps, clone repo
#   4. Start distillation training inside tmux
#
# USAGE:
#   chmod +x scripts/azure_auto_deploy.sh
#   ./scripts/azure_auto_deploy.sh
#
# PREREQUISITES:
#   - Azure CLI installed: brew install azure-cli (macOS) or curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
#   - Logged in: az login
#   - Credits available on your subscription
#   - HuggingFace token exported: export HF_TOKEN=hf_xxxxx
#
# ═══════════════════════════════════════════════════════════════
set -e

# ── CONFIG ──────────────────────────────────────────────────────
RESOURCE_GROUP="lasmoid-train-rg"
LOCATION="eastus"                          # Good A100/T4 availability
VM_NAME="lasmoid-gpu"
ADMIN_USER="azureuser"
SSH_KEY="~/.ssh/id_rsa.pub"                # Your SSH public key
GITHUB_REPO="https://github.com/Theory903/Lasmoid.git"
HF_TOKEN="${HF_TOKEN:-}"                   # Export before running

# VM priority: try A100 spot first, then T4 spot, then T4 PAYG
VM_SIZES=("Standard_NC24ads_A100_v4" "Standard_NC4as_T4_v3")
PRIORITY="Spot"                            # Spot = cheapest (~80% off)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  LASMOID — Automated Azure GPU Deploy                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Verify prerequisites ───────────────────────────────
echo "📋 Checking prerequisites..."

if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Install: brew install azure-cli"
    exit 1
fi

# Check if logged in
if ! az account show &> /dev/null; then
    echo "  Logging in to Azure..."
    az login
fi

SUB_NAME=$(az account show --query "name" -o tsv)
echo "  ✅ Azure CLI: logged in (subscription: $SUB_NAME)"

if [ -z "$HF_TOKEN" ]; then
    echo "  ⚠️  HF_TOKEN not set. Export it:"
    echo "     export HF_TOKEN=hf_your_token_here"
    echo "  (Needed for Gemma-4 model access)"
    read -p "  Continue without HF_TOKEN? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi
fi

# ── Step 1: Create Resource Group ──────────────────────────────
echo ""
echo "🏗️  Creating resource group: $RESOURCE_GROUP ($LOCATION)..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none 2>/dev/null || true
echo "  ✅ Resource group ready"

# ── Step 2: Try to create VM (A100 → T4 fallback) ─────────────
echo ""
echo "🖥️  Launching GPU VM..."

VM_CREATED=false
for VM_SIZE in "${VM_SIZES[@]}"; do
    echo "  Trying $VM_SIZE ($PRIORITY)..."
    
    if az vm create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --size "$VM_SIZE" \
        --image "Canonical:ubuntu-24_04-lts:server:latest" \
        --admin-username "$ADMIN_USER" \
        --ssh-key-values "$SSH_KEY" \
        --priority "$PRIORITY" \
        --eviction-policy Deallocate \
        --max-price -1 \
        --os-disk-size-gb 256 \
        --public-ip-sku Standard \
        --output none 2>/dev/null; then
        echo "  ✅ VM created: $VM_SIZE ($PRIORITY)"
        VM_CREATED=true
        break
    else
        echo "  ⚠️  $VM_SIZE not available. Trying next..."
    fi
done

# If spot fails, try PAYG on T4
if [ "$VM_CREATED" = false ]; then
    echo "  Trying T4 (Pay-As-You-Go)..."
    PRIORITY="Regular"
    VM_SIZE="Standard_NC4as_T4_v3"
    az vm create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --size "$VM_SIZE" \
        --image "Canonical:ubuntu-24_04-lts:server:latest" \
        --admin-username "$ADMIN_USER" \
        --ssh-key-values "$SSH_KEY" \
        --priority Regular \
        --os-disk-size-gb 128 \
        --public-ip-sku Standard \
        --output none
    echo "  ✅ VM created: $VM_SIZE (PAYG)"
    VM_CREATED=true
fi

if [ "$VM_CREATED" = false ]; then
    echo "❌ Could not create any GPU VM. You may need to request quota."
    echo "   Go to: Azure Portal → Help + Support → New support request"
    echo "   Request: 'Increase GPU quota for NC24ads_A100_v4 in eastus'"
    exit 1
fi

# ── Step 3: Get VM IP ──────────────────────────────────────────
echo ""
echo "🌐 Getting VM IP address..."
VM_IP=$(az vm show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --show-details \
    --query "publicIps" -o tsv)
echo "  ✅ VM IP: $VM_IP"

# ── Step 4: Wait for SSH ───────────────────────────────────────
echo ""
echo "⏳ Waiting for SSH to be ready..."
for i in {1..30}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$ADMIN_USER@$VM_IP" "echo ok" &>/dev/null; then
        echo "  ✅ SSH connected"
        break
    fi
    sleep 10
    echo "  ... waiting ($i/30)"
done

# ── Step 5: Setup & Start Training (remote) ────────────────────
echo ""
echo "🚀 Installing deps and starting training on remote VM..."

ssh -o StrictHostKeyChecking=no "$ADMIN_USER@$VM_IP" bash -s "$GITHUB_REPO" "$HF_TOKEN" "$VM_SIZE" << 'REMOTE_SCRIPT'
#!/bin/bash
set -e
GITHUB_REPO="$1"
HF_TOKEN="$2"
VM_SIZE="$3"

echo "════════════ REMOTE SETUP START ════════════"

# Install NVIDIA driver + CUDA (if not pre-installed)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-driver-535 nvidia-cuda-toolkit
fi

# Install Python deps
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git tmux
pip install -q torch transformers safetensors tiktoken numpy tqdm datasets huggingface_hub accelerate

# Clone repo
cd /home/$USER
if [ ! -d "Lasmoid" ]; then
    git clone "$GITHUB_REPO"
fi
cd Lasmoid

# HuggingFace login
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "✅ HuggingFace logged in"
fi

# Verify GPU
python3 -c "
import torch
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu} ({vram:.0f}GB)')
assert torch.cuda.is_available()
print('✅ GPU ready')
"

# Determine which training script to run based on GPU
echo ""
echo "VM Size: $VM_SIZE"
if echo "$VM_SIZE" | grep -qi "A100\|H100"; then
    echo "🚀 A100/H100 detected → Running 500M distillation from Gemma-4"
    TRAIN_CMD="python3 scripts/distill_gemma4.py"
else
    echo "📐 T4 detected → Running 300M training on FineWeb-Edu"
    TRAIN_CMD="python3 scripts/gpu_train.py --config config_300m.json --device cuda:0 --max_iters 10000 --batch_size 4"
fi

# Start training in tmux (survives SSH disconnect)
tmux new-session -d -s train "$TRAIN_CMD 2>&1 | tee /home/$USER/train.log"
echo ""
echo "════════════ TRAINING STARTED ════════════"
echo "Running: $TRAIN_CMD"
echo "Log: /home/$USER/train.log"
echo "Reconnect: ssh $USER@$(hostname -I | awk '{print $1}') then: tmux attach -t train"
REMOTE_SCRIPT

# ── Step 6: Done ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ DONE — Training is running on Azure                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  VM: $VM_NAME ($VM_SIZE)                                 ║"
echo "║  IP: $VM_IP                                              ║"
echo "║  SSH: ssh $ADMIN_USER@$VM_IP                             ║"
echo "║                                                          ║"
echo "║  Monitor:                                                ║"
echo "║    ssh $ADMIN_USER@$VM_IP 'tmux attach -t train'         ║"
echo "║    ssh $ADMIN_USER@$VM_IP 'tail -f ~/train.log'          ║"
echo "║    ssh $ADMIN_USER@$VM_IP 'nvidia-smi'                   ║"
echo "║                                                          ║"
echo "║  Stop VM (to save money):                                ║"
echo "║    az vm deallocate -g $RESOURCE_GROUP -n $VM_NAME       ║"
echo "║                                                          ║"
echo "║  Delete everything when done:                            ║"
echo "║    az group delete -n $RESOURCE_GROUP --yes              ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
