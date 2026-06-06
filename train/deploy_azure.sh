#!/bin/bash
# ========================================================================
# Lasmoid — deploy_azure.sh
# ========================================================================
# Run this script directly inside the Azure Cloud Shell (https://shell.azure.com)
# It creates a resource group and provisions a GPU VM (NVIDIA V100).
# ========================================================================

# 1. Configuration Variables (Change as needed)
RG_NAME="lasmoid-rg"
LOCATION="eastus"              # Region with GPU availability
VM_NAME="lasmoid-gpu-vm"
VM_SIZE="Standard_NC6s_v3"     # 1x NVIDIA V100 GPU (16GB VRAM)
# Alternate T4 GPU if V100 is unavailable:
# VM_SIZE="Standard_NC4as_T4_v3"

# Deep Learning VM Image with CUDA, PyTorch, and NVIDIA Drivers pre-configured
IMAGE="microsoft-dsvm:ubuntu-2004:pytorch-latest:latest"

echo "=== Starting Azure GPU VM Deployment ==="

# 2. Create Resource Group
echo "Creating Resource Group '$RG_NAME' in location '$LOCATION'..."
az group create --name $RG_NAME --location $LOCATION

# 3. Create GPU VM
echo "Provisioning GPU VM '$VM_NAME' of size '$VM_SIZE'..."
echo "This takes about 2 to 3 minutes..."
az vm create \
  --resource-group $RG_NAME \
  --name $VM_NAME \
  --image $IMAGE \
  --size $VM_SIZE \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# 4. Open Port 22 for SSH access
echo "Opening Port 22 for SSH..."
az vm open-port --resource-group $RG_NAME --name $VM_NAME --port 22

# 5. Fetch Public IP Address
PUBLIC_IP=$(az vm show -d -g $RG_NAME -n $VM_NAME --query publicIps -o tsv)

echo "=== Deployment Successful ==="
echo "VM Public IP Address: $PUBLIC_IP"
echo ""
echo "To connect to your GPU VM from your local Mac terminal, run:"
echo "ssh azureuser@$PUBLIC_IP"
echo ""
echo "Once connected, run these commands to start training:"
echo "--------------------------------------------------------"
echo "git clone https://github.com/Theory903/Lasmoid.git"
echo "cd Lasmoid"
echo "python3 -m venv .venv && source .venv/bin/activate"
echo "pip install datasets tqdm transformers torch numpy"
echo "python train/prepare_data.py --dataset fineweb-edu --max_seq_len 512 --limit 200000"
echo "nohup python train/train.py --train_bin train/fineweb-edu_train.bin --val_bin train/fineweb-edu_val.bin --train_meta train/fineweb-edu_train_meta.json --val_meta train/fineweb-edu_val_meta.json --batch_size 4 --grad_accum 8 --max_iters 5000 --device cuda > train.log 2>&1 &"
echo "--------------------------------------------------------"
