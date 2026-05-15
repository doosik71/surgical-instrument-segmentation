#!/usr/bin/env bash
set -euo pipefail

KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
SOURCE_LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list
REPO_LIST_URL=https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o "$KEYRING"

curl -fsSL "$REPO_LIST_URL" \
  | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" \
  | sudo tee "$SOURCE_LIST" >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

echo
echo "NVIDIA Container Toolkit installation is complete."
echo "Run the following command to restart Docker:"
echo "  sudo systemctl restart docker"
