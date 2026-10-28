#!/usr/bin/env bash
# One-time setup for the Broward plant ID project on a fresh Raspberry Pi OS (64-bit).
# Run from inside the project folder:
#   bash setup_pi.sh
set -euo pipefail

cd "$(dirname "$0")"
APP_DIR="$(pwd)"
RUN_USER="$(id -un)"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "This needs 64-bit Raspberry Pi OS (found $(uname -m)). Re-flash the card with the 64-bit image." >&2
  exit 1
fi

echo "== Installing system packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-pip libopenblas0 libgl1 v4l-utils

echo "== Giving $RUN_USER access to the webcam =="
sudo usermod -aG video "$RUN_USER"

echo "== Creating Python environment (venv/) =="
python3 -m venv --system-site-packages venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo "== Pre-downloading AlexNet weights (so train.py works offline) =="
venv/bin/python -c "import torchvision; torchvision.models.alexnet(weights='IMAGENET1K_V1')"

mkdir -p data models

echo
echo "== Checking the webcam =="
v4l2-ctl --list-devices 2>/dev/null || echo "No webcam found. Plug it into a USB port."

echo
echo "Done. Reboot now with:  sudo reboot"
echo "After reboot:  cd $APP_DIR && source venv/bin/activate && python collect.py --list"
