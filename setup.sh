#!/bin/bash
# Linux Mint / Ubuntu: 가상환경 만들고 의존성 설치
set -e
cd "$(dirname "$0")"
sudo apt-get install -y python3-venv python3-dev libgl1 libglfw3 ffmpeg >/dev/null || true
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import jax; print('JAX devices:', jax.devices())"
echo "설치 완료. 다음:  source .venv/bin/activate && python viewer.py"
