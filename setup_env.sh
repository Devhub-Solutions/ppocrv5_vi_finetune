#!/bin/bash
# =============================================
# Vietnamese OCR Toolkit - Cài đặt môi trường
# =============================================
# Yêu cầu: Python 3.10-3.12, pip, venv
# Chạy: chmod +x setup_env.sh && ./setup_env.sh

set -e

echo "========================================="
echo "  Vietnamese OCR Toolkit - Setup"
echo "========================================="

# Kiểm tra Python version
PYTHON=${PYTHON:-python3}
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "[1/5] Python version: $PY_VERSION"

# Tạo virtual environment
VENV_DIR=${VENV_DIR:-".venv"}
echo "[2/5] Tạo virtual environment tại: $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"

# Kích hoạt venv
source "$VENV_DIR/bin/activate"
echo "  → Đã kích hoạt venv"

# Upgrade pip
echo "[3/5] Upgrade pip..."
pip install --upgrade pip

# Cài đặt requirements
echo "[4/5] Cài đặt dependencies (có thể mất vài phút)..."
pip install -r requirements.txt

# Tạo cấu trúc thư mục
echo "[5/5] Tạo cấu trúc thư mục..."
mkdir -p input_images
mkdir -p training_output
mkdir -p finetuned_models

echo ""
echo "========================================="
echo "  Cài đặt hoàn tất!"
echo "========================================="
echo ""
echo "Cách sử dụng:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "  # 1. Tạo training data từ ảnh"
echo "  python 1_generate_training_data.py --input_dir ./input_images --output_dir ./training_output"
echo ""
echo "  # 2. Finetune recognition model"
echo "  python 2_finetune_recognition.py --train_dir ./training_output/recognition --output_dir ./finetuned_models/rec"
echo ""
echo "  # 3. Finetune detection model"
echo "  python 3_finetune_detection.py --train_dir ./training_output/detection --output_dir ./finetuned_models/det"
echo ""
echo "  # 4. Chạy inference với model đã finetune"
echo "  python 4_inference.py --image test.png --det_model ./finetuned_models/det/best --rec_model ./finetuned_models/rec/best"
