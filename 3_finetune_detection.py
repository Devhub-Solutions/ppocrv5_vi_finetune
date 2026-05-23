#!/usr/bin/env python3
"""
Fine-tune PaddleOCR Detection Model cho tiếng Việt
===================================================
Fine-tune model detection (DB - Differentiable Binarization) với dữ liệu
tiếng Việt đã được tạo từ pipeline 1_generate_training_data.py.

PaddleOCR detection sử dụng kiến trúc DB (PP-OCRv4: PP-OCRv4_det).

Training detection model phức tạp hơn recognition vì cần:
- Ảnh gốc full-page
- Annotation dạng polygon (4 điểm mỗi bbox)
- Format: image_path\t[{"transcription":"text","points":[[x1,y1],...]}]

Usage:
    # Fine-tune với GPU
    python 3_finetune_detection.py --train_dir ./training_output/detection --output_dir ./finetuned_models/det

    # Fine-tune với CPU (rất chậm)
    python 3_finetune_detection.py --train_dir ./training_output/detection --output_dir ./finetuned_models/det --use_cpu

    # Tùy chỉnh tham số
    python 3_finetune_detection.py --train_dir ./training_output/detection --output_dir ./finetuned_models/det --epochs 200 --batch_size 8 --lr 0.001
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================

DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 0.001
DEFAULT_TRAIN_RATIO = 0.9

# Vietnamese character dict (cho detection không bắt buộc, nhưng cần trong config)
VIETNAMESE_DICT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vietnamese_dict.txt")


# ============================================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU
# ============================================================

def prepare_detection_data(train_dir: str, output_dir: str, train_ratio: float = 0.9):
    """
    Đọc annotations.txt và chia train/val cho detection training.

    Format annotations.txt:
    image_path\t[{"transcription":"text","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]
    """
    ann_file = os.path.join(train_dir, "annotations.txt")
    img_dir = os.path.join(train_dir, "images")

    if not os.path.exists(ann_file):
        print(f"LỖI: Không tìm thấy {ann_file}")
        print("Hãy chạy 1_generate_training_data.py trước!")
        sys.exit(1)

    # Đọc tất cả annotations
    with open(ann_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Validate data
    valid_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            img_path = os.path.join(train_dir, parts[0])
            if os.path.exists(img_path):
                try:
                    anns = json.loads(parts[1])
                    if len(anns) > 0:
                        valid_lines.append(line)
                except json.JSONDecodeError:
                    print(f"  WARNING: JSON không hợp lệ: {parts[1][:50]}...")
            else:
                print(f"  WARNING: Ảnh không tồn tại: {img_path}")

    if not valid_lines:
        print("LỖI: Không có dữ liệu detection hợp lệ!")
        sys.exit(1)

    # Shuffle và chia train/val
    random.seed(42)
    random.shuffle(valid_lines)

    split_idx = int(len(valid_lines) * train_ratio)
    train_lines = valid_lines[:split_idx]
    val_lines = valid_lines[split_idx:]

    # Tạo thư mục data
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "val.txt")

    # Ghi train/val files với absolute paths
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            parts = line.split("\t", 1)
            abs_img_path = os.path.abspath(os.path.join(train_dir, parts[0]))
            f.write(f"{abs_img_path}\t{parts[1]}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            parts = line.split("\t", 1)
            abs_img_path = os.path.abspath(os.path.join(train_dir, parts[0]))
            f.write(f"{abs_img_path}\t{parts[1]}\n")

    print(f"Dữ liệu detection: {len(train_lines)} ảnh train, {len(val_lines)} ảnh val")
    return train_file, val_file


# ============================================================
# BƯỚC 2: TẠO CONFIG YAML
# ============================================================

def generate_det_config(
    train_file: str,
    val_file: str,
    output_dir: str,
    dict_path: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    use_cpu: bool = False,
    pretrained_model: str = None,
    image_shape: str = "3,640,640",
):
    """Tạo file config YAML cho PaddleOCR detection training."""

    config_dir = os.path.join(output_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # Parse image shape
    shapes = [int(x) for x in image_shape.split(",")]
    img_channel, img_height, img_width = shapes[0], shapes[1], shapes[2]

    # Tìm pretrained detection model
    if pretrained_model and os.path.exists(pretrained_model):
        pretrain_path = pretrained_model
    else:
        possible_paths = [
            os.path.expanduser("~/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer"),
            os.path.expanduser("~/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer"),
        ]
        pretrain_path = None
        for p in possible_paths:
            if os.path.exists(p):
                pretrain_path = p
                break

    # DB detection config (PP-OCRv4 style)
    config_content = f"""Global:
  debug: false
  use_gpu: {'false' if use_cpu else 'true'}
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {os.path.abspath(output_dir)}/det_model
  save_epoch_step: 20
  eval_batch_step:
  - 0
  - {max(1, batch_size // 2)}
  cal_metric_during_train: true
  pretrained_model: {'null' if not pretrain_path else pretrain_path}
  checkpoints: null
  save_inference_dir: {os.path.abspath(output_dir)}/det_inference
  use_visualdl: true
  infer_img: null
  character_dict_path: {os.path.abspath(dict_path)}
  distributed: true

Architecture:
  model_type: det
  algorithm: DB
  Transform: null
  Backbone:
    name: PPLCNetV2
    scale: 0.75
    det: true
  Neck:
    name: RSEFPN
    out_channels: 96
    shortcut: true
  Head:
    name: DBHead
    k: 50

Loss:
  name: DBLoss
  balance_loss: true
  main_loss_type: DiceLoss
  alpha: 5
  beta: 10
  ohem_ratio: 3

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    learning_rate: {learning_rate}
  regularizer:
    name: L2
    factor: 5.0e-05

PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
    - {os.path.abspath(train_file)}
    ratio_list: [1.0]
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - CopyPaste: null
    - IaaAugment:
        augmenter_args:
        - type: Fliplr
          args:
            p: 0.5
        - type: Affine
          args:
            rotate:
            - -10
            - 10
        - type: Resize
          args:
            size:
            - 0.5
            - 3.0
    - EastRandomCropData:
        size:
        - {img_height}
        - {img_width}
        max_tries: 50
        keep_ratio: true
    - MakeBorderMap:
        shrink_ratio: 0.4
        thresh_min: 0.3
        thresh_max: 0.7
    - MakeShrinkMap:
        shrink_ratio: 0.4
        min_text_size: 8
    - NormalizeImage:
        scale: 1.0/255.0
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - threshold_map
        - threshold_mask
        - shrink_map
        - shrink_mask
  loader:
    shuffle: true
    drop_last: true
    batch_size_per_card: {batch_size}
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./
    label_file_list:
    - {os.path.abspath(val_file)}
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - DetResizeForTestData:
        keep_ratio: true
    - NormalizeImage:
        scale: 1.0/255.0
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - shape
        - polys
        - ignore_tags
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 1
    num_workers: 2
"""

    config_file = os.path.join(config_dir, "det_vi_config.yml")
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"Config saved: {config_file}")
    return config_file


# ============================================================
# BƯỚC 3: CHẠY TRAINING
# ============================================================

def run_training(config_file: str, use_cpu: bool = False):
    """Chạy PaddleOCR detection training."""
    print("\n" + "=" * 60)
    print("BẮT ĐẦU FINE-TUNE DETECTION MODEL")
    print("=" * 60)

    # Tìm PaddleOCR train script
    try:
        paddleocr_path = os.path.dirname(__import__("paddleocr").__file__)
    except ImportError:
        paddleocr_path = None

    possible_train_scripts = []
    if paddleocr_path:
        possible_train_scripts = [
            os.path.join(paddleocr_path, "..", "tools", "train.py"),
            os.path.join(paddleocr_path, "tools", "train.py"),
        ]

    train_script = None
    for p in possible_train_scripts:
        if os.path.exists(p):
            train_script = p
            break

    if train_script:
        cmd = f"python {train_script} -c {config_file}"
        if use_cpu:
            cmd += " Global.use_gpu=false"
        print(f"Chạy: {cmd}")
        os.system(cmd)
    else:
        print("""
Không tìm thấy PaddleOCR tools/train.py trực tiếp.
Chạy training thủ công:

========================================
Cách 1: Clone PaddleOCR và train (KHUYẾN NGHỊ)
========================================
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt

# Fine-tune detection model
python tools/train.py -c %s

# Export model sau khi train
python tools/export_model.py -c %s -o Global.pretrained_model=./det_model/best_accuracy Global.save_inference_dir=./det_inference

========================================
Cách 2: Dùng docker
========================================
docker pull paddlepaddle/paddle:latest-gpu
# Mount data và chạy training
""" % (config_file, config_file))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune PaddleOCR Detection Model cho tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python 3_finetune_detection.py --train_dir ./training_output/detection --output_dir ./finetuned_models/det
  python 3_finetune_detection.py --train_dir ./training_output/detection --output_dir ./finetuned_models/det --use_cpu
        """,
    )
    parser.add_argument(
        "--train_dir", "-d",
        type=str,
        required=True,
        help="Thư mục chứa dữ liệu detection (có annotations.txt + images/)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./finetuned_models/det",
        help="Thư mục lưu model đã finetune",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Số epoch training (mặc định: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (mặc định: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (mặc định: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default=VIETNAMESE_DICT,
        help="Đường dẫn file vietnamese_dict.txt",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Đường dẫn pretrained model (mặc định: tự tìm)",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Dùng CPU thay vì GPU",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Tỷ lệ train/val (mặc định: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Chỉ chuẩn bị data + config, không chạy training",
    )

    args = parser.parse_args()

    # Bước 1: Chuẩn bị data
    print("Bước 1: Chuẩn bị dữ liệu detection...")
    train_file, val_file = prepare_detection_data(
        args.train_dir, args.output_dir, args.train_ratio
    )

    # Bước 2: Tạo config
    print("\nBước 2: Tạo config YAML...")
    config_file = generate_det_config(
        train_file=train_file,
        val_file=val_file,
        output_dir=args.output_dir,
        dict_path=args.dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_cpu=args.use_cpu,
        pretrained_model=args.pretrained,
    )

    if args.prepare_only:
        print(f"\nĐã chuẩn bị xong! Config tại: {config_file}")
        print("Chạy training:")
        print(f"  cd PaddleOCR && python tools/train.py -c {config_file}")
        return

    # Bước 3: Chạy training
    print("\nBước 3: Chạy training...")
    run_training(config_file, args.use_cpu)

    print("\n" + "=" * 60)
    print("HOÀN TẤT FINE-TUNE DETECTION MODEL")
    print(f"Model lưu tại: {args.output_dir}/det_inference/")
    print("=" * 60)


if __name__ == "__main__":
    main()
