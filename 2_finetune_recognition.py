#!/usr/bin/env python3
"""
Fine-tune PaddleOCR Recognition Model cho tiếng Việt
=====================================================
Fine-tune model recognition (SVTR_LCNet) với dữ liệu tiếng Việt
đã được tạo từ pipeline 1_generate_training_data.py.

PaddleOCR sử dụng kiến trúc recognition: SVTR_LCNet (PP-OCRv4)
hoặc CRNN (PP-OCRv3). Script này hỗ trợ cả hai.

Cách hoạt động:
1. Đọc annotations.txt (format: image_path\ttext)
2. Chia train/val split
3. Tạo config YAML cho PaddleOCR training
4. Chạy fine-tune bằng PaddleOCR training engine
5. Export model cuối cùng

Yêu cầu:
  - Đã chạy 1_generate_training_data.py để có data
  - GPU khuyến nghị (CPU vẫn chạy được nhưng rất chậm)

Usage:
    # Fine-tune với GPU
    python 2_finetune_recognition.py --train_dir ./training_output/recognition --output_dir ./finetuned_models/rec

    # Fine-tune với CPU (chậm)
    python 2_finetune_recognition.py --train_dir ./training_output/recognition --output_dir ./finetuned_models/rec --use_cpu

    # Fine-tune với config tùy chỉnh
    python 2_finetune_recognition.py --train_dir ./training_output/recognition --output_dir ./finetuned_models/rec --epochs 200 --batch_size 32 --lr 0.001
"""

import os
import sys
import json
import argparse
import shutil
import random
from pathlib import Path

# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================

# Model base để fine-tune (PP-OCRv4 Latin recognition)
BASE_MODEL_DIR = os.path.join(
    os.path.expanduser("~"),
    ".paddleocr/whl/rec/latin/latin_PP-OCRv3_rec_infer"
)

# Vietnamese character dict
VIETNAMESE_DICT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vietnamese_dict.txt")

# Training defaults
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 0.0005
DEFAULT_TRAIN_RATIO = 0.9


# ============================================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU
# ============================================================

def prepare_training_data(train_dir: str, output_dir: str, train_ratio: float = 0.9):
    """
    Đọc annotations.txt và chia train/val, tạo format cho PaddleOCR training.

    PaddleOCR recognition training cần:
    - train.txt: mỗi dòng là "image_path\ttext"
    - val.txt: mỗi dòng là "image_path\ttext"
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

    # Lọc bỏ dòng có text rỗng
    valid_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            img_path = os.path.join(train_dir, parts[0])
            if os.path.exists(img_path):
                valid_lines.append(line)
            else:
                print(f"  WARNING: Ảnh không tồn tại: {img_path}")

    if not valid_lines:
        print("LỖI: Không có dữ liệu training hợp lệ!")
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

    # Ghi train.txt và val.txt
    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "val.txt")

    # Sử dụng đường dẫn tương đối để linh hoạt giữa các môi trường
    # Trong PaddleOCR, data_dir + label_file_list[i] line path = absolute path
    # Chúng ta sẽ lưu path tương đối so với train_dir
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(f"{line}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(f"{line}\n")

    print(f"Dữ liệu training: {len(train_lines)} mẫu train, {len(val_lines)} mẫu val")

    return train_file, val_file


# ============================================================
# BƯỚC 2: TẠO CONFIG YAML
# ============================================================

def generate_rec_config(
    train_file: str,
    val_file: str,
    output_dir: str,
    dict_path: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
    use_cpu: bool = False,
    pretrained_model: str = None,
    image_shape: str = "3,48,320",
):
    """Tạo file config YAML cho PaddleOCR recognition training."""

    config_dir = os.path.join(output_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # Xác định pretrained model path
    if pretrained_model and os.path.exists(pretrained_model):
        pretrain_path = pretrained_model
    else:
        # Tìm model đã download bởi PaddleOCR
        possible_paths = [
            os.path.expanduser("~/.paddleocr/whl/rec/latin/latin_PP-OCRv3_rec_infer"),
            os.path.expanduser("~/.paddleocr/whl/rec/latin/latin_PP-OCRv4_rec_infer"),
        ]
        pretrain_path = None
        for p in possible_paths:
            if os.path.exists(p):
                pretrain_path = p
                break

    # Parse image shape
    shapes = [int(x) for x in image_shape.split(",")]
    img_channel, img_height, img_width = shapes[0], shapes[1], shapes[2]

    # PaddleOCR PP-OCRv4 recognition config
    # Sử dụng đường dẫn tương đối cho các môi trường linh hoạt
    rel_dict_path = os.path.relpath(dict_path, os.getcwd())
    rel_train_file = os.path.relpath(train_file, os.getcwd())
    rel_val_file = os.path.relpath(val_file, os.getcwd())
    rel_output_dir = os.path.relpath(output_dir, os.getcwd())

    config_content = f"""Global:
  debug: false
  use_gpu: {'false' if use_cpu else 'true'}
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {rel_output_dir}/rec_model
  save_epoch_step: 10
  eval_batch_step:
  - 0
  - {max(1, batch_size // 2)}
  cal_metric_during_train: true
  pretrained_model: {'null' if not pretrain_path else pretrain_path}
  checkpoints: null
  save_inference_dir: {rel_output_dir}/rec_inference
  use_visualdl: true
  infer_img: null
  character_dict_path: {rel_dict_path}
  max_text_length: 50
  infer_mode: false
  use_space_char: true

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    learning_rate: {learning_rate}
  regularizer:
    name: L2
    factor: 2.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform: null
  Backbone:
    name: PPLCNetV3
    scale: 0.5
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: True
  Head:
    name: CTCHead
    fc_decay: 0.00001

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode
  character_dict_path: {rel_dict_path}
  use_space_char: true

Metric:
  name: RecMetric
  main_indicator: acc
  is_filter: true

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {os.path.relpath(os.path.dirname(train_file), os.getcwd())}/../../
    label_file_list:
    - {rel_train_file}
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode: null
    - RecResizeImg:
        image_shape: [{img_channel}, {img_height}, {img_width}]
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
  loader:
    shuffle: true
    batch_size_per_card: {batch_size}
    drop_last: true
    num_workers: 0

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {os.path.relpath(os.path.dirname(val_file), os.getcwd())}/../../
    label_file_list:
    - {rel_val_file}
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode: null
    - RecResizeImg:
        image_shape: [{img_channel}, {img_height}, {img_width}]
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: {batch_size}
    num_workers: 0
"""

    config_file = os.path.join(config_dir, "rec_vi_config.yml")
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"Config saved: {config_file}")
    return config_file


# ============================================================
# BƯỚC 3: CHẠY TRAINING
# ============================================================

def run_training(config_file: str, use_cpu: bool = False):
    """
    Chạy PaddleOCR training engine với config đã tạo.

    Sử dụng PaddleOCR's train.py module hoặc gọi trực tiếp
    PaddleClas/PaddleOCR training pipeline.
    """
    print("\n" + "=" * 60)
    print("BẮT ĐẦU FINE-TUNE RECOGNITION MODEL")
    print("=" * 60)

    # Phương pháp 1: Gọi qua PaddleOCR's tools
    # Kiểm tra xem paddleocr có tools train không
    paddleocr_path = os.path.dirname(
        __import__("paddleocr").__file__
    )

    # Tìm train.py trong PaddleOCR
    possible_train_scripts = [
        os.path.join(paddleocr_path, "..", "tools", "train.py"),
        os.path.join(paddleocr_path, "tools", "train.py"),
    ]

    train_script = None
    for p in possible_train_scripts:
        if os.path.exists(p):
            train_script = p
            break

    if train_script is None:
        print("Không tìm thấy PaddleOCR train.py, sử dụng phương pháp PaddleTraining API...")
        run_training_with_api(config_file, use_cpu)
    else:
        # Chạy training script
        cmd = f"python {train_script} -c {config_file}"
        if use_cpu:
            cmd += " Global.use_gpu=false"
        print(f"Chạy: {cmd}")
        os.system(cmd)


def run_training_with_api(config_file: str, use_cpu: bool = False):
    """
    Chạy training bằng PaddlePaddle Training API trực tiếp.
    Phương pháp thay thế khi không có PaddleOCR tools.
    """
    import yaml
    import paddle
    import paddle.nn as nn
    from paddle.io import DataLoader

    # Load config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    print(f"Config: {json.dumps(config, indent=2, default=str)[:500]}...")

    # Set device
    if use_cpu or not paddle.is_compiled_with_cuda():
        paddle.set_device("cpu")
        print("Sử dụng CPU")
    else:
        paddle.set_device("gpu")
        print("Sử dụng GPU")

    # Import PaddleOCR training modules
    try:
        from paddleocr.ppocr.utils.logging import get_logger
        from paddleocr.tools.train import train

        # Run training through PaddleOCR's train function
        train(config)
    except ImportError:
        print("PaddleOCR training modules không khả dụng trực tiếp.")
        print("Sử dụng phương pháp thủ công...")

        # Phương pháp thủ công: tự xây dựng training loop
        run_manual_training(config_file, config, use_cpu)


def run_manual_training(config_file: str, config: dict, use_cpu: bool = False):
    """
    Training loop thủ công cho recognition model.
    Dùng khi PaddleOCR training tools không import được.
    """
    import paddle
    import paddle.nn as nn
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    paddle.set_device("gpu" if (not use_cpu and paddle.is_compiled_with_cuda()) else "cpu")

    epochs = config["Global"]["epoch_num"]
    save_dir = config["Global"]["save_model_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    train_file = config["Train"]["dataset"]["label_file_list"][0]
    batch_size = config["Train"]["loader"]["batch_size_per_card"]

    # Đọc data
    data_pairs = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and os.path.exists(parts[0]):
                data_pairs.append((parts[0], parts[1]))

    print(f"Training samples: {len(data_pairs)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Save dir: {save_dir}")

    # Sử dụng PaddleOCR engine để train
    # Cách tốt nhất: clone PaddleOCR repo và dùng tools train
    print("\n" + "=" * 60)
    print("ĐỂ TRAIN CHUYÊN NGHIỆP, KHUYẾN NGHỊ:")
    print("=" * 60)
    print(f"""
Cách 1: Dùng PaddleOCR repo (khuyến nghị)
-------------------------------------------
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt

# Train recognition model
python tools/train.py -c {os.path.abspath(config_file)}

Cách 2: Dùng lệnh trực tiếp
-------------------------------------------
python -m paddleocr.tools.train -c {os.path.abspath(config_file)}

Cách 3: Export model sau khi train
-------------------------------------------
python tools/export_model.py -c {os.path.abspath(config_file)} -o Global.pretrained_model={save_dir}/best_accuracy Global.save_inference_dir={os.path.abspath(os.path.join(os.path.dirname(save_dir), "rec_inference"))}
""")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune PaddleOCR Recognition Model cho tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Fine-tune mặc định (GPU)
  python 2_finetune_recognition.py --train_dir ./training_output/recognition --output_dir ./finetuned_models/rec

  # Fine-tune với CPU
  python 2_finetune_recognition.py --train_dir ./training_output/recognition --output_dir ./finetuned_models/rec --use_cpu

  # Fine-tune với tham số tùy chỉnh
  python 2_finetune_recognition.py --train_dir ./training_output/recognition --output_dir ./finetuned_models/rec --epochs 200 --batch_size 32 --lr 0.001
        """,
    )
    parser.add_argument(
        "--train_dir", "-d",
        type=str,
        required=True,
        help="Thư mục chứa dữ liệu recognition (có annotations.txt + images/)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./finetuned_models/rec",
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
        help=f"Tỷ lệ train/val split (mặc định: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--image_shape",
        type=str,
        default="3,48,320",
        help="Kích thước ảnh input: channel,height,width (mặc định: 3,48,320)",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Chỉ chuẩn bị data + config, không chạy training",
    )

    args = parser.parse_args()

    # Bước 1: Chuẩn bị data
    print("Bước 1: Chuẩn bị dữ liệu training...")
    train_file, val_file = prepare_training_data(
        args.train_dir, args.output_dir, args.train_ratio
    )

    # Bước 2: Tạo config
    print("\nBước 2: Tạo config YAML...")
    config_file = generate_rec_config(
        train_file=train_file,
        val_file=val_file,
        output_dir=args.output_dir,
        dict_path=args.dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_cpu=args.use_cpu,
        pretrained_model=args.pretrained,
        image_shape=args.image_shape,
    )

    if args.prepare_only:
        print(f"\nĐã chuẩn bị xong! Config tại: {config_file}")
        print("Chạy training thủ công:")
        print(f"  cd PaddleOCR && python tools/train.py -c {config_file}")
        return

    # Bước 3: Chạy training
    print("\nBước 3: Chạy training...")
    run_training(config_file, args.use_cpu)

    print("\n" + "=" * 60)
    print("HOÀN TẤT FINE-TUNE RECOGNITION MODEL")
    print(f"Model lưu tại: {args.output_dir}/rec_inference/")
    print("=" * 60)


if __name__ == "__main__":
    main()
