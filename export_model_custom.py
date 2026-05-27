import os
import sys

# Thêm PaddleOCR vào path
paddleocr_path = "/home/ubuntu/PaddleOCR"
if os.path.exists(paddleocr_path):
    sys.path.append(paddleocr_path)
    print(f"Added {paddleocr_path} to sys.path")

import paddle
from ppocr.utils.export_model import export

def main():
    config_path = "/home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/rec_model/config.yml"
    pretrained_model = "/home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/rec_model/best_accuracy"
    save_inference_dir = "/home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/rec_inference"

    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Cập nhật đường dẫn trong config
    config['Global']['pretrained_model'] = pretrained_model
    config['Global']['save_inference_dir'] = save_inference_dir
    config['Global']['use_gpu'] = False # Export trên CPU cho an toàn trong sandbox
    config['Global']['export_with_pir'] = False # Tắt PIR để tương thích với Paddle < 3.0.0

    print(f"Exporting model from {pretrained_model}...")
    try:
        export(config)
        print(f"Model exported successfully to {save_inference_dir}")
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
