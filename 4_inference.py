#!/usr/bin/env python3
"""
Inference với PaddleOCR model đã fine-tune tiếng Việt
=====================================================
Sử dụng detection + recognition model đã finetune để OCR ảnh tiếng Việt.

Hỗ trợ 3 chế độ:
1. Dùng model finetune (det + rec custom)
2. Dùng model finetune chỉ recognition (det gốc + rec finetune)
3. So sánh model gốc vs model finetune

Usage:
    # Dùng cả det + rec model đã finetune
    python 4_inference.py --image test.png \\
        --det_model ./finetuned_models/det/det_inference \\
        --rec_model ./finetuned_models/rec/rec_inference \\
        --dict ./vietnamese_dict.txt

    # Chỉ finetune rec (det dùng model gốc)
    python 4_inference.py --image test.png \\
        --rec_model ./finetuned_models/rec/rec_inference \\
        --dict ./vietnamese_dict.txt

    # So sánh model gốc vs finetune
    python 4_inference.py --image test.png --compare \\
        --rec_model ./finetuned_models/rec/rec_inference \\
        --dict ./vietnamese_dict.txt

    # Batch xử lý thư mục ảnh
    python 4_inference.py --input_dir ./test_images/ \\
        --det_model ./finetuned_models/det/det_inference \\
        --rec_model ./finetuned_models/rec/rec_inference \\
        --dict ./vietnamese_dict.txt \\
        --output_dir ./results
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# INFERENCE ENGINE
# ============================================================

class VietnameseOCR:
    """
    OCR engine sử dụng PaddleOCR với model đã finetune tiếng Việt.

    Có thể load:
    - Model detection finetune
    - Model recognition finetune + Vietnamese dict
    - Hoặc kết hợp cả hai
    """

    def __init__(
        self,
        det_model_dir: str = None,
        rec_model_dir: str = None,
        dict_path: str = None,
        rec_algorithm: str = "SVTR_LCNet",
        use_gpu: bool = True,
    ):
        """
        Args:
            det_model_dir: Đường dẫn thư mục chứa model detection đã finetune
                           (chứa inference.pdiparams, inference.pdmodel)
            rec_model_dir: Đường dẫn thư mục chứa model recognition đã finetune
            dict_path: Đường dẫn file vietnamese_dict.txt
            rec_algorithm: Algorithm recognition (SVTR_LCNet hoặc CRNN)
            use_gpu: Sử dụng GPU
        """
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.dict_path = dict_path
        self.ocr = None

        self._init_ocr(use_gpu, rec_algorithm)

    def _init_ocr(self, use_gpu: bool, rec_algorithm: str):
        """Khởi tạo PaddleOCR với model custom"""
        from paddleocr import PaddleOCR

        # Build kwargs
        ocr_kwargs = {
            "use_angle_cls": True,
            "lang": "vi",  # Base language
            "show_log": False,
            "use_gpu": use_gpu,
        }

        # Nếu có custom detection model
        if self.det_model_dir and os.path.exists(self.det_model_dir):
            det_model = os.path.join(self.det_model_dir, "inference.pdparams")
            if not os.path.exists(det_model):
                # Thử tên file khác
                det_model = self.det_model_dir
            ocr_kwargs["det_model_dir"] = det_model
            print(f"[DET] Sử dụng model: {self.det_model_dir}")

        # Nếu có custom recognition model
        if self.rec_model_dir and os.path.exists(self.rec_model_dir):
            rec_model = os.path.join(self.rec_model_dir, "inference.pdparams")
            if not os.path.exists(rec_model):
                rec_model = self.rec_model_dir
            ocr_kwargs["rec_model_dir"] = rec_model
            print(f"[REC] Sử dụng model: {self.rec_model_dir}")

        # Nếu có custom dict
        if self.dict_path and os.path.exists(self.dict_path):
            ocr_kwargs["rec_char_dict_path"] = self.dict_path
            print(f"[DICT] Sử dụng: {self.dict_path}")

        # Set rec algorithm
        if rec_algorithm:
            ocr_kwargs["rec_algorithm"] = rec_algorithm

        print("Đang khởi tạo OCR engine...")
        self.ocr = PaddleOCR(**ocr_kwargs)
        print("OCR engine sẵn sàng!")

    def ocr_image(self, image_path: str, cls: bool = True) -> list:
        """
        OCR một ảnh, trả về danh sách các dòng text.

        Returns:
            List of dict:
                - bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                - text: str
                - confidence: float
        """
        result = self.ocr.ocr(image_path, cls=cls)

        outputs = []
        if result is None or len(result) == 0:
            return outputs

        for page in result:
            if page is None:
                continue
            for line in page:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                outputs.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": round(confidence, 4),
                })

        return outputs

    def ocr_image_full_text(self, image_path: str) -> str:
        """OCR và trả về toàn bộ text gộp"""
        results = self.ocr_image(image_path)
        texts = [r["text"] for r in results]
        return "\n".join(texts)


# ============================================================
# VISUALIZATION
# ============================================================

def draw_ocr_results(image_path: str, results: list, output_path: str = None):
    """
    Vẽ bounding box + text lên ảnh kết quả OCR.

    Args:
        image_path: Ảnh gốc
        results: List of dict từ ocr_image()
        output_path: Đường dẫn lưu ảnh kết quả (mặc định: tự tạo)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không đọc được ảnh: {image_path}")
        return

    # Vẽ từng bbox + text
    for r in results:
        bbox = np.array(r["bbox"], dtype=np.int32)
        text = r["text"]
        conf = r["confidence"]

        # Vẽ polygon bbox
        cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

        # Ghi text + confidence
        x_min = int(np.min(bbox[:, 0]))
        y_min = int(np.min(bbox[:, 1])) - 10
        label = f"{text} ({conf:.2f})"
        cv2.putText(img, label, (x_min, max(y_min, 15)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Lưu ảnh kết quả
    if output_path is None:
        stem = Path(image_path).stem
        suffix = Path(image_path).suffix
        output_path = f"{stem}_result{suffix}"

    cv2.imwrite(output_path, img)
    print(f"Ảnh kết quả: {output_path}")
    return output_path


# ============================================================
# SO SÁNH MODEL GỐC VS FINETUNE
# ============================================================

def compare_models(image_path: str, custom_rec_model: str = None,
                   custom_det_model: str = None, dict_path: str = None):
    """So sánh kết quả OCR giữa model gốc và model finetune"""
    from paddleocr import PaddleOCR

    print("\n" + "=" * 60)
    print("SO SÁNH: MODEL GỐC vs MODEL FINETUNE")
    print("=" * 60)

    # Model gốc
    print("\n[1] Model gốc (PaddleOCR vi mặc định)...")
    ocr_original = PaddleOCR(
        use_angle_cls=True, lang="vi", show_log=False
    )
    result_original = ocr_original.ocr(image_path, cls=True)

    # Model finetune
    print("\n[2] Model finetune...")
    ocr_kwargs = {"use_angle_cls": True, "lang": "vi", "show_log": False}
    if custom_det_model:
        ocr_kwargs["det_model_dir"] = custom_det_model
    if custom_rec_model:
        ocr_kwargs["rec_model_dir"] = custom_rec_model
    if dict_path:
        ocr_kwargs["rec_char_dict_path"] = dict_path

    ocr_finetuned = PaddleOCR(**ocr_kwargs)
    result_finetuned = ocr_finetuned.ocr(image_path, cls=True)

    # In kết quả so sánh
    print("\n" + "-" * 60)
    print(f"{'Model Gốc (PaddleOCR vi)':<40} | {'Model Finetune':<40}")
    print("-" * 60)

    orig_texts = []
    if result_original and result_original[0]:
        orig_texts = [line[1][0] for line in result_original[0]]

    ft_texts = []
    if result_finetuned and result_finetuned[0]:
        ft_texts = [line[1][0] for line in result_finetuned[0]]

    max_len = max(len(orig_texts), len(ft_texts))
    for i in range(max_len):
        orig = orig_texts[i] if i < len(orig_texts) else ""
        ft = ft_texts[i] if i < len(ft_texts) else ""
        print(f"{orig:<40} | {ft:<40}")

    print("-" * 60)


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_directory(input_dir: str, ocr_engine: VietnameseOCR,
                      output_dir: str, draw: bool = True):
    """Xử lý tất cả ảnh trong thư mục"""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for f in sorted(os.listdir(input_dir)):
        if Path(f).suffix.lower() in extensions:
            image_files.append(os.path.join(input_dir, f))

    if not image_files:
        print(f"Không tìm thấy ảnh trong: {input_dir}")
        return

    # Lưu kết quả JSON
    all_results = {}

    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path}")
        start_time = time.time()

        results = ocr_engine.ocr_image(img_path)
        elapsed = time.time() - start_time

        # In kết quả
        for r in results:
            print(f"  [{r['confidence']:.4f}] {r['text']}")

        full_text = "\n".join([r["text"] for r in results])
        print(f"  → {len(results)} dòng, {elapsed:.2f}s")

        # Lưu vào results
        all_results[img_path] = {
            "text": full_text,
            "lines": results,
            "time_seconds": round(elapsed, 2),
        }

        # Vẽ bbox lên ảnh
        if draw:
            output_img = os.path.join(
                output_dir,
                f"{Path(img_path).stem}_result.png"
            )
            draw_ocr_results(img_path, results, output_img)

        # Lưu text file
        txt_file = os.path.join(
            output_dir,
            f"{Path(img_path).stem}.txt"
        )
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(full_text)

    # Lưu tổng kết JSON
    summary_file = os.path.join(output_dir, "ocr_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Đã xử lý {len(image_files)} ảnh")
    print(f"Kết quả lưu tại: {output_dir}/")
    print(f"{'='*60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference với PaddleOCR model đã fine-tune tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # OCR 1 ảnh với model finetune
  python 4_inference.py --image test.png \\
      --det_model ./finetuned_models/det/det_inference \\
      --rec_model ./finetuned_models/rec/rec_inference \\
      --dict ./vietnamese_dict.txt

  # OCR thư mục ảnh
  python 4_inference.py --input_dir ./test_images/ \\
      --rec_model ./finetuned_models/rec/rec_inference \\
      --dict ./vietnamese_dict.txt --output_dir ./results

  # So sánh model gốc vs finetune
  python 4_inference.py --image test.png --compare \\
      --rec_model ./finetuned_models/rec/rec_inference \\
      --dict ./vietnamese_dict.txt
        """,
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Đường dẫn ảnh cần OCR",
    )
    parser.add_argument(
        "--input_dir", "-d",
        type=str,
        help="Thư mục chứa ảnh cần OCR (batch mode)",
    )
    parser.add_argument(
        "--det_model",
        type=str,
        default=None,
        help="Đường dẫn model detection đã finetune",
    )
    parser.add_argument(
        "--rec_model",
        type=str,
        default=None,
        help="Đường dẫn model recognition đã finetune",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vietnamese_dict.txt"),
        help="Đường dẫn file vietnamese_dict.txt",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./ocr_results",
        help="Thư mục lưu kết quả (batch mode)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="So sánh model gốc vs model finetune",
    )
    parser.add_argument(
        "--no_draw",
        action="store_true",
        help="Không vẽ bbox lên ảnh kết quả",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Sử dụng CPU (không dùng GPU)",
    )

    args = parser.parse_args()

    # Validate
    if not args.image and not args.input_dir:
        parser.error("Phải chỉ định --image hoặc --input_dir")

    # Chế độ so sánh
    if args.compare and args.image:
        compare_models(
            image_path=args.image,
            custom_rec_model=args.rec_model,
            custom_det_model=args.det_model,
            dict_path=args.dict,
        )
        return

    # Khởi tạo OCR engine
    ocr_engine = VietnameseOCR(
        det_model_dir=args.det_model,
        rec_model_dir=args.rec_model,
        dict_path=args.dict,
        use_gpu=not args.use_cpu,
    )

    # Chế độ 1 ảnh
    if args.image:
        print(f"\nOCR: {args.image}")
        start_time = time.time()

        results = ocr_engine.ocr_image(args.image)
        elapsed = time.time() - start_time

        print(f"\nKết quả ({len(results)} dòng, {elapsed:.2f}s):")
        print("-" * 60)
        for r in results:
            print(f"  [{r['confidence']:.4f}] {r['text']}")

        # In toàn bộ text
        full_text = "\n".join([r["text"] for r in results])
        print(f"\n--- Full text ---\n{full_text}\n---")

        # Vẽ kết quả
        if not args.no_draw:
            draw_ocr_results(args.image, results)

    # Chế độ batch
    elif args.input_dir:
        process_directory(
            input_dir=args.input_dir,
            ocr_engine=ocr_engine,
            output_dir=args.output_dir,
            draw=not args.no_draw,
        )


if __name__ == "__main__":
    main()
