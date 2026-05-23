#!/usr/bin/env python3
"""
Vietnamese OCR Training Data Pipeline
======================================
Pipeline: PaddleOCR bbox detection → Crop images → Ollama Cloud OCR → Save training data

Mục đích: Tạo dữ liệu training để finetune PaddleOCR tiếng Việt
- Bước 1: PaddleOCR quét ảnh lấy bbox (chỉ detection, không cần nhận dạng text chính xác)
- Bước 2: Crop ảnh theo từng bbox
- Bước 3: Gửi ảnh crop lên Ollama Cloud API (qwen3.5:397b-cloud) để OCR text chính xác
- Bước 4: Lưu ảnh crop + text annotation dưới dạng training data cho PaddleOCR

Output directory structure:
    training_output/
    ├── detection/                # Data cho fine-tune detection model
    │   ├── images/               # Ảnh gốc
    │   └── annotations.txt       # Format: img_path\t[{"transcription":"...","points":[[x1,y1],...]}]
    ├── recognition/              # Data cho fine-tune recognition model
    │   ├── images/               # Ảnh crop từng dòng text
    │   └── annotations.txt       # Format: img_path\ttranscription_text
    ├── raw_crops/                # Backup ảnh crop gốc
    ├── logs/                     # Log files
    └── pipeline_summary.json     # Thống kê pipeline

Usage:
    python vietnamese_ocr_pipeline.py --input_dir ./input_images --output_dir ./training_data
    python vietnamese_ocr_pipeline.py --input image.png --output_dir ./training_data
    python vietnamese_ocr_pipeline.py --input_dir ./input_images --output_dir ./training_data --skip_ollama
"""

import os
import sys
import json
import base64
import argparse
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import requests
from PIL import Image

# ============================================================
# CẤU HÌNH
# ============================================================

# Ollama Cloud API
OLLAMA_API_KEY = os.environ.get(
    "OLLAMA_API_KEY",
    "c6dff740c541467aaab9ee30c5b1ed50.ojB50W8G5Xk71WwrliR-dRLP"
)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/api")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:235b")

# PaddleOCR config
PADDLEOCR_LANG = "vi"  # Vietnamese

# Pipeline config
BBOX_PADDING = 5          # Padding around bbox khi crop (pixels)
MIN_CROP_WIDTH = 10       # Bỏ qua crop quá nhỏ
MIN_CROP_HEIGHT = 10      # Bỏ qua crop quá nhỏ
MAX_CROP_SIZE = 2048      # Resize crop nếu quá lớn
OLLAMA_TIMEOUT = 90       # Timeout cho Ollama API call (seconds)
OLLAMA_RETRY = 3          # Số lần retry khi API lỗi
OLLAMA_DELAY = 0.5        # Delay giữa các API call (seconds) để tránh rate limit

# OCR prompt cho Ollama
OCR_PROMPT = (
    "You are an expert OCR system specializing in Vietnamese text recognition. "
    "Please extract and return ONLY the text visible in this image. "
    "Preserve the original text exactly as written, including:\n"
    "- All Vietnamese diacritical marks (dấu): ă, â, đ, ê, ô, ơ, ư and tone marks\n"
    "- Original capitalization and punctuation\n"
    "- Line breaks if there are multiple lines\n\n"
    "Return ONLY the extracted text, nothing else. "
    "If the image contains no readable text, return an empty string."
)


# ============================================================
# LOGGING
# ============================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Thiết lập logging cho pipeline"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("OCR_Pipeline")
    logger.setLevel(logging.DEBUG)
    # Xóa handler cũ nếu có
    logger.handlers.clear()

    # File handler
    log_file = os.path.join(
        log_dir,
        f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================
# BƯỚC 1: PADDLEOCR - PHÁT HIỆN BBOX
# ============================================================

class PaddleOCRExtractor:
    """Sử dụng PaddleOCR để phát hiện vùng text (bbox) trong ảnh"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ocr = None

    def initialize(self):
        """Khởi tạo PaddleOCR engine"""
        self.logger.info("Đang khởi tạo PaddleOCR (lang=%s)...", PADDLEOCR_LANG)
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=PADDLEOCR_LANG,
            show_log=False,
        )
        self.logger.info("PaddleOCR đã sẵn sàng!")

    def detect_bboxes(self, image_path: str) -> list:
        """
        Phát hiện bbox text trong ảnh bằng PaddleOCR.

        Returns:
            List of dict, mỗi dict chứa:
                - bbox: list of [x1,y1], [x2,y2], [x3,y3], [x4,y4] (4 điểm polygon)
                - confidence: float
                - paddle_text: text do PaddleOCR nhận dạng (để tham khảo)
        """
        if self.ocr is None:
            self.initialize()

        self.logger.info("Đang phát hiện bbox: %s", image_path)
        result = self.ocr.ocr(image_path, cls=True)

        detections = []
        if result is None or len(result) == 0:
            self.logger.warning("Không phát hiện text nào trong: %s", image_path)
            return detections

        for page_result in result:
            if page_result is None:
                continue
            for line in page_result:
                bbox_points = line[0]       # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text_info = line[1]         # (text, confidence)
                text = text_info[0]
                confidence = text_info[1]

                detections.append({
                    "bbox": bbox_points,
                    "confidence": round(confidence, 4),
                    "paddle_text": text,
                })

        self.logger.info("Phát hiện %d vùng text trong: %s", len(detections), image_path)
        return detections


# ============================================================
# BƯỚC 2: CROP ẢNH THEO BBOX
# ============================================================

def crop_image_by_bbox(image_path: str, bbox_points: list, padding: int = BBOX_PADDING) -> tuple:
    """
    Crop vùng ảnh theo bbox polygon.

    Args:
        image_path: Đường dẫn ảnh gốc
        bbox_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        padding: Padding thêm quanh bbox

    Returns:
        (cropped_pil_image, cropped_bbox_info) hoặc (None, None) nếu lỗi
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    h, w = img.shape[:2]

    # Chuyển bbox points sang numpy array
    pts = np.array(bbox_points, dtype=np.float32)

    # Lấy bounding rectangle
    x_min = int(np.min(pts[:, 0])) - padding
    y_min = int(np.min(pts[:, 1])) - padding
    x_max = int(np.max(pts[:, 0])) + padding
    y_max = int(np.max(pts[:, 1])) + padding

    # Clamp vào bounds ảnh
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Kiểm tra kích thước tối thiểu
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    if crop_w < MIN_CROP_WIDTH or crop_h < MIN_CROP_HEIGHT:
        return None, None

    # Crop ảnh
    cropped = img[y_min:y_max, x_min:x_max]

    # Resize nếu quá lớn
    if crop_w > MAX_CROP_SIZE or crop_h > MAX_CROP_SIZE:
        scale = MAX_CROP_SIZE / max(crop_w, crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Chuyển sang PIL Image
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    bbox_info = {
        "original_bbox": bbox_points,
        "crop_region": [x_min, y_min, x_max, y_max],
        "crop_size": [cropped_pil.width, cropped_pil.height],
    }

    return cropped_pil, bbox_info


# ============================================================
# BƯỚC 3: OLLAMA CLOUD OCR
# ============================================================

class OllamaCloudOCR:
    """Gửi ảnh lên Ollama Cloud API để nhận dạng text"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_url = f"{OLLAMA_BASE_URL}/chat"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
        }

    def _encode_image_base64(self, pil_image: Image.Image) -> str:
        """Chuyển PIL Image sang base64 string"""
        import io
        buffer = io.BytesIO()
        # Lưu ở PNG để giữ chất lượng tốt nhất cho OCR
        pil_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return b64

    def ocr_image(self, pil_image: Image.Image, prompt: str = None) -> str:
        """
        Gửi ảnh lên Ollama Cloud API để OCR.

        Args:
            pil_image: PIL Image cần OCR
            prompt: Prompt tùy chỉnh (mặc định: hướng dẫn OCR tiếng Việt)

        Returns:
            Text được nhận dạng từ ảnh
        """
        if prompt is None:
            prompt = OCR_PROMPT

        image_b64 = self._encode_image_base64(pil_image)

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }

        for attempt in range(1, OLLAMA_RETRY + 1):
            try:
                self.logger.debug(
                    "Gọi Ollama API (lần %d/%d)...", attempt, OLLAMA_RETRY
                )
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=OLLAMA_TIMEOUT,
                )
                response.raise_for_status()

                data = response.json()
                text = data.get("message", {}).get("content", "").strip()

                # Loại bỏ thinking tags nếu model có trả về
                # Qwen models có thể trả về <think...</think_> tags
                import re
                text = re.sub(r'<think[\s\S]*?</think\s*>', '', text).strip()

                return text

            except requests.exceptions.Timeout:
                self.logger.warning(
                    "Timeout Ollama API (lần %d/%d)", attempt, OLLAMA_RETRY
                )
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else "?"
                self.logger.warning(
                    "HTTP Error %s: %s (lần %d/%d)",
                    status_code,
                    str(e)[:200],
                    attempt,
                    OLLAMA_RETRY,
                )
                # Nếu 401/403 thì không retry
                if e.response and e.response.status_code in (401, 403):
                    self.logger.error(
                        "Lỗi xác thực API key! Kiểm tra OLLAMA_API_KEY."
                    )
                    break
                # Nếu 429 rate limit thì đợi lâu hơn
                if e.response and e.response.status_code == 429:
                    wait = OLLAMA_DELAY * attempt * 3
                    self.logger.info("Rate limited, đợi %.1fs...", wait)
                    time.sleep(wait)
                    continue
            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    "Lỗi API: %s (lần %d/%d)", str(e)[:200], attempt, OLLAMA_RETRY
                )

            if attempt < OLLAMA_RETRY:
                wait_time = OLLAMA_DELAY * attempt
                self.logger.info("Đợi %.1fs trước khi retry...", wait_time)
                time.sleep(wait_time)

        self.logger.error("Không thể OCR qua Ollama API sau %d lần thử", OLLAMA_RETRY)
        return ""


# ============================================================
# BƯỚC 4: LƯU TRAINING DATA
# ============================================================

class TrainingDataSaver:
    """
    Lưu training data cho PaddleOCR fine-tune.
    Hỗ trợ 2 định dạng:
    1. Detection training: ảnh gốc + annotation bbox
    2. Recognition training: ảnh crop + text label
    """

    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

        # Tạo cấu trúc thư mục
        self.det_dir = os.path.join(output_dir, "detection")
        self.det_img_dir = os.path.join(self.det_dir, "images")
        self.rec_dir = os.path.join(output_dir, "recognition")
        self.rec_img_dir = os.path.join(self.rec_dir, "images")
        self.raw_dir = os.path.join(output_dir, "raw_crops")

        for d in [self.det_img_dir, self.rec_img_dir, self.raw_dir]:
            os.makedirs(d, exist_ok=True)

        # Annotation files
        self.det_ann_file = os.path.join(self.det_dir, "annotations.txt")
        self.rec_ann_file = os.path.join(self.rec_dir, "annotations.txt")

        # Xóa annotation cũ nếu có
        for f in [self.det_ann_file, self.rec_ann_file]:
            if os.path.exists(f):
                os.remove(f)

        # Stats
        self.stats = {
            "total_images": 0,
            "total_crops": 0,
            "total_ocr_success": 0,
            "total_ocr_failed": 0,
            "total_ocr_empty": 0,
            "skipped_small_crops": 0,
        }

    def save_detection_data(self, image_path: str, detections: list):
        """
        Lưu annotation cho detection training (PP-OCR format).
        Format: image_path\t[{"transcription": "text", "points": [[x1,y1],...]}]
        """
        img_filename = os.path.basename(image_path)

        # Copy ảnh gốc vào detection/images (đổi tên nếu trùng)
        dest_img = os.path.join(self.det_img_dir, img_filename)
        if os.path.exists(dest_img):
            stem = Path(img_filename).stem
            suffix = Path(img_filename).suffix
            timestamp = int(time.time() * 1000) % 100000
            img_filename = f"{stem}_{timestamp}{suffix}"
            dest_img = os.path.join(self.det_img_dir, img_filename)

        shutil.copy2(image_path, dest_img)

        # Tạo annotation theo format PP-OCR
        annotations = []
        for det in detections:
            ann = {
                "transcription": det.get("ollama_text", det.get("paddle_text", "")),
                "points": det["bbox"],
            }
            annotations.append(ann)

        # Ghi annotation
        rel_img_path = os.path.join("images", img_filename)
        with open(self.det_ann_file, "a", encoding="utf-8") as f:
            f.write(f"{rel_img_path}\t{json.dumps(annotations, ensure_ascii=False)}\n")

        self.stats["total_images"] += 1

    def save_recognition_data(self, crop_index: int, crop_image: Image.Image,
                               text: str, source_image: str):
        """
        Lưu crop image + text label cho recognition training.
        Format: image_path\ttext
        """
        # Tạo tên file crop duy nhất
        source_name = Path(source_image).stem
        crop_filename = f"{source_name}_crop_{crop_index:04d}.png"

        # Lưu crop image vào recognition/images
        crop_path = os.path.join(self.rec_img_dir, crop_filename)
        crop_image.save(crop_path, "PNG")

        # Lưu raw crop (backup)
        raw_path = os.path.join(self.raw_dir, crop_filename)
        crop_image.save(raw_path, "PNG")

        # Ghi annotation cho recognition training
        rel_img_path = os.path.join("images", crop_filename)
        with open(self.rec_ann_file, "a", encoding="utf-8") as f:
            f.write(f"{rel_img_path}\t{text}\n")

        self.stats["total_crops"] += 1
        if text:
            self.stats["total_ocr_success"] += 1
        else:
            self.stats["total_ocr_empty"] += 1

        return crop_path

    def save_summary(self):
        """Lưu file tổng kết pipeline"""
        summary_path = os.path.join(self.output_dir, "pipeline_summary.json")
        self.stats["generated_at"] = datetime.now().isoformat()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        self.logger.info("=" * 60)
        self.logger.info("TỔNG KẾT PIPELINE")
        self.logger.info("  Ảnh xử lý:          %d", self.stats["total_images"])
        self.logger.info("  Crops tạo ra:       %d", self.stats["total_crops"])
        self.logger.info("  OCR thành công:     %d", self.stats["total_ocr_success"])
        self.logger.info("  OCR text rỗng:      %d", self.stats["total_ocr_empty"])
        self.logger.info("  OCR thất bại:       %d", self.stats["total_ocr_failed"])
        self.logger.info("  Crops bỏ qua (nhỏ): %d", self.stats["skipped_small_crops"])
        self.logger.info("=" * 60)
        self.logger.info("Training data đã lưu tại: %s", self.output_dir)
        self.logger.info("  Detection:   %s/", self.det_dir)
        self.logger.info("  Recognition: %s/", self.rec_dir)


# ============================================================
# PIPELINE CHÍNH
# ============================================================

class VietnameseOCRPipeline:
    """Pipeline chính: PaddleOCR bbox → Crop → Ollama Cloud OCR → Save"""

    def __init__(self, output_dir: str, skip_ollama: bool = False,
                 padding: int = BBOX_PADDING):
        self.output_dir = output_dir
        self.skip_ollama = skip_ollama
        self.padding = padding
        self.logger = setup_logging(output_dir)

        # Khởi tạo các components (lazy init)
        self.paddle_extractor = PaddleOCRExtractor(self.logger)
        self.ollama_ocr = OllamaCloudOCR(self.logger) if not skip_ollama else None
        self.saver = TrainingDataSaver(output_dir, self.logger)

    def process_single_image(self, image_path: str):
        """Xử lý một ảnh qua toàn bộ pipeline"""
        self.logger.info("=" * 60)
        self.logger.info("XỬ LÝ: %s", image_path)
        self.logger.info("=" * 60)

        if not os.path.exists(image_path):
            self.logger.error("File không tồn tại: %s", image_path)
            return

        # ---- Bước 1: PaddleOCR phát hiện bbox ----
        detections = self.paddle_extractor.detect_bboxes(image_path)
        if not detections:
            self.logger.warning("Không có bbox nào được phát hiện, bỏ qua: %s", image_path)
            return

        # ---- Bước 2 & 3: Crop + OCR từng bbox ----
        for idx, det in enumerate(detections):
            self.logger.info(
                "  [%d/%d] Conf=%.4f | PaddleOCR: '%s'",
                idx + 1, len(detections),
                det["confidence"],
                det["paddle_text"][:80],
            )

            # Crop ảnh theo bbox
            crop_img, bbox_info = crop_image_by_bbox(
                image_path, det["bbox"], padding=self.padding
            )
            if crop_img is None:
                self.logger.warning("  Crop bị bỏ qua (kích thước quá nhỏ)")
                self.saver.stats["skipped_small_crops"] += 1
                continue

            # OCR qua Ollama Cloud (hoặc dùng text PaddleOCR nếu skip)
            if self.skip_ollama:
                ollama_text = det["paddle_text"]
                self.logger.info("  [SKIP OLLAMA] Dùng PaddleOCR text: '%s'",
                                 ollama_text[:80])
            else:
                ollama_text = self.ollama_ocr.ocr_image(crop_img)
                if ollama_text:
                    self.logger.info("  Ollama OCR: '%s'", ollama_text[:80])
                else:
                    self.logger.warning(
                        "  Ollama OCR trả về rỗng, dùng PaddleOCR text thay thế"
                    )
                    ollama_text = det["paddle_text"]
                    self.saver.stats["total_ocr_failed"] += 1

            # Lưu text vào detection dict
            det["ollama_text"] = ollama_text
            det["crop_info"] = bbox_info

            # ---- Bước 4: Lưu training data ----
            self.saver.save_recognition_data(
                crop_index=idx,
                crop_image=crop_img,
                text=ollama_text,
                source_image=image_path,
            )

            # Delay giữa các API call để tránh rate limit
            if not self.skip_ollama and idx < len(detections) - 1:
                time.sleep(OLLAMA_DELAY)

        # Lưu detection annotation cho ảnh gốc
        self.saver.save_detection_data(image_path, detections)

    def process_directory(self, input_dir: str, extensions: tuple = None):
        """Xử lý tất cả ảnh trong thư mục"""
        if extensions is None:
            extensions = (
                ".jpg", ".jpeg", ".png", ".bmp",
                ".tiff", ".tif", ".webp", ".pdf"
            )

        self.logger.info("Quét thư mục: %s", input_dir)
        image_files = []
        for f in sorted(os.listdir(input_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                image_files.append(os.path.join(input_dir, f))

        if not image_files:
            self.logger.error("Không tìm thấy ảnh nào trong: %s", input_dir)
            return

        self.logger.info("Tìm thấy %d ảnh để xử lý", len(image_files))

        for i, img_path in enumerate(image_files):
            self.logger.info("\n[%d/%d] Đang xử lý: %s",
                             i + 1, len(image_files), img_path)
            try:
                self.process_single_image(img_path)
            except Exception as e:
                self.logger.error("Lỗi xử lý %s: %s", img_path, str(e),
                                  exc_info=True)

    def run(self, input_path: str):
        """Chạy pipeline với input (file hoặc thư mục)"""
        self.logger.info("VIETNAMESE OCR TRAINING DATA PIPELINE")
        self.logger.info("Input: %s", input_path)
        self.logger.info("Output: %s", self.output_dir)
        self.logger.info("Ollama model: %s", OLLAMA_MODEL)
        self.logger.info("Skip Ollama: %s", self.skip_ollama)
        self.logger.info("BBox padding: %d px", self.padding)

        # Khởi tạo PaddleOCR
        self.paddle_extractor.initialize()

        if os.path.isfile(input_path):
            self.process_single_image(input_path)
        elif os.path.isdir(input_path):
            self.process_directory(input_path)
        else:
            self.logger.error("Input không hợp lệ: %s", input_path)
            return

        # Lưu tổng kết
        self.saver.save_summary()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese OCR Training Data Pipeline - "
                    "Tạo dữ liệu training finetune PaddleOCR tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Xử lý một ảnh
  python vietnamese_ocr_pipeline.py --input image.png --output_dir ./training_data

  # Xử lý thư mục ảnh
  python vietnamese_ocr_pipeline.py --input_dir ./input_images --output_dir ./training_data

  # Chỉ chạy PaddleOCR (không gọi Ollama Cloud - dùng để test)
  python vietnamese_ocr_pipeline.py --input_dir ./input_images --output_dir ./training_data --skip_ollama

  # Dùng model Ollama khác
  OLLAMA_MODEL="qwen2.5:72b-cloud" python vietnamese_ocr_pipeline.py --input image.png --output_dir ./out

Environment variables:
  OLLAMA_API_KEY   - API key cho Ollama Cloud (mặc định: đã cấu hình sẵn)
  OLLAMA_BASE_URL  - Base URL Ollama API (mặc định: https://ollama.com/api)
  OLLAMA_MODEL     - Model Ollama (mặc định: qwen3.5:397b-cloud)
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Đường dẫn đến file ảnh cần xử lý",
    )
    parser.add_argument(
        "--input_dir", "-d",
        type=str,
        help="Đường dẫn đến thư mục chứa ảnh",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./training_output",
        help="Thư mục lưu training data (mặc định: ./training_output)",
    )
    parser.add_argument(
        "--skip_ollama",
        action="store_true",
        help="Bỏ qua bước Ollama Cloud OCR (chỉ dùng text từ PaddleOCR)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=BBOX_PADDING,
        help=f"Padding quanh bbox khi crop (mặc định: {BBOX_PADDING}px)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input and not args.input_dir:
        parser.error("Phải chỉ định --input hoặc --input_dir")

    input_path = args.input or args.input_dir

    # Chạy pipeline
    pipeline = VietnameseOCRPipeline(
        output_dir=args.output_dir,
        skip_ollama=args.skip_ollama,
        padding=args.padding,
    )
    pipeline.run(input_path)


if __name__ == "__main__":
    main()
