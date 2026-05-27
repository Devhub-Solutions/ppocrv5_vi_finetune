import os
import sys
import cv2
import numpy as np
import time

# Thêm PaddleOCR vào path
sys.path.append("/home/ubuntu/PaddleOCR")

from paddleocr import PaddleOCR
from PPOCRv5Detection import PPOCRv5Detection as TextDetector

def main():
    # 1. Khởi tạo mô hình
    print("--- Khởi tạo mô hình ---")
    
    # Khởi tạo Detector (sử dụng PP-OCRv5 ONNX)
    det_model_path = "/home/ubuntu/ppocrv5_vi_finetune/weights/ppocrv5-server-det.onnx"
    detector = TextDetector(onnx_path=det_model_path)
    
    # Khởi tạo Recognizer với model đã fine-tune
    # Lưu ý: PaddleOCR class hỗ trợ truyền rec_model_dir
    rec_model_dir = "/home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/rec_inference"
    rec_char_dict = "/home/ubuntu/ppocrv5_vi_finetune/vietnamese_dict.txt"
    
    ocr_engine = PaddleOCR(
        text_recognition_model_dir=rec_model_dir,
        use_textline_orientation=False
    )
    
    # 2. Chọn ảnh test
    image_dir = "/home/ubuntu/ppocrv5_vi_finetune/vietnamese_images"
    if not os.path.exists(image_dir):
        print(f"Không tìm thấy thư mục ảnh: {image_dir}")
        return
        
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Không có ảnh để test.")
        return
        
    test_image_path = image_files[0]
    print(f"Đang test ảnh: {test_image_path}")
    
    # 3. Chạy Inference
    img = cv2.imread(test_image_path)
    if img is None:
        print("Không thể đọc ảnh.")
        return
        
    start_time = time.time()
    
    # Bước 1: Detect vùng văn bản
    dt_boxes = detector.detect(img)
    print(f"Tìm thấy {len(dt_boxes)} vùng văn bản.")
    
    # Bước 2: Nhận diện từng vùng
    results = []
    for box in dt_boxes:
        # Cắt ảnh theo box (đơn giản hóa)
        # PaddleOCR hỗ trợ nhận diện từ ảnh đã cắt hoặc toàn bộ ảnh với boxes
        # Ở đây ta dùng ocr_engine trực tiếp với ảnh và boxes để tối ưu
        pass
    
    # Chạy full OCR engine
    result = ocr_engine.ocr(test_image_path, cls=False)
    
    end_time = time.time()
    print(f"Thời gian xử lý: {end_time - start_time:.2f}s")
    
    # 4. Hiển thị kết quả
    print("\n--- KẾT QUẢ OCR ---")
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            print(f"Văn bản: {text} (Độ tin cậy: {confidence:.2f})")
    else:
        print("Không tìm thấy văn bản hoặc lỗi nhận diện.")

if __name__ == "__main__":
    main()
