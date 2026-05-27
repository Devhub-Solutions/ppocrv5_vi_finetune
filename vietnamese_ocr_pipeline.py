import os
import cv2
import numpy as np
import time
from PPOCRv5Detection import PPOCRv5Detection
from VietnameseRecognition import VietnameseRecognition

def crop_image(img, box):
    """Cắt ảnh theo bounding box xoay"""
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    pts1 = np.float32(box)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (width, height))
    return dst

def vietnamese_ocr(image_path):
    # 1. Khởi tạo
    det_model_path = "/home/ubuntu/ppocrv5_vi_finetune/weights/ppocrv5-server-det.onnx"
    rec_model_dir = "/home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/rec_inference"
    dict_path = "/home/ubuntu/ppocrv5_vi_finetune/vietnamese_dict.txt"
    
    detector = PPOCRv5Detection(onnx_path=det_model_path)
    recognizer = VietnameseRecognition(model_dir=rec_model_dir, character_dict_path=dict_path)
    
    # 2. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None, 0
        
    # 3. Detection
    start_time = time.time()
    # Sử dụng hàm __call__ của detector để lấy boxes
    dt_boxes = detector(img)
    
    # 4. Crop và Recognition
    if len(dt_boxes) == 0:
        return [], time.time() - start_time
        
    img_crops = []
    for box in dt_boxes:
        crop = crop_image(img, box)
        img_crops.append(crop)
        
    texts, confs = recognizer(img_crops)
    
    duration = time.time() - start_time
    
    # 5. Kết quả
    results = []
    for i in range(len(texts)):
        results.append({
            "box": dt_boxes[i].tolist(),
            "text": texts[i],
            "confidence": float(confs[i])
        })
        
    return results, duration

if __name__ == "__main__":
    # Test thử
    image_dir = "/home/ubuntu/ppocrv5_vi_finetune/vietnamese_images"
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_files:
        test_img = image_files[0]
        print(f"--- Testing Vietnamese OCR on: {test_img} ---")
        
        results, duration = vietnamese_ocr(test_img)
        
        if results is not None:
            print(f"Tìm thấy {len(results)} vùng văn bản.")
            print(f"Thời gian xử lý: {duration:.2f}s")
            print("\n--- KẾT QUẢ OCR ---")
            for res in results:
                print(f"Văn bản: {res['text']} (Độ tin cậy: {res['confidence']:.2f})")
    else:
        print("Không tìm thấy ảnh để test.")
