import os
import cv2
import numpy as np
import base64
import requests
import json
import csv
import time
from pathlib import Path
from PPOCRv5Detection import PPOCRv5Detection, crop_bbox, encode_crop_to_base64

def ocr_via_api(crop_base64, api_url="http://qwen-3vl.devhub.io.vn/v1/chat/completions"):
    """Send base64 encoded crop to Qwen VL API for OCR"""
    if not crop_base64:
        return None
    
    payload = {
        "model": "models/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Hãy thực hiện OCR cho hình ảnh này.

Yêu cầu:
- Trích xuất toàn bộ văn bản trong ảnh.
- Giữ nguyên xuống dòng và định dạng cơ bản.
- Phát hiện ngôn ngữ chính của văn bản.
- Ước lượng độ chính xác OCR dưới dạng số từ 0 đến 1.

Chỉ trả về JSON hợp lệ, không giải thích thêm.

Định dạng JSON:
{
  "text": "...",
  "language": "...",
  "confidence": 0.95
}""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{crop_base64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.1,
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            # Clean content if it contains markdown code blocks
            if content.startswith("```json"):
                content = content.strip("```json").strip("```").strip()
            elif content.startswith("```"):
                content = content.strip("```").strip()
                
            try:
                ocr_result = json.loads(content)
                return ocr_result
            except json.JSONDecodeError:
                return {"text": content, "language": "unknown", "confidence": 0.5}
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

def load_processed_logs(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_processed_logs(log_file, logs):
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

def main():
    # Paths
    det_model_path = "./weights/ppocrv5-server-det.onnx"
    images_dir = "./vietnamese_images"
    output_dir = "./dataset_output"
    crops_dir = os.path.join(output_dir, "images")
    log_file = os.path.join(output_dir, "processing_log.json")
    ann_file = os.path.join(output_dir, "annotations.txt")
    api_url = "http://qwen-3vl.devhub.io.vn/v1/chat/completions"

    # Create directories
    os.makedirs(crops_dir, exist_ok=True)
    
    # Initialize detector
    det = PPOCRv5Detection(det_model_path)
    
    # Load logs
    processed_logs = load_processed_logs(log_file)
    
    # Get image list
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
    images.sort()
    
    print(f"Found {len(images)} images in {images_dir}")
    
    # Open annotations file in append mode
    with open(ann_file, 'a', encoding='utf-8') as f_ann:
        for idx, img_name in enumerate(images):
            if img_name in processed_logs:
                print(f"[{idx+1}/{len(images)}] Skipping {img_name} (already processed)")
                continue
            
            img_path = os.path.join(images_dir, img_name)
            print(f"[{idx+1}/{len(images)}] Processing {img_name}...")
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Error: Could not read {img_name}")
                continue
            
            # Detect boxes
            try:
                boxes = det(img)
            except Exception as e:
                print(f"  Error during detection: {e}")
                continue
                
            print(f"  Found {len(boxes)} text boxes")
            
            box_results = []
            for i, box in enumerate(boxes):
                crop = crop_bbox(img, box)
                if crop.size == 0:
                    continue
                
                crop_b64 = encode_crop_to_base64(crop)
                ocr_result = ocr_via_api(crop_b64, api_url)
                
                if ocr_result and ocr_result.get("text"):
                    text = ocr_result["text"].replace('\n', ' ').strip()
                    crop_filename = f"{Path(img_name).stem}_box_{i:03d}.png"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    
                    cv2.imwrite(crop_path, crop)
                    
                    # Write to annotations.txt (format: relative_path\ttext)
                    f_ann.write(f"images/{crop_filename}\t{text}\n")
                    f_ann.flush()
                    
                    box_results.append({
                        "box_idx": i,
                        "crop_file": crop_filename,
                        "text": text,
                        "confidence": ocr_result.get("confidence", 0)
                    })
                    print(f"    Box {i}: {text[:30]}...")
                else:
                    print(f"    Box {i}: OCR failed or empty")
            
            # Update log
            processed_logs[img_name] = {
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "boxes_found": len(boxes),
                "boxes_saved": len(box_results),
                "details": box_results
            }
            save_processed_logs(log_file, processed_logs)
            
            # Optional: Limit for testing
            # if len(processed_logs) >= 5:
            #     break

    print("\nProcessing complete!")
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    main()
