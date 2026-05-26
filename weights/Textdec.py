import cv2
import numpy as np
import base64
import requests
import json
from pathlib import Path
from PIL import Image
from weights.detection.index import PPOCRv5Detection as Detection


def get_mime_type(image_path):
    """Determine MIME type from file extension"""
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(Path(image_path).suffix.lower(), "image/png")


def crop_bbox(image, box):
    """Crop image region using bbox coordinates"""
    x_min = max(0, int(np.min(box[:, 0])))
    y_min = max(0, int(np.min(box[:, 1])))
    x_max = min(image.shape[1], int(np.max(box[:, 0])))
    y_max = min(image.shape[0], int(np.max(box[:, 1])))
    return image[y_min:y_max, x_min:x_max]


def encode_crop_to_base64(crop):
    """Encode cropped image to base64"""
    if crop.size == 0:
        return None
    success, buffer = cv2.imencode('.png', crop)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')


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


def draw_bbox_with_text(image, box, text, confidence):
    """Draw bounding box and OCR text on image"""
    box_int = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [box_int], True, (0, 255, 0), 2)
    
    if text:
        label = f"{text} ({confidence:.2f})"
        x, y = int(box[0][0]), max(10, int(box[0][1]) - 8)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - th - 8), (x + tw + 6, y + 4), (0, 255, 0), -1)
        cv2.putText(image, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    det_model_path = r"D:\airflow\OCR\weights\ppocrv5-server-det.onnx"
    image_path = r"D:\airflow\image.png"
    output_path = r"D:\airflow\image_ocr.png"
    api_url = "http://qwen-3vl.devhub.io.vn/v1/chat/completions"

    # Initialize detector
    det = Detection(det_model_path)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Detect text boxes
    boxes = det(img)
    print(f"Found {len(boxes)} text boxes")

    # Process each box
    for i, box in enumerate(boxes):
        print(f"\nProcessing box {i+1}/{len(boxes)}...")
        
        # Crop the region
        crop = crop_bbox(img, box)
        if crop.size == 0:
            continue
        
        # Encode to base64
        crop_b64 = encode_crop_to_base64(crop)
        if not crop_b64:
            continue
        
        # Send to API for OCR
        ocr_result = ocr_via_api(crop_b64, api_url)
        
        if ocr_result:
            text = ocr_result.get("text", "")
            confidence = ocr_result.get("confidence", 0.0)
            language = ocr_result.get("language", "unknown")
            
            print(f"  Text: {text}")
            print(f"  Language: {language}")
            print(f"  Confidence: {confidence:.2f}")
            
            # Draw on image
            draw_bbox_with_text(img, box, text, confidence)
        else:
            print(f"  Failed to get OCR result")

    # Save result
    cv2.imwrite(output_path, img)
    print(f"\nSaved result to: {output_path}")
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()


if __name__ == "__main__":
    main()