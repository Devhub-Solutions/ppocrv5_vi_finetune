import os
import cv2
import numpy as np
import base64
import requests
import json
import csv
from pathlib import Path
from PIL import Image
from pyclipper import ET_CLOSEDPOLYGON, JT_ROUND, PyclipperOffset
from shapely.geometry import Polygon


class PPOCRv5Detection:
    """PPOCRv5 Server/Mobile detection model wrapper"""
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None

            from onnxruntime import InferenceSession
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available]
            self.session = InferenceSession(onnx_path, providers=providers or available)
        
        self.inputs = self.session.get_inputs()[0]
        self.input_name = self.inputs.name
        self.output_name = self.session.get_outputs()[0].name
        
        self.min_size = 3
        self.max_size = 960
        self.box_thresh = 0.3  # Lower threshold for PPOCRv5
        self.mask_thresh = 0.3
        
        # PPOCRv5 uses ImageNet normalization
        self.mean = np.array([123.675, 116.28, 103.53]).reshape(1, -1).astype('float64')
        self.std = 1 / np.array([58.395, 57.12, 57.375]).reshape(1, -1).astype('float64')

    def filter_polygon(self, points, shape):
        width, height = shape[1], shape[0]
        filtered = []
        for point in points:
            if type(point) is list:
                point = np.array(point)
            point = self.clockwise_order(point)
            point = self.clip(point, height, width)
            w = int(np.linalg.norm(point[0] - point[1]))
            h = int(np.linalg.norm(point[0] - point[3]))
            if w <= 3 or h <= 3:
                continue
            filtered.append(point.astype("int32"))
        return np.array(filtered, dtype="int32")

    def boxes_from_bitmap(self, output, mask, dest_width, dest_height):
        mask = (mask * 255).astype(np.uint8)
        height, width = mask.shape
        outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[0] if len(outs) == 2 else outs[1]
        boxes, scores = [], []
        for contour in contours:
            points, min_side = self.get_min_boxes(contour)
            if min_side < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score(output, contour)
            if self.box_thresh > score:
                continue
            polygon = Polygon(points)
            distance = polygon.area / polygon.length
            offset = PyclipperOffset()
            offset.AddPath(points, JT_ROUND, ET_CLOSEDPOLYGON)
            points = np.array(offset.Execute(distance * 1.5)).reshape((-1, 1, 2))
            box, min_side = self.get_min_boxes(points)
            if min_side < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    @staticmethod
    def get_min_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score(bitmap, contour):
        h, w = bitmap.shape[:2]
        contour = contour.copy().reshape(-1, 2)
        x1 = np.clip(np.min(contour[:, 0]), 0, w - 1)
        y1 = np.clip(np.min(contour[:, 1]), 0, h - 1)
        x2 = np.clip(np.max(contour[:, 0]), 0, w - 1)
        y2 = np.clip(np.max(contour[:, 1]), 0, h - 1)
        mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
        contour[:, 0] -= x1
        contour[:, 1] -= y1
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), color=(1, 1))
        return cv2.mean(bitmap[y1:y2 + 1, x1:x2 + 1], mask)[0]

    @staticmethod
    def clockwise_order(point):
        poly = np.zeros((4, 2), dtype="float32")
        s = point.sum(axis=1)
        poly[0] = point[np.argmin(s)]
        poly[2] = point[np.argmax(s)]
        tmp = np.delete(point, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        poly[1] = tmp[np.argmin(diff)]
        poly[3] = tmp[np.argmax(diff)]
        return poly

    @staticmethod
    def clip(points, h, w):
        for i in range(points.shape[0]):
            points[i, 0] = int(min(max(points[i, 0], 0), w - 1))
            points[i, 1] = int(min(max(points[i, 1], 0), h - 1))
        return points

    def resize(self, image):
        h, w = image.shape[:2]
        ratio = float(self.max_size) / max(h, w) if max(h, w) > self.max_size else 1.0
        resize_h = max(int(round(int(h * ratio) / 32) * 32), 32)
        resize_w = max(int(round(int(w * ratio) / 32) * 32), 32)
        return cv2.resize(image, (resize_w, resize_h))

    @staticmethod
    def zero_pad(image):
        h, w, c = image.shape
        pad = np.zeros((max(32, h), max(32, w), c), np.uint8)
        pad[:h, :w, :] = image
        return pad

    def __call__(self, x):
        h, w = x.shape[:2]
        if sum([h, w]) < 64:
            x = self.zero_pad(x)
        x = self.resize(x).astype('float32')
        cv2.subtract(x, self.mean, x)
        cv2.multiply(x, self.std, x)
        x = np.expand_dims(x.transpose((2, 0, 1)), axis=0)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: x})
        output = outputs[0][0, 0]  # Extract from batch
        
        boxes, scores = self.boxes_from_bitmap(output, output > self.mask_thresh, w, h)
        return self.filter_polygon(boxes, (h, w))


# Keep old name for backward compatibility
Detection = PPOCRv5Detection


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


def save_dataset_item(crop, ocr_result, dataset_dir, image_name, box_idx, metadata_list):
    """Save crop image and record metadata"""
    crop_filename = f"{Path(image_name).stem}_box_{box_idx:03d}.png"
    crop_path = os.path.join(dataset_dir, crop_filename)
    
    # Save crop
    cv2.imwrite(crop_path, crop)
    
    # Record metadata
    text = ocr_result.get("text", "") if ocr_result else ""
    language = ocr_result.get("language", "unknown") if ocr_result else "unknown"
    confidence = ocr_result.get("confidence", 0.0) if ocr_result else 0.0
    
    metadata_list.append({
        "image_file": crop_filename,
        "text": text,
        "language": language,
        "confidence": confidence,
        "source_image": image_name
    })
    
    return crop_path


def create_dataset(image_path, det_model_path, dataset_base_dir="D:\\airflow\\dataset", 
                   api_url="http://qwen-3vl.devhub.io.vn/v1/chat/completions"):
    """Detect text boxes, extract crops, and create labeled dataset"""
    
    # Create dataset directory structure
    dataset_dir = os.path.join(dataset_base_dir, "crops")
    annotations_dir = os.path.join(dataset_base_dir, "annotations")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Initialize detector
    det = Detection(det_model_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    image_name = Path(image_path).name
    metadata_list = []
    
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
        else:
            print(f"  Failed to get OCR result")
            ocr_result = None
        
        # Save to dataset
        crop_path = save_dataset_item(crop, ocr_result, dataset_dir, image_name, i, metadata_list)
        print(f"  Saved: {crop_path}")
    
    # Save metadata as CSV
    csv_path = os.path.join(annotations_dir, f"{Path(image_path).stem}_annotations.csv")
    if metadata_list:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_file', 'text', 'language', 'confidence', 'source_image'])
            writer.writeheader()
            writer.writerows(metadata_list)
        print(f"\nSaved annotations to: {csv_path}")
    
    # Save metadata as JSON
    json_path = os.path.join(annotations_dir, f"{Path(image_name).stem}_annotations.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON annotations to: {json_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset created successfully!")
    print(f"Total items: {len(metadata_list)}")
    print(f"Crops directory: {dataset_dir}")
    print(f"Annotations directory: {annotations_dir}")
    print(f"{'='*50}")
    
    return dataset_base_dir, metadata_list


def main():
    det_model_path = r"./weights/ppocrv5-server-det.onnx"
    image_path = r"./image.png"
    output_path = r"./image_ocr.png"
    dataset_base_dir = r"./dataset"
    api_url = "http://qwen-3vl.devhub.io.vn/v1/chat/completions"

    # Tạo tập dữ liệu và thực hiện OCR cho từng vùng
    dataset_dir, metadata = create_dataset(image_path, det_model_path, dataset_base_dir, api_url)
    
    # Vẽ kết quả phát hiện và nội dung OCR lên ảnh gốc
    det = Detection(det_model_path)
    img = cv2.imread(image_path)
    boxes = det(img)
    
    for i, box in enumerate(boxes):
        if i < len(metadata):
            draw_bbox_with_text(img, box, metadata[i].get("text", ""), metadata[i].get("confidence", 0.0))
    
    cv2.imwrite(output_path, img)
    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()
