# Vietnamese OCR Fine-tuning Pipeline

Đây là một pipeline tự động để tạo dataset cho việc fine-tune mô hình OCR tiếng Việt sử dụng PaddleOCR, kết hợp với phát hiện văn bản bằng PPOCRv5Detection và nhận dạng văn bản bằng API Qwen-3VL.

## Mục lục
1.  [Tổng quan](#1-tổng-quan)
2.  [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3.  [Cài đặt](#3-cài-đặt)
4.  [Tạo Dataset](#4-tạo-dataset)
5.  [Fine-tune Model](#5-fine-tune-model)
6.  [Lưu ý về tài nguyên](#6-lưu-ý-về-tài-nguyên)

## 1. Tổng quan

Pipeline này giúp tự động hóa quá trình tạo dữ liệu huấn luyện cho mô hình nhận dạng ký tự quang học (OCR) tiếng Việt. Nó bao gồm các bước chính:

*   **Phát hiện văn bản**: Sử dụng mô hình PPOCRv5Detection để xác định vị trí các hộp giới hạn (bounding box) của văn bản trong ảnh.
*   **Cắt ảnh (Crop)**: Cắt các vùng ảnh chứa văn bản dựa trên các hộp giới hạn đã phát hiện.
*   **Nhận dạng văn bản (OCR)**: Gửi các ảnh đã cắt lên API Qwen-3VL để nhận dạng văn bản, đảm bảo độ chính xác cao cho tiếng Việt.
*   **Lưu trữ Dataset**: Lưu trữ các ảnh đã cắt cùng với nhãn văn bản tương ứng theo định dạng PaddleOCR, sẵn sàng cho quá trình fine-tune.
*   **Hệ thống Log**: Ghi lại tiến độ xử lý để tránh trùng lặp và cho phép tiếp tục quá trình nếu bị gián đoạn.
*   **Fine-tune**: Hướng dẫn cách sử dụng dữ liệu đã tạo để fine-tune mô hình nhận dạng của PaddleOCR.

## 2. Cấu trúc thư mục

```
ppocrv5_vi_finetune/
├── build_dataset.py              # Script chính để tạo dataset
├── PPOCRv5Detection.py           # Module phát hiện văn bản PPOCRv5
├── Classification.py             # (Chưa sử dụng trong pipeline này)
├── 1_generate_training_data.py   # (Phiên bản cũ, không dùng)
├── 2_finetune_recognition.py     # Script để fine-tune mô hình nhận dạng
├── 3_finetune_detection.py       # (Chưa sử dụng trong pipeline này)
├── 4_inference.py                # (Chưa sử dụng trong pipeline này)
├── vietnamese_images/            # Thư mục chứa ảnh đầu vào để tạo dataset
├── weights/                      # Chứa các model trọng số (ví dụ: ppocrv5-server-det.onnx)
├── vietnamese_dict.txt           # Từ điển ký tự tiếng Việt cho PaddleOCR
├── dataset_output/               # Thư mục đầu ra của quá trình tạo dataset
│   ├── images/                   # Các ảnh đã cắt (cropped images)
│   ├── annotations.txt           # File annotation cho PaddleOCR recognition training
│   └── processing_log.json       # Log các ảnh đã xử lý
├── finetuned_models/             # Thư mục chứa các model đã fine-tune
│   └── rec/                      # Model nhận dạng đã fine-tune
│       ├── config/               # File cấu hình training
│       ├── rec_model/            # Trọng số model
│       └── rec_inference/        # Model đã export để inference
└── README.md                     # File hướng dẫn này
```

## 3. Cài đặt

Để chạy pipeline này, bạn cần cài đặt các thư viện Python sau:

```bash
sudo pip3 install onnxruntime pyclipper shapely paddlepaddle paddleocr opencv-python
```

Ngoài ra, bạn cần clone repository PaddleOCR để sử dụng các công cụ training của họ:

```bash
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git /home/ubuntu/PaddleOCR
cd /home/ubuntu/PaddleOCR
sudo pip install -r requirements.txt
```

## 4. Tạo Dataset

Script `build_dataset.py` sẽ tự động phát hiện văn bản, cắt ảnh và gọi API Qwen-3VL để nhận dạng. Kết quả sẽ được lưu vào thư mục `dataset_output`.

**Chuẩn bị:**

1.  Đặt các ảnh tiếng Việt của bạn vào thư mục `./vietnamese_images/`.
2.  Đảm bảo file trọng số `ppocrv5-server-det.onnx` có trong thư mục `./weights/`.

**Chạy script:**

```bash
cd /home/ubuntu/ppocrv5_vi_finetune
python3 build_dataset.py
```

*Lưu ý*: API Qwen-3VL (`http://qwen-3vl.devhub.io.vn/v1/chat/completions`) được hardcode trong `build_dataset.py`. Nếu bạn có API khác, hãy chỉnh sửa file này.

`processing_log.json` sẽ ghi lại các ảnh đã được xử lý. Nếu script bị dừng và chạy lại, nó sẽ tiếp tục từ ảnh cuối cùng chưa được xử lý.

## 5. Fine-tune Model

Sau khi có đủ dữ liệu trong `dataset_output/`, bạn có thể tiến hành fine-tune mô hình nhận dạng của PaddleOCR.

**Bước 1: Chuẩn bị dữ liệu và tạo file cấu hình training**

Chạy script `2_finetune_recognition.py` để chia dữ liệu thành tập train/validation và tạo file cấu hình YAML cho PaddleOCR.

```bash
cd /home/ubuntu/ppocrv5_vi_finetune
python3 2_finetune_recognition.py --train_dir ./dataset_output --output_dir ./finetuned_models/rec --epochs 100 --batch_size 4 --lr 0.0005 --use_cpu
```

*   `--train_dir`: Thư mục chứa dữ liệu đã tạo (`dataset_output`).
*   `--output_dir`: Thư mục để lưu model đã fine-tune.
*   `--epochs`: Số lượng epoch để huấn luyện (ví dụ: 100).
*   `--batch_size`: Kích thước batch (ví dụ: 4 để tiết kiệm RAM).
*   `--lr`: Learning rate.
*   `--use_cpu`: Sử dụng CPU để huấn luyện (nếu không có GPU).

**Bước 2: Chạy quá trình training**

Sử dụng công cụ `train.py` từ repository PaddleOCR đã clone để bắt đầu huấn luyện.

```bash
cd /home/ubuntu/PaddleOCR
export PYTHONPATH=$PYTHONPATH:.
python3 tools/train.py -c /home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/config/rec_vi_config.yml
```

Sau khi training hoàn tất, model đã fine-tune sẽ được lưu trong thư mục `./finetuned_models/rec/rec_model/`.

**Bước 3: Export model để inference**

Để sử dụng model đã fine-tune cho việc inference, bạn cần export nó:

```bash
cd /home/ubuntu/PaddleOCR
python3 tools/export_model.py -c /home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/config/rec_vi_config.yml -o Global.pretrained_model=/home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/rec_model/best_accuracy Global.save_inference_dir=/home/ubuntu/ppocrv5_vi_finetune/finetuned_models/rec/rec_inference
```

Model đã export sẽ nằm trong thư mục `./finetuned_models/rec/rec_inference/`.

## 6. Lưu ý về tài nguyên

Quá trình tạo dataset và fine-tune mô hình có thể tốn nhiều tài nguyên (RAM, CPU, GPU). Trong môi trường sandbox này, tôi đã điều chỉnh các tham số như `batch_size` và `num_workers` xuống thấp để tránh lỗi tràn bộ nhớ. Nếu bạn chạy trên môi trường có GPU và nhiều RAM hơn, bạn có thể tăng các giá trị này để tăng tốc độ xử lý và huấn luyện.
