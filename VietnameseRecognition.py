import os
import cv2
import numpy as np
import math
import onnxruntime as ort

class CTCDecoder:
    def __init__(self, character_dict_path):
        self.character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        
        # PaddleOCR CTC decoder: index 0 is blank, then dict characters
        self.character_str = ["blank"] + self.character_str
        # Space is usually at the end or included in dict
        if " " not in self.character_str:
            self.character_str.append(" ")

    def __call__(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text_results = []
        conf_results = []
        for i in range(len(preds_idx)):
            char_list = []
            conf_list = []
            for j in range(len(preds_idx[i])):
                if preds_idx[i][j] != 0 and (not (j > 0 and preds_idx[i][j - 1] == preds_idx[i][j])):
                    char_list.append(self.character_str[preds_idx[i][j]])
                    conf_list.append(preds_prob[i][j])
            if len(conf_list) == 0:
                conf_list = [0.0]
            text_results.append(''.join(char_list))
            conf_results.append(np.mean(conf_list))
        return text_results, conf_results

class VietnameseRecognition:
    def __init__(self, model_dir, character_dict_path, use_gpu=False):
        import paddle.inference as paddle_infer
        
        model_file = os.path.join(model_dir, "inference.pdmodel")
        params_file = os.path.join(model_dir, "inference.pdiparams")
        
        config = paddle_infer.Config(model_file, params_file)
        if use_gpu:
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(4)
            
        config.enable_memory_optim()
        self.predictor = paddle_infer.create_predictor(config)
        
        self.input_names = self.predictor.get_input_names()
        self.input_tensor = self.predictor.get_input_handle(self.input_names[0])
        self.output_names = self.predictor.get_output_names()
        self.output_tensor = self.predictor.get_output_handle(self.output_names[0])
        
        self.input_shape = [3, 48, 320]
        self.ctc_decoder = CTCDecoder(character_dict_path)

    def resize(self, image, max_wh_ratio):
        input_h, input_w = self.input_shape[1], self.input_shape[2]
        assert self.input_shape[0] == image.shape[2]
        
        input_w = int((input_h * max_wh_ratio))
        # Lấy shape từ input tensor của Paddle Inference
        tensor_shape = self.input_tensor.shape()
        if len(tensor_shape) > 3:
            w_shape = tensor_shape[3]
            if w_shape > 0:
                input_w = w_shape
            
        h, w = image.shape[:2]
        ratio = w / float(h)
        if math.ceil(input_h * ratio) > input_w:
            resized_w = input_w
        else:
            resized_w = int(math.ceil(input_h * ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        
        padded_image = np.zeros((self.input_shape[0], input_h, input_w), dtype=np.float32)
        padded_image[:, :, 0:resized_w] = resized_image
        return padded_image

    def __call__(self, images):
        if not images:
            return [], []
            
        batch_size = 6
        num_images = len(images)

        results = [''] * num_images
        confidences = [0.0] * num_images
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))

        for index in range(0, num_images, batch_size):
            input_h, input_w = self.input_shape[1], self.input_shape[2]
            max_wh_ratio = input_w / input_h
            
            current_batch_indices = indices[index:min(num_images, index + batch_size)]
            for i in current_batch_indices:
                h, w = images[i].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)
            
            norm_images = []
            for i in current_batch_indices:
                norm_image = self.resize(images[i], max_wh_ratio)
                norm_image = norm_image[np.newaxis, :]
                norm_images.append(norm_image)
            
            norm_images = np.concatenate(norm_images)

            self.input_tensor.copy_from_cpu(norm_images)
            self.predictor.run()
            outputs = self.output_tensor.copy_to_cpu()
            result, confidence = self.ctc_decoder(outputs)
            
            for i, idx in enumerate(current_batch_indices):
                results[idx] = result[i]
                confidences[idx] = confidence[i]
                
        return results, confidences
