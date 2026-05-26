import math
import os
import json
import warnings
from argparse import ArgumentParser
from importlib import resources
from pathlib import Path
from warnings import filterwarnings

import cv2
import numpy
from PIL import Image
from pyclipper import ET_CLOSEDPOLYGON, JT_ROUND, PyclipperOffset
from shapely.geometry import Polygon
import urllib.request
import zipfile

class Classification:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
       
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider',
                                                       'CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()[0]
        self.threshold = 0.98
        self.labels = ['0', '180']

    @staticmethod
    def resize(image):
        input_c, input_h, input_w = 3, 48, 192
        h, w = image.shape[:2]
        ratio = w / float(h)
        resized_w = input_w if math.ceil(input_h * ratio) > input_w else int(math.ceil(input_h * ratio))
        resized_image = cv2.resize(image, (resized_w, input_h)).transpose((2, 0, 1)).astype('float32')
        resized_image = resized_image / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        padded = numpy.zeros((input_c, input_h, input_w), dtype=numpy.float32)
        padded[:, :, 0:resized_w] = resized_image
        return padded

    def __call__(self, images):
        num_images = len(images)
        results = [['', 0.0]] * num_images
        indices = numpy.argsort(numpy.array([x.shape[1] / x.shape[0] for x in images]))
        batch_size = 6
        for i in range(0, num_images, batch_size):
            norm_images = []
            for j in range(i, min(num_images, i + batch_size)):
                norm_images.append(self.resize(images[indices[j]])[numpy.newaxis, :])
            norm_images = numpy.concatenate(norm_images)
            outputs = self.session.run(None, {self.inputs.name: norm_images})[0]
            outputs = [(self.labels[idx], outputs[k, idx])
                       for k, idx in enumerate(outputs.argmax(axis=1))]
            for j, (label, score) in enumerate(outputs):
                results[indices[i + j]] = [label, score]
                if '180' in label and score > self.threshold:
                    images[indices[i + j]] = cv2.rotate(images[indices[i + j]], 1)
        return images, results