import cv2
import os
import onnx

from tqdm import tqdm
from utils.preprocess import YOLOPreProcessor
from furiosa.optimizer import optimize_model
from furiosa.quantizer import (
    CalibrationMethod, Calibrator, quantize,
    ModelEditor, TensorType, get_pure_input_names,
)

# Load ONNX Model
model = onnx.load("yolov8n.onnx")
# ONNX Graph Optimization
model = optimize_model(
    model = model, opset_version=13, input_shapes={"images": [1, 3, 640, 640]}
)

# FuriosaAI SDK Calibrator: onnx model, calibration method
calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

data_dir = "../val2017"
calibration_dataset = os.listdir(data_dir)[:10]

# Preprocessing
preprocessor = YOLOPreProcessor()

for data_name in tqdm(calibration_dataset):
    data_path = os.path.join(data_dir, data_name)
    input_img = cv2.imread(data_path)
    input_, contexts = preprocessor(input_img, new_shape=(640, 640), tensor_type="float32")
    calibrator.collect_data([[input_]])

# Calculate Calibration Ranges
calibration_range = calibrator.compute_range()

## Optimize Quantize Operator uinsg Model Editor
editor = ModelEditor(model)
input_names = get_pure_input_names(model)

for input_name in input_names:
    editor.convert_input_type(input_name, TensorType.UINT8)

# Qauntization
quantized_model = quantize(model, calibration_range)

with open("yolov8n_opt_i8.onnx", "wb") as f:
    f.write(bytes(quantized_model))