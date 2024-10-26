import os, subprocess, asyncio
import cv2
import time
import threading

from utils.preprocess import YOLOPreProcessor
from utils.postprocess import ObjDetPostprocess
from furiosa.runtime.sync import create_runner

def furiosa_runtime_sync(model_path, input_img, input_, contexts, data_name):
    postprocessor = ObjDetPostprocess()
    with create_runner(model_path, device = "warboy(2)*1") as runner:
        for _ in range(1000):
            preds = runner.run([input_]) # FuriosaAI Runtime
            output_img = postprocessor(preds, contexts, input_img)
            cv2.imwrite(os.path.join("result", data_name), output_img)


model_path = "yolov8n_opt_i8.onnx"
data_dir = "../val2017"
data_name = os.listdir(data_dir)[0]

if os.path.exists("result"):
    subprocess.run(["rm", "-rf", "result"])
os.makedirs("result")

preprocessor = YOLOPreProcessor()
input_img = cv2.imread(os.path.join(data_dir, data_name))
input_, contexts = preprocessor(input_img, new_shape=(640, 640), tensor_type="uint8")


t1 = time.time()
furiosa_runtime_sync(model_path, input_img, input_, contexts, data_name)
t2 = time.time()
print(f"Inference Time for 1000 images: {t2-t1}s")