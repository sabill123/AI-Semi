import os, subprocess, asyncio
import cv2
import time
import threading

from utils.preprocess import YOLOPreProcessor
from utils.postprocess import ObjDetPostprocess
from furiosa.runtime import create_queue


async def submit_with(submitter, input_, contexts):
    for _ in range(1000):
        await submitter.submit(input_, context=(contexts))

async def recv_with(receiver, input_img, data_name):
    postprocessor = ObjDetPostprocess()
    for _ in range(1000):
        contexts, outputs = await receiver.recv()
        output_img = postprocessor(outputs, contexts, input_img)
        cv2.imwrite(os.path.join("result", data_name), output_img)

async def furiosa_runtime_queue(model_path, input_img, input_, contexts, data_name):

    async with create_queue(
        model=model_path, worker_num=8, device="warboy(2)*1"
    ) as (submitter, receiver):
        submit_task = asyncio.create_task(submit_with(submitter, input_, contexts))
        recv_task = asyncio.create_task(recv_with(receiver, input_img, data_name))
        await submit_task
        await recv_task


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
asyncio.run(furiosa_runtime_queue(model_path, input_img, input_, contexts, data_name))
t2 = time.time()
print(f"Inference Time for 1000 images: {t2-t1}s")