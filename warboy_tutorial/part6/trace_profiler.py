# inference.py
import numpy as np
from furiosa.runtime.sync import create_runner
from furiosa.runtime.profiler import profile

with open("tracing.json", "w") as output:
    with profile(file = output) as profiler:
        with create_runner("yolov8n.enf") as runner:
            input_shape = runner.model.input(0).shape
            dummy_input = np.uint8(np.random.randn(*input_shape))
            with profiler.record("trace") as record:
                runner.run([dummy_input])
