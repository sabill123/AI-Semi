from furiosa.models.vision import SSDMobileNet
from furiosa.runtime.sync import create_runner
image = ["./assets/images/000000334555.jpg"]
mobilenet = SSDMobileNet()
with create_runner(mobilenet.model_source()) as runner:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = runner.run(inputs)
    result = mobilenet.postprocess(outputs, contexts)
    print(result)
