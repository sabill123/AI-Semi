import onnx
from onnx.utils import Extractor

def edit_onnx_graph(model, input_name, input_shape):
    output_to_shape = []

    for idx in range(3):
        box_layer = (
            f"/model.22/cv2.{idx}/cv2.{idx}.2/Conv_output_0",
            (1, 64, int(input_shape[2] / (8 * (1 << idx))), int(input_shape[3] / (8 * (1 << idx))),),
        )
        cls_layer = (
            f"/model.22/cv3.{idx}/cv3.{idx}.2/Conv_output_0",
            (1, 80, int(input_shape[2] / (8 * (1 << idx))), int(input_shape[3] / (8 * (1 << idx))),),
        )
        output_to_shape.append(box_layer)
        output_to_shape.append(cls_layer)

    output_to_shape = {
        tensor_name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
            for dimension_size in shape
        ]
        for tensor_name, shape in output_to_shape
    }

    extracted_model = Extractor(model).extract_model(
        input_names=list([input_name]),
        output_names=list(output_to_shape),
    )
    for value_info in extracted_model.graph.output:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(
            output_to_shape[value_info.name]
        )
    return extracted_model

# 1. ONNX Graph Load
model = onnx.load("yolov8n.onnx")
# 2. ONNX Graph Edit
edit_model = edit_onnx_graph(model, "images", (1,3,640,640))
# 3. Save edited ONNX
onnx.save(onnx.shape_inference.infer_shapes(edit_model), "yolov8n.onnx")