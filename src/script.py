import onnxruntime as ort

session = ort.InferenceSession("../models/arcface.onnx")
print([i.name for i in session.get_inputs()])
