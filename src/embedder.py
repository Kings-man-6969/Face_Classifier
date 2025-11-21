import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX session
session = ort.InferenceSession("../models/recognition_resnet27.onnx", providers=["CPUExecutionProvider"])

def preprocess(img):
    # Expecting RGB input
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32)

    # Normalization: (img - 127.5) / 128 — common for face embeddings
    img = (img - 127.5) / 128.0

    # Change HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

def get_embedding(img):
    """
    img: RGB frame (np.array)
    returns: L2-normalized embedding (1D vector)
    """
    inp = preprocess(img)

    output = session.run(None, {"input": inp})[0][0]

    # Normalize
    norm = np.linalg.norm(output)
    if norm == 0:
        return output
    return output / norm
