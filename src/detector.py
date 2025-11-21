import cv2
import numpy as np

MODEL = "../models/res10_300x300_ssd_iter_140000.caffemodel"
PROTO = "../models/deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

def detect_faces(img_rgb):
    h, w = img_rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img_rgb, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "landmarks": {}  # we will use bbox-only alignment fallback
        })

    return results


def align_face(img_rgb, landmarks, output_size=(160,160)):
    """
    Align face using simple center crop using bbox since no landmarks available.
    """
    # fallback: just resize image to Facenet size
    return cv2.resize(img_rgb, output_size)
