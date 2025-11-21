# topk_predict.py
import pickle, numpy as np
from deepface import DeepFace # type: ignore

CLASSIFIER_FILE = "classifier.pkl"
ENCODER_FILE = "label_encoder.pkl"

with open(CLASSIFIER_FILE, "rb") as f:
    clf = pickle.load(f)
with open(ENCODER_FILE, "rb") as f:
    le = pickle.load(f)

def predict_topk(img_path, k=3):
    rep = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
    emb = np.array(rep[0]["embedding"]).reshape(1, -1)
    probs = clf.predict_proba(emb)[0]
    idxs = probs.argsort()[::-1][:k]
    return [(le.inverse_transform([i])[0], float(probs[i])) for i in idxs]

if __name__ == "__main__":
    test_img = "../data/Swati Mishra/Xiaomi9660.jpg"  # change if needed
    print("Top predictions (name, prob):")
    for name, p in predict_topk(test_img):
        print(name, f"{p*100:.2f}%")
