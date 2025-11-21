# compare_similarities.py
import numpy as np, pickle
from deepface import DeepFace # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import os

EMB_FILE = "embeddings.npy"
LABEL_FILE = "labels.pkl"

X = np.load(EMB_FILE)   # shape (N, D)
with open(LABEL_FILE, "rb") as f:
    y = pickle.load(f)

def find_closest_imgs(test_img, topk=5):
    rep = DeepFace.represent(img_path=test_img, model_name="Facenet", enforce_detection=False)
    emb = np.array(rep[0]["embedding"]).reshape(1, -1)
    sims = cosine_similarity(emb, X)[0]
    idxs = sims.argsort()[::-1][:topk]
    return [(idx, y[idx], float(sims[idx])) for idx in idxs]

if __name__ == "__main__":
    test_img = "../data/Swati Mishra/Xiaomi9660.jpg"
    close = find_closest_imgs(test_img, topk=8)
    print("Closest stored images (index, label, cosine):")
    for idx, label, sim in close:
        print(idx, label, f"{sim:.4f}")
