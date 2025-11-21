# centroid_check.py
import numpy as np, pickle
from deepface import DeepFace # pyright: ignore[reportMissingImports]
from sklearn.metrics.pairwise import cosine_similarity

X = np.load("embeddings.npy")
with open("labels.pkl","rb") as f:
    y = pickle.load(f)
with open("label_encoder.pkl","rb") as f:
    le = pickle.load(f)

# build centroids
import numpy as np
classes = sorted(set(y))
centroids = {}
for c in classes:
    c_embs = X[[i for i,lab in enumerate(y) if lab==c]]
    centroids[c] = c_embs.mean(axis=0)  # simple mean centroid

def predict_by_centroid(test_img):
    rep = DeepFace.represent(img_path=test_img, model_name="Facenet", enforce_detection=False)
    emb = np.array(rep[0]["embedding"]).reshape(1, -1)
    sims = {c: float(cosine_similarity(emb, centroid.reshape(1,-1))[0,0]) for c, centroid in centroids.items()}
    sorted_s = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return sorted_s

if __name__ == "__main__":
    test_img = "../data/Swati Mishra/Xiaomi9660.jpg"
    res = predict_by_centroid(test_img)
    print("Centroid similarities (top 5):")
    for name, sim in res[:5]:
        print(name, f"{sim:.4f}")
