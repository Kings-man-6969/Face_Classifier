import numpy as np
import pickle
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# load classifier and encoders
with open("classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# load stored embeddings + labels (for centroid)
X = np.load("embeddings.npy")
with open("labels.pkl", "rb") as f:
    y = pickle.load(f)

# build centroids
classes = sorted(set(y))
centroids = {}
for c in classes:
    c_embs = X[[i for i, lab in enumerate(y) if lab == c]]
    centroids[c] = c_embs.mean(axis=0)

def predict_face(img_path, prob_threshold=0.60, sim_threshold=0.55):

    # get embedding
    rep = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
    emb = np.array(rep[0]["embedding"]).reshape(1, -1)

    # SVM prediction
    pred = clf.predict(emb)[0]
    probs = clf.predict_proba(emb)[0]
    max_prob = float(probs[pred])
    name = le.inverse_transform([pred])[0]

    # centroid similarity
    sims = {c: float(cosine_similarity(emb, centroids[c].reshape(1, -1))[0, 0]) for c in centroids}
    best_class, best_sim = max(sims.items(), key=lambda x: x[1])

    # ---------- HERE IS YOUR THRESHOLD LOGIC ----------
    if max_prob < prob_threshold or best_sim < sim_threshold:
        return "Unknown", max_prob, best_class, best_sim
    # -------------------------------------------------

    return name, max_prob, best_class, best_sim


# test
if __name__ == "__main__":
    test_img = "../data/Swati Mishra/Xiaomi9660.jpg"
    name, prob, centroid_class, sim = predict_face(test_img)
    print("Final Result:", name)
    print(f"Prob: {prob:.2f}, Best centroid: {centroid_class}, Similarity: {sim:.3f}")
