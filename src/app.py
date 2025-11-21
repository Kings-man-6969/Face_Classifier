import streamlit as st
import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace

# -------------------------------
# LOAD MODELS & DATA
# -------------------------------

@st.cache_resource
def load_models():
    with open("classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    X = np.load("embeddings.npy")
    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)

    # Build centroids
    centroids = {}
    classes = sorted(set(labels))

    for c in classes:
        c_embs = X[[i for i, lab in enumerate(labels) if lab == c]]
        centroids[c] = c_embs.mean(axis=0)

    return clf, le, centroids

clf, le, centroids = load_models()


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------

def predict_face(frame, prob_threshold=0.60, sim_threshold=0.55):

    rep = DeepFace.represent(
        img_path = frame,
        model_name = "Facenet",
        enforce_detection = False
    )

    emb = np.array(rep[0]["embedding"]).reshape(1, -1)

    # Classifier prediction
    pred = clf.predict(emb)[0]
    probs = clf.predict_proba(emb)[0]
    max_prob = float(probs[pred])
    name = le.inverse_transform([pred])[0]

    # Centroid similarity
    sims = {
        c: float(cosine_similarity(emb, centroids[c].reshape(1, -1))[0, 0])
        for c in centroids
    }
    best_class, best_sim = max(sims.items(), key=lambda x: x[1])

    # Threshold logic
    if max_prob < prob_threshold or best_sim < sim_threshold:
        return "Unknown", max_prob, best_class, best_sim

    return name, max_prob, best_class, best_sim


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("Real-Time Face Recognition (ONNX)")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        continue

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prediction
    name, prob, centroid_class, sim = predict_face(rgb)

    # Draw result text
    text = f"{name} ({prob*100:.1f}%)"
    color = (0,255,0) if name != "Unknown" else (255,0,0)
    cv2.putText(rgb, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    FRAME_WINDOW.image(rgb)

camera.release()
