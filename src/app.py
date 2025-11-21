import streamlit as st
st.set_page_config(layout="wide")

import cv2
import numpy as np
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
from detector import detect_faces, align_face


# -------------------------------
# LOAD MODELS & DATA
# -------------------------------

@st.cache(allow_output_mutation=True)
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
# PREDICT USING EMBEDDING
# -------------------------------

def predict_emb(emb, prob_threshold, sim_threshold):
    emb = np.array(emb).reshape(1, -1)

    pred = clf.predict(emb)[0]
    probs = clf.predict_proba(emb)[0]

    max_prob = float(probs[pred])
    name = le.inverse_transform([pred])[0]

    sims = {
        c: float(cosine_similarity(emb, centroids[c].reshape(1, -1))[0, 0])
        for c in centroids
    }
    best_class, best_sim = max(sims.items(), key=lambda x: x[1])

    if max_prob < prob_threshold or best_sim < sim_threshold:
        return "Unknown", max_prob

    return name, max_prob


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("Real-Time Multi-Face Recognition")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Settings")
    prob_threshold = st.slider("Probability threshold", 0.0, 1.0, 0.60, 0.01)
    sim_threshold = st.slider("Centroid similarity threshold", 0.0, 1.0, 0.55, 0.01)

    show_fps = st.checkbox("Show FPS", value=True)

    start_btn = st.button("Start Camera")
    stop_btn = st.button("Stop Camera")

with col1:
    frame_window = st.empty()


# -------------------------------
# CAMERA STATE
# -------------------------------

if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

# -------------------------------
# CAMERA LOOP  (Streamlit-safe)
# -------------------------------

if st.session_state.running:

    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while st.session_state.running:

        ret, frame = cap.read()
        if not ret:
            frame_window.write("Camera error")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = detect_faces(rgb)

        for det in detections:
            bbox = det["bbox"]
            lm = det["landmarks"]

            aligned = align_face(rgb, lm, output_size=(160, 160))

            try:
                rep = DeepFace.represent(
                    img_path=aligned,
                    model_name="Facenet",
                    enforce_detection=False
                )
                emb = rep[0]["embedding"]
            except:
                continue

            name, prob = predict_emb(emb, prob_threshold, sim_threshold)

            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)

            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                rgb,
                f"{name} {prob*100:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # FPS
        if show_fps:
            now = time.time()
            dt = now - prev_time
            fps = 1 / dt if dt > 0 else 0
            prev_time = now

            cv2.putText(
                rgb,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        frame_window.image(rgb)

        # IMPORTANT: allow Streamlit to process UI events
        time.sleep(0.01)

    cap.release()
