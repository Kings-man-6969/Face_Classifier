# Working Multi-Face Recognition Branch

This branch contains the FIRST STABLE version of the real-time multi-face 
recognition system built using:

- DeepFace (Facenet 128D embeddings)
- RetinaFace / MTCNN detection
- SVM classifier with centroid-based verification
- Streamlit real-time UI

---

## ✔ What Works
- Multi-face detection in real-time.
- High recognition accuracy for the 5 trained identities.
- Live bounding boxes, labels, and probability display.
- Stable FPS (~2–5 on CPU, higher with GPU).

---

## ⚠ Limitations
- DeepFace.represent() per-frame slows FPS.
- Face alignment may be imperfect for certain angles.
- Model is sensitive to lighting differences.
- No ONNX acceleration yet.

---

## 🌱 Future Improvements
Suggested upgrades for the next branch:

1. **Switch to ArcFace (512D ONNX)**  
   Faster, more accurate, GPU-friendly.

2. **Add object tracking (SORT/ByteTrack)**  
   Runs detection every 10 frames → FPS 2×–3× boost.

3. **Introduce threading for camera + inference**  
   Smooth UI and higher throughput.

4. **Session-controlled model loading**  
   Avoid reloading models each frame.

---

## 🧩 Purpose of This Branch
This branch serves as a *checkpoint* for the stable working version before
adding new optimizations such as ONNX, tracking, GPU improvements, or redesigning
the pipeline.

This ensures we can always fall back to a known good state.
