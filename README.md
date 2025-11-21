# Face Classifier – Real-Time Face Recognition App  
A fast, lightweight, real-time face recognition system built using **DeepFace (Facenet model)**, **OpenCV**, **scikit-learn (SVM classifier)**, and a **Streamlit UI**.

This project performs real-time face recognition for a fixed set of known people.  
It extracts facial embeddings using **Facenet (128-dimensional)** and trains an SVM classifier to identify individuals with high accuracy.

### ✔ Real-time recognition  
### ✔ 5 known identities  
### ✔ Highly accurate SVM classifier  
### ✔ Centroid similarity + probability thresholding (to detect unknown faces)  
### ✔ GPU-accelerated TensorFlow 2.10 support  
### ✔ Streamlit web interface  
### ✔ Clean modular codebase  

---

## 📌 Features

### 🔹 **1. Embedding Extraction**
Uses **DeepFace → Facenet** to generate 128-D embeddings from cropped face images.  
Embeddings are saved as:

- `embeddings.npy`  
- `labels.pkl`

### 🔹 **2. Classifier Training**
An **SVM classifier** is trained on the embeddings and saved as:

- `classifier.pkl`  
- `label_encoder.pkl`

Accuracy typically ranges between **88–95%** depending on dataset quality.

### 🔹 **3. Real-Time Face Recognition UI**
The Streamlit interface:

- Starts webcam feed  
- Detects faces in the frame  
- Generates embeddings live  
- Predicts identity using SVM  
- Performs centroid similarity check  
- Displays name + confidence  
- Flags unknown faces (below threshold)

### 🔹 **4. Thresholding for Reliability**
Two checks determine if a face is “known”:

- SVM probability  
- Centroid cosine similarity  

This improves reliability significantly.

---

## 📁 Project Structure

```
Face_Classifier/
│
├── src/
│   ├── app.py                # Streamlit real-time UI
│   ├── embed.py              # Embedding extraction script
│   ├── train_classifier.py   # SVM training script
│   ├── ...
│
├── models/                   # (Ignored in git) DeepFace/Facenet weights if needed
├── data/                     # (Ignored) Cropped training images
│
├── embeddings.npy            # (Ignored) Generated embeddings
├── labels.pkl                # (Ignored)
├── classifier.pkl            # (Ignored)
├── label_encoder.pkl         # (Ignored)
│
└── README.md
```

---

## 🚀 Getting Started

### **1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Face_Classifier.git
cd Face_Classifier
```

### **2. Create and activate a conda environment**
```bash
conda create -n tf2 python=3.9
conda activate tf2
```

### **3. Install GPU-compatible packages**
(Optional but recommended)
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### **4. Install required Python packages**
```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow==2.10
pip install deepface==0.0.95
pip install streamlit==1.23
pip install scikit-learn==1.2.2
pip install opencv-python
pip install numpy==1.23.5
```

---

## 🔧 How to Use

### **Step 1 — Prepare dataset**
Place cropped face images into:

```
data/
   Person1/
       img1.jpg
       img2.jpg
   Person2/
       img1.jpg
       img2.jpg
   ...
```

### **Step 2 — Extract embeddings**
Inside `src/`:

```bash
python embed.py
```

### **Step 3 — Train the classifier**
```bash
python train_classifier.py
```

### **Step 4 — Run the real-time Streamlit app**
```bash
streamlit run app.py
```

---

## 🧠 How It Works (Technical Overview)

- **DeepFace (Facenet)**  
  Generates a 128-dimensional face embedding vector.

- **SVM Classifier**  
  Trained on embeddings → predicts identity based on embedding similarity.

- **Centroid Similarity**  
  Mean embedding is computed for each class.  
  Cosine similarity is used for “unknown face detection”.

- **Thresholding**  
  Both probability + similarity must exceed threshold → improves accuracy.

---

## 🛡 Git Security
All personal training data is ignored using `.gitignore`.

---

## 🧩 Future Enhancements

- Multi-face detection  
- Automated face enrollment via webcam  
- Attendance system (CSV/Excel)  
- Streamlit UI redesign  
- Faster embedding caching  
- SQLite/PostgreSQL logging  
- REST API for external apps  

---

## 📜 License
Add your preferred license (MIT recommended).
