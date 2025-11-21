import os
import numpy as np
from deepface import DeepFace
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import Normalizer
import pickle
import cv2

# Path to your cropped dataset
DATASET_DIR = "../data"   # modify if needed

# Where to save output
EMB_FILE = "embeddings.npy"
LABEL_FILE = "labels.pkl"

# Load Facenet model
print("Loading Facenet model...")
model = DeepFace.build_model("Facenet")
print("Model loaded.")

# L2 normalizer for embeddings
normalizer = Normalizer(norm='l2')

embeddings = []
labels = []

def preprocess_img(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


# Walk through dataset
print("Extracting embeddings...")
for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            rep = DeepFace.represent(
                img_path = img_path,
                model_name = "Facenet",
                enforce_detection = False
            )
            embed = rep[0]["embedding"]
            embeddings.append(embed)
            labels.append(person)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

embeddings = np.array(embeddings)

# Save embeddings
np.save(EMB_FILE, embeddings)

# Save labels
with open(LABEL_FILE, "wb") as f:
    pickle.dump(labels, f)

print("\nDone!")
print(f"Total embeddings extracted: {len(embeddings)}")
print("Saved to:", EMB_FILE, "and", LABEL_FILE)
