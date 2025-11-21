# sanity_checks.py
import numpy as np
import pickle

EMB_FILE = "embeddings.npy"
LABEL_FILE = "labels.pkl"
ENCODER_FILE = "label_encoder.pkl"

X = np.load(EMB_FILE)   # (N, D)
with open(LABEL_FILE, "rb") as f:
    y = pickle.load(f)
with open(ENCODER_FILE, "rb") as f:
    from sklearn.preprocessing import LabelEncoder
    label_encoder = pickle.load(f)

print("Total embeddings:", X.shape)
print("Total labels:", len(y))
print("Label encoder classes:", label_encoder.classes_)
# Check distribution per class
from collections import Counter
print("Samples per class:", Counter(y))
