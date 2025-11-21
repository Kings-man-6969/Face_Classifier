import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

EMB_FILE = "embeddings.npy"
LABEL_FILE = "labels.pkl"

print("Loading embeddings and labels...")
X = np.load(EMB_FILE)
with open(LABEL_FILE, "rb") as f:
    y = pickle.load(f)

# Encode string labels (names → numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training SVM classifier...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nClassifier accuracy: {acc * 100:.2f}%")

# Save classifier
with open("classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nSaved classifier.pkl and label_encoder.pkl")
print("Training complete!")
