import pickle
import numpy as np
from deepface import DeepFace # type: ignore

# files
CLASSIFIER_FILE = "classifier.pkl"
ENCODER_FILE = "label_encoder.pkl"

# load classifier
with open(CLASSIFIER_FILE, "rb") as f:
    clf = pickle.load(f)

# load label encoder
with open(ENCODER_FILE, "rb") as f:
    label_encoder = pickle.load(f)

def predict_face(img_path):
    # get embedding using DeepFace
    rep = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet",
        enforce_detection=False
    )
    embed = rep[0]["embedding"]

    # convert to array shape (1, n)
    embed = np.array(embed).reshape(1, -1)

    # predict
    pred = clf.predict(embed)[0]
    name = label_encoder.inverse_transform([pred])[0]

    # optional: probability
    probs = clf.predict_proba(embed)[0]
    confidence = max(probs)

    return name, confidence

# test
test_img = r'C:\Users\gungu\Projects\Face_Classifier\photos\Pita Ji\20220420_111436.jpg'  # change path
name, conf = predict_face(test_img)

print("\nPrediction:")
print("Name:", name)
print(f"Confidence: {conf*100:.2f}%")
