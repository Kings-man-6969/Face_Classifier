# confusion_report.py
import numpy as np, pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X = np.load("embeddings.npy")
with open("labels.pkl","rb") as f:
    y = pickle.load(f)

# encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)

# quick train/test split and evaluation (same training recipe)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
clf = SVC(kernel='linear', probability=True).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix shape:", cm.shape)
