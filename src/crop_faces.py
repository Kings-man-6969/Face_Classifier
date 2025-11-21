import os
import cv2
from deepface import DeepFace # type: ignore

# SCRIPT LOCATION: src/crop_faces.py
# PHOTOS LOCATION: data/photos/

INPUT_ROOT = "../photos"                # the people folders
OUTPUT_ROOT = "../data"      # will be created automatically

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for person in os.listdir(INPUT_ROOT):
    person_in = os.path.join(INPUT_ROOT, person)

    # Skip files, keep only folders
    if not os.path.isdir(person_in):
        continue

    # Output folder per person
    person_out = os.path.join(OUTPUT_ROOT, person)
    os.makedirs(person_out, exist_ok=True)

    print(f"Processing: {person}")

    for fname in os.listdir(person_in):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        in_path = os.path.join(person_in, fname)
        out_path = os.path.join(person_out, fname)

        # Skip if previously processed
        if os.path.exists(out_path):
            continue

        try:
            # extract_faces returns a list of detected faces
            results = DeepFace.extract_faces(
                img_path=in_path,
                detector_backend='retinaface',
                enforce_detection=False
            )

            if len(results) == 0:
                print(f"[NO FACE] {in_path}")
                continue

            # Take the first detected face
            face = results[0]["face"]
            face = (face * 255).astype("uint8")

            # Standardize the crop size
            face = cv2.resize(face, (160, 160))

            # Save cropped face
            cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"[SKIP] {in_path}: {e}")

print("✔ Face cropping completed successfully!")
