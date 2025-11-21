import cv2
from detector import detect_faces

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detect_faces(rgb)

    print("Detections:", detections)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
