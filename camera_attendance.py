import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

# Folder containing known faces
path = 'images'
images = []
classNames = []
myList = os.listdir(path)
print(f"✅ Loaded {len(myList)} known faces.")

# Load and encode known faces
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0])
    return encodeList

def mark_attendance(name):
    filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Name,Time\n")

    with open(filename, 'r+') as f:
        lines = f.readlines()
        nameList = [line.split(',')[0] for line in lines]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{name},{dtString}\n')
            print(f"🟢 Attendance marked for {name.upper()} at {dtString}")

# Encode faces
encodeListKnown = findEncodings(images)
print("✅ Encoding complete.")

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found.")
    exit()

print("📷 Camera running — Press 'q' to quit manually.")

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to grab frame, retrying...")
        time.sleep(1)
        continue

    # Resize for performance
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect and encode faces
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # For each face in the frame
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # Green box for recognized face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                mark_attendance(name)
            else:
                # Red box for unknown face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                              (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display
    cv2.imshow('Hospital Attendance System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Camera closed manually.")
        break

# Release
cap.release()
cv2.destroyAllWindows()
