import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# 🔹 Path to your known images
path = r'C:\Users\shine\Face_recognition\Face-Recognition\images'
images = []
classNames = []

# 🔹 Load images and their class names
mylist = os.listdir(path)
print("Found files:", mylist)

for cl in mylist:
    curimage = cv2.imread(f"{path}/{cl}")
    if curimage is not None:  # Handle unreadable files
        images.append(curimage)
        classNames.append(os.path.splitext(cl)[0])

print("Class names:", classNames)

# 🔹 Function to find face encodings for known images
def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodelist.append(encodings[0])  # Only take first face
    return encodelist

# 🔹 Function to mark attendance in CSV file
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")

# 🔹 Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete ✅')

# 🔹 Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break
    img = cv2.flip(img, 1)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for speed
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("Face distances:", faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print("Matched:", name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # Scale back to original size

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance
            markAttendance(name)

    cv2.imshow('Webcam', img)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔚 Clean up
cap.release()
cv2.destroyAllWindows()
