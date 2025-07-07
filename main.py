import cv2
import numpy as np
import face_recognition  # ✅ correct import (no hyphens)

# Load and convert training image (Elon Musk)
imgElon = face_recognition.load_image_file(
    r'C:\Users\shine\Face_recognition\Face-Recognition\images\Elon_Musk_train.png')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# Load and convert test image
imgTest = face_recognition.load_image_file(
    r'C:\Users\shine\Face_recognition\Face-Recognition\images\Elon_Musk_test.webp')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detect face and encode in training image
faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

# Detect face and encode in test image
facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (255, 0, 255), 2)

# Compare faces and calculate distance
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

print("Match Result:", results)
print("Face Distance:", faceDis)

# Display result on test image
cv2.putText(imgTest, f'{results[0]} {round(faceDis[0], 2)}',
            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Show the images
cv2.imshow('Elon Musk (Train)', imgElon)
cv2.imshow('Elon Musk (Test)', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
