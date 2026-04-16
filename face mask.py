import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and face detector
model = load_model('mask_detector.model')
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capture
source = cv2.VideoCapture(0)
while True:
    ret, img = source.read()
    faces = face_clsfr.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Pre-process, predict, and draw bounding boxes
        face_img = cv2.resize(img[y:y+h, x:x+w], (224, 224)) / 255.0
        result = model.predict(np.reshape(face_img, (1, 224, 224, 3)))
        label = np.argmax(result, axis=1)[0]
        
        color = (0, 255, 0) if label == 0 else (0, 0, 255) # Green: Mask, Red: No Mask
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, 'MASK' if label == 0 else 'NO MASK', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    cv2.imshow('Face Mask Detector', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

source.release()
cv2.destroyAllWindows()
