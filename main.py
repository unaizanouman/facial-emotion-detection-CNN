import numpy as np
import cv2
import tensorflow as tf
import os

#  Load the trained model
model = tf.keras.models.load_model('emotion_detection_model_gsn.h5')

#  Emotion label dictionary (maps model's output to emotion names)
emotion_dict = {
    0: "Angry", 
    1: "Disgusted", 
    2: "Fearful", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprised"
}

#  Load Haar Cascade for face detection
haar_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(haar_path):
    print(f" Haar Cascade file not found at: {haar_path}")
    exit()

facecasc = cv2.CascadeClassifier(haar_path)

# ✅ Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to capture frame from webcam.")
        break

    #  Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #  Detect faces in the image
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around faq qqqqqce
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # Crop and resize the face region
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = cropped_img.reshape(1, 48, 48, 1)
        cropped_img = cropped_img.astype('float32') / 255.0  # Normalize

        # Predict emotion
        prediction = model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))

        # Show the emotion label
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the resulting frame
    cv2.imshow('Facial Emotion Detection', cv2.resize(frame, (800, 480)))

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resourcespython -c "import tensorflow as tf; print(tf.__version__)"
cap.release()
cv2.destroyAllWindows()
