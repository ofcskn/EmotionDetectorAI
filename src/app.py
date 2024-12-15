# File: app.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('../data/model/emotion_detector_model_1.h5')

# Label map for the emotions
label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale and resize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))

    # Normalize and expand dimensions for prediction
    resized_frame = np.expand_dims(resized_frame, axis=-1)  # Add color channel dimension
    resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    resized_frame = resized_frame / 255.0  # Normalize the frame

    # Predict emotion
    prediction = model.predict(resized_frame)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability

    # Get the emotion label
    emotion = label_map[predicted_class]

    # Display the predicted emotion on the frame
    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with emotion label
    cv2.imshow('Emotion Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
