import cv2  # for webcams and images
import mediapipe as mp  # detect body parts
import joblib
import numpy as np
import time

# Load the trained KNN model
model = joblib.load("sign_knn_model.pkl")

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up hand detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

time.sleep(2)  # Allow camera to warm up
print("Press q to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)  # Mirror image for natural interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    results = hands.process(rgb_frame)  # Process the frame with MediaPipe

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract all 63 landmark coordinates: x, y, z
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks_np = np.array(landmarks).reshape(1, -1)  # Reshape for model input

            predicted_label = model.predict(landmarks_np)[0]  # Predict label from model

            # Display prediction
            cv2.putText(
                frame,
                f'Prediction: {predicted_label}',
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Show the frame
    cv2.imshow("Hand Tracker", frame)

    # Exit loop on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
