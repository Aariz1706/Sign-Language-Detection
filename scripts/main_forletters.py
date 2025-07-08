import cv2
import mediapipe as mp
import csv
import os

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

# Set label for gesture you're collecting (change it each time)
label = "Z"  # <- CHANGE THIS TO 'B', 'C', etc. as needed

# Create CSV if it doesn't exist
csv_file = "hand_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        header = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        csv_writer.writerow(header)

# Loop for real-time video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Save landmarks when 's' is pressed
            if cv2.waitKey(10) & 0xFF == ord('s'):
                with open(csv_file, mode="a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([label] + landmarks)
                print(f"Saved {label} gesture")

    cv2.imshow("Hand Tracker", frame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
