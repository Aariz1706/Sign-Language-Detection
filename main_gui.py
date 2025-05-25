import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk # Used to create the GUI window
from threading import Thread # To run video capture without freezing the GUI

window = tk.Tk()
window.title("Sign Language Recognition")
window.geometry("400x200")

label = tk.Label(window , text = "Prediction: " , font = ("Arial" , 20)) #Label widget or button to show prediction
label.pack(pady = 20) #Adds label with paddin g tot he window
word_label = tk.Label(window , text = "Word:" , font = ("Arial" , 24))
word_label.pack(pady = 20)

try:
    model = joblib.load("sign_knn_model.pkl") #Load the pre-trained KNN model
except Exception as e:
    print(f"Error loading model:{e}")
    exit()

mp_hands = mp.solutions.hands #Access mediapipe hands soln
mp_drawing = mp.solutions.drawing_utils #Utilities to draw landmarks
hands = mp_hands.Hands(min_detection_confidence = 0.7 , min_tracking_confidence = 0.5) #Initialize hand detection with CI

buffer_letter = []
word = ""

def video_loop():
    global word
    cap = cv2.VideoCapture(0)
    while True:
        success , frame = cap.read() #Capture a single frame
        if not success:
            print("Failed to read frame from webcam")
            continue

        frame = cv2.flip(frame , 1) #Flip horizontaaly for mirroring
        frame = cv2.resize(frame , (480,480))
        rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB) #Converting to rgb in mediapipe
        results = hands.process(rgb_frame)
        predicted_letter = ""
        if results.multi_hand_landmarks: #If mp detected any hands
            for hand_landmarks in results.multi_hand_landmarks: #Loop from each hand that is 21 landmarks of each hand
                mp_drawing.draw_landmarks(frame , hand_landmarks , mp_hands.HAND_CONNECTIONS) #frame- video image, hand_landmarks- data for that hand, and mp_hands.HAND_CONNECTIONS- which landmarks are connected by lines.
                landmarks=[]
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x , lm.y , lm.z])
                landmarks_np = np.array(landmarks).reshape(1 , -1)
                predicted_letter = model.predict(landmarks_np)[0]

            buffer_letter.append(predicted_letter)
            if len(buffer_letter) > 15:
                buffer_letter.pop(0)
            most_common_letter = max(set(buffer_letter) , key = buffer_letter.count)
            frequency = buffer_letter.count(most_common_letter)
            if frequency > 12 and (len(word) == 0 or most_common_letter != word[-1]):
                word +=most_common_letter
                buffer_letter.clear()
                word_label.config(text = f"Word: {word}")
            label.config(text = f"Prediction: {predicted_letter}")
        cv2.imshow("Hand Tracker" , frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
Thread(target = video_loop ,  daemon = True).start()
window.mainloop()
