import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk
import time
from threading import Thread
from tkinter import colorchooser #Pick a color using a built-in pop-up
from tkinter import ttk #Used for a drop down good-looking  

# --- Tkinter GUI Setup ---
window = tk.Tk()    
window.title("Sign Language Detection")
window.geometry("400x200")
window.configure(bg="#1e1e1e")

font_main = ("Segoe UI", 18, "bold")
font_word = ("Segoe UI", 22)
font_button = ("Segoe UI", 12)
text_color = "#ffffff"

style = {
    "font": font_button,
    "width": 14,
    "height": 1,
    "bd": 0,
    "fg": "#ffffff",
    "activeforeground": "#ffffff",
}

label = tk.Label(window, text="Prediction:", font = ("Arial", 20), fg = text_color, bg = "#1e1e1e")
label.pack(pady = 20) #Used for vertical spacing

word_label = tk.Label(window, text = "Word:", font = ("Arial", 24), fg = text_color, bg = "#1e1e1e")
word_label.pack(pady = 20) #Used for vertical spacing

history_text = tk.Text(window, height = 6, width = 40, font = ("Segoe UI", 12), fg = "#00ffcc", bg = "#2c2c2c")
history_text.pack(pady = 10) #Used for vertical spacing

button_frame = tk.Frame(window, bg = "#1e1e1e")
button_frame.pack(pady = 10) #Used for vertical spacing

frame = tk.Frame(window , bg = "#1e1e1e")
frame.pack(pady = 10) #Used for vertical spacing

font_options = [ "Arial" , "Comic Sans MS" , "Times New Roman" , "Courier New" , "Segoe UI" , "Helvetica"]
selected_font = tk.StringVar(value = "Segoe UI")

def update_font(*args):
    word_label.config(font=(selected_font.get(), 22))
    label.config(font=(selected_font.get(), 18, "bold"))

font_dropdown = ttk.Combobox(frame , textvariable = selected_font, values = font_options, width = 20, state = "readonly") # Creates a dropdown menu which links to the variable storing the selected font with font list , width and user will not type and can ony choose(readonly)
font_dropdown.bind("<<ComboboxSelected>>", update_font) #This tells the dropdown to use the new font
font_dropdown.grid(row = 0, column = 0, padx = 10) #Places the dropdown using a grid layout and adds horizontal spacing

# Color picker
def pick_color():
    color = colorchooser.askcolor(title="Choose text color") #askcolor() gives a color palette. It returns a tuple like (R,G,B) with the hexcode
    if color[1]: #If hex code choosen then it updates the text color of the word and prediction levels
        word_label.config(fg = color[1])
        label.config(fg = color[1])

color_button = tk.Button(frame, text = "Pick Text Color", command = pick_color, **style, bg = "#9b59b6", activebackground = "#7f4ca0") #Adds a button labeled "Pick Text Color" , when clicked it runs pick_color() to open the color chooser. It is styled with a purple background and placed to the right of the dropdown.
color_button.grid(row = 0, column = 1, padx = 10)

# --- Load your pre-trained KNN model ---
try:
    model = joblib.load("sign_knn_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) #Initialize hand detection with CI

buffer_letter = []
word = ""
save_words = []

# --- Button commands ---
def remove_last_letter():
    global word
    if word:
        word = word[:-1]
        word_label.config(text=f"Word: {word}")

def clear_word():
    global word
    word = ""
    word_label.config(text="Word:")

def wordings():
    global word , save_words
    if word.strip():
        save_words.append(word)
        with open("saved_words.txt" , "a") as f:
            f.write(word + "\n")
        print("Saved words:" , word)
        word_label.config(text="Word:")
        word = ""

def history():
    global save_words
    if save_words:
        filename = f"exported_words_{int(time.time())}.txt"
        with open(filename , "w") as f:
            f.write("\n".join(save_words)) 
        print(f"Exported history to: {filename}")
        history_text.delete(1.0 , tk.END)
        history_text.insert(tk.END , "Exported Words:\n" + "\n".join(save_words))

button_remove = tk.Button(button_frame, text="Remove Last", command = remove_last_letter, bg = "#ff8800", activebackground = "#cc7000", **style)
button_remove.grid(row=0, column=0, padx=10)

button_clear = tk.Button(button_frame, text="Clear Word", command = clear_word, bg = "#e74c3c", activebackground = "#c0392b", **style)
button_clear.grid(row=0, column=1, padx=10)

button_save = tk.Button(button_frame, text="Save Word", command = wordings, bg = "#27ae60", activebackground = "#1e8449", **style)
button_save.grid(row=1, column=0, padx=5, pady=10)

button_export = tk.Button(button_frame, text="Export History", command = history, bg = "#2980b9", activebackground = "#1c5980", **style)
button_export.grid(row=1, column=1, padx=5, pady=10)

# --- Video capture and prediction thread ---
def video_loop():
    global word
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam could not be opened")
        return

    while True:
        time.sleep(0.015)  # small delay to reduce CPU load
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)  # mirror image horizontally
        frame = cv2.resize(frame, (480, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        predicted_letter = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks: #Loop from each hand that is 21 landmarks of each hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #frame- video image, hand_landmarks- data for that hand, and mp_hands.HAND_CONNECTIONS- which landmarks are connected by lines.

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                landmarks_np = np.array(landmarks).reshape(1, -1)
                predicted_letter = model.predict(landmarks_np)[0]

            # Buffer smoothing logic
            buffer_letter.append(predicted_letter)
            if len(buffer_letter) > 15:
                buffer_letter.pop(0)

            most_common_letter = max(set(buffer_letter), key=buffer_letter.count)
            frequency = buffer_letter.count(most_common_letter)

            # Add to word only if prediction is stable and new
            if frequency > 12 and (len(word) == 0 or most_common_letter != word[-1]):
                word += most_common_letter
                buffer_letter.clear()
                word_label.config(text=f"Word: {word}")

            label.config(text=f"Prediction: {predicted_letter}")

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start video loop in a separate thread to avoid freezing GUI
Thread(target=video_loop, daemon=True).start()

window.mainloop()

