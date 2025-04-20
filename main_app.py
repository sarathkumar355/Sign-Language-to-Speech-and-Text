import tkinter as tk
from tkinter import Label
import cv2
import pyttsx3
from PIL import Image, ImageTk
import threading
import numpy as np
import mediapipe as mp
import pickle

# Load the trained gesture model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Gesture → Sentence mapping
gesture_to_sentence = {
    "A": "I need help.",
    "B": "Call the doctor.",
    "C": "I'm feeling hungry.",
    "D": "Thank you.",
    "E": "I love you.",
    "F": "I am thirsty.",
    "G": "I am tired.",
    "H": "I am happy.",
    "I": "I am sad.",
    "J": "I am angry.",
    "K": "I am scared.",
    "L": "I am bored.",
    "M": "I am confused.",
    "N": "I am excited.",
    "O": "I am surprised.",
    "P": "I am worried.",
    "Q": "I am lonely.",
    "R": "I am in pain.",
    "S": "I am sick.",
    "T": "I am cold.",
    "U": "I am hot.",
    "V": "I am sleepy.",    
    "W": "I am busy."
    }

# Initialize TTS
engine = pyttsx3.init()

def speak(sentence):
    engine.say(sentence)
    engine.runAndWait()

class GestureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Gesture to Speech")
        self.video_label = Label(window)
        self.video_label.pack()

        self.sentence_label = Label(window, text="", font=("Arial", 20))
        self.sentence_label.pack()

        self.cap = cv2.VideoCapture(0)

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

        self.last_prediction = ""
        self.prediction_delay = 30
        self.frame_count = 0

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y])

                    if len(landmarks) == 42:
                        # Predict every N frames to reduce noise
                        if self.frame_count % self.prediction_delay == 0:
                            gesture = model.predict([landmarks])[0]
                            if gesture != self.last_prediction:
                                sentence = gesture_to_sentence.get(gesture, "Unknown gesture")
                                self.sentence_label.config(text=sentence)
                                threading.Thread(target=speak, args=(sentence,)).start()
                                self.last_prediction = gesture

            self.frame_count += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
