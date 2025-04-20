import cv2
import mediapipe as mp
import csv

gesture_label = input("Enter gesture label (e.g. A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W): ")
filename = "gesture_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    print("Press 's' to save a frame, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                cv2.putText(frame, f"Gesture: {gesture_label} | Press 's' to save", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    writer.writerow([gesture_label] + landmarks)
                    print(f"Saved gesture: {gesture_label}")

        cv2.imshow("Collect Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
