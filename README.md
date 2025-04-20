🤟 Hand Gesture Translator (Sign Language to Text & Speech)
This project is a real-time hand gesture translator that converts sign language (using hand and finger gestures) into text and speech using computer vision and machine learning. It is designed to assist individuals with hearing or speech impairments in communicating more effectively with others.

✨ Features
🎥 Real-Time Gesture Detection via Webcam

🧠 Gesture Classification using a trained Machine Learning model (Random Forest Classifier)

📝 Text Display of recognized hand gestures

🔊 Speech Output using Offline Text-to-Speech (TTS)

🖥️ User-Friendly GUI built with Tkinter

🛠️ Tech Stack
Frontend: Tkinter (Python GUI Library)

Computer Vision: OpenCV & MediaPipe

Machine Learning:

Scikit-learn (Random Forest Classifier)

Pickle (for model serialization)

Data Handling: CSV (for storing hand landmarks and labels)

Text-to-Speech: pyttsx3 (offline TTS engine)

🔍 How It Works
Webcam captures the user’s hand gestures in real-time.

MediaPipe extracts hand landmarks (x, y, z coordinates).

These landmarks are fed into a trained Random Forest Classifier.

The model predicts the corresponding sign/gesture label.

The label is displayed on screen and spoken using the TTS engine.

🚀 Future Improvements
👐 Support for multiple hands and multi-gesture detection

📚 Expansion of the gesture dataset for more comprehensive sign coverage

🧩 Sentence construction from sequences of detected gestures

🌍 Language translation for multilingual support

💡 Use Case
This tool can bridge communication gaps for people with hearing or speech disabilities by enabling them to express themselves using sign language, which is then translated into text and voice for others to understand.
