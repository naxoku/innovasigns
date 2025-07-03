import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os
import csv
import matplotlib.pyplot as plt

def main():
    # Cargar modelo
    model = tf.keras.models.load_model("sign_language_model_filtered.keras")

    # Cargar label encoder
    with open("label_encoder_filtered.pkl", "rb") as f:
        le = pickle.load(f)

    # Cargar scaler
    with open("scaler_filtered.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Configurar MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    def predict_from_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("❌ No se pudo leer la imagen.")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print("🖐️ No se detectó ninguna mano en la imagen.")
            return

        # Extraer landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) != 63:
            print("⚠️ La cantidad de landmarks no es válida.")
            return

        X_input = np.array(landmarks).reshape(1, -1)
        X_scaled = scaler.transform(X_input)  # ¡Importante!

        prediction = model.predict(X_scaled)
        predicted_index = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_index])[0]

        print(f"✅ Predicción: {predicted_label.upper()}")
        
        mp_drawing = mp.solutions.drawing_utils

        # Dibujar los landmarks detectados
        img_copy = img.copy()
        mp_drawing.draw_landmarks(img_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar la imagen con landmarks
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.title("Landmarks detectados")
        plt.axis('off')
        plt.show()

    # TESTEAR AQUÍ
    predict_from_image("./dataset/datasets/test/E/1f25f60e-7742-42e7-a93d-8884d3bf9472.jpg")
    predict_from_image("./dataset/datasets/test/A/20190428_235302.jpg")
    predict_from_image("./dataset/datasets/test/A/IMG-20190430-WA0003.jpg")
    predict_from_image("./dataset/datasets/test/A/IMG-20190505-WA0013.jpg")
    predict_from_image("./dataset/datasets/test/L/bccdf985-4c90-4608-9053-2d2c96b64391.jpg")
if __name__ == "__main__":
    main()