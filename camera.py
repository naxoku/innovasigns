import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf

# Cargar modelo y preprocessors
model = tf.keras.models.load_model("sign_language_model_filtered.keras")
with open("label_encoder_filtered.pkl", "rb") as f:
    le = pickle.load(f)
with open("scaler_filtered.pkl", "rb") as f:
    scaler = pickle.load(f)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Función para extraer landmarks de frame
def extract_landmarks_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks, hand_landmarks
    return None, None

# Abrir cámara
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("🔴 Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, hand_landmarks = extract_landmarks_from_frame(frame)
    if landmarks is not None:
        # Preprocesar entrada para modelo
        landmarks_np = np.array(landmarks).reshape(1, -1)
        landmarks_scaled = scaler.transform(landmarks_np)

        # Predecir
        pred_probs = model.predict(landmarks_scaled)
        pred_class = np.argmax(pred_probs)
        pred_label = le.inverse_transform([pred_class])[0].upper()
        confidence = pred_probs[0][pred_class]

        # Dibujar landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar predicción en pantalla
        cv2.putText(frame, f"Prediccion: {pred_label} ({confidence*100:.1f}%)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Detección Lengua de Señas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()