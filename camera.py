import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
from collections import deque # AÑADIDO: Para el suavizado de predicciones

class CameraPredictor:
    def __init__(self):
        # AÑADIDO: Carga robusta de archivos desde la carpeta 'outputs'
        self._load_artifacts()

        # AÑADIDO: Parámetros para optimización y suavizado
        self.prediction_interval = 5  # Predecir cada 5 frames
        self.smoothing_window_size = 10 # Usar las últimas 10 predicciones para suavizar
        
        # AÑADIDO: Variables de estado
        self.frame_counter = 0
        self.recent_predictions = deque(maxlen=self.smoothing_window_size)
        self.stable_prediction = ""
        self.last_confidence = 0.0

        # Configuración de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Configuración de la cámara
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Cámara 1 no encontrada. Intentando con cámara 0...")
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("No se pudo abrir ninguna cámara.")
    
    def _load_artifacts(self, model_dir='outputs'):
        """Carga el modelo, el scaler y el label encoder."""
        print("📦 Cargando artefactos del modelo...")
        base_path = Path(__file__).resolve().parent
        model_path = base_path / model_dir
        
        info_path = model_path / "model_info.pkl"
        if not info_path.exists():
            raise FileNotFoundError(f"No se encontró 'model_info.pkl' en '{model_path}'.")
        
        with open(info_path, "rb") as f:
            model_info = pickle.load(f)

        self.model = tf.keras.models.load_model(model_path / model_info['model_filename'])
        with open(model_path / model_info['label_encoder_filename'], "rb") as f:
            self.le = pickle.load(f)
        with open(model_path / model_info['scaler_filename'], "rb") as f:
            self.scaler = pickle.load(f)
        print("✅ Artefactos cargados exitosamente.")

    def run(self):
        """Inicia el bucle principal para la captura y predicción en tiempo real."""
        print("🔴 Presiona 'q' para salir")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Invertir el frame horizontalmente para un efecto espejo
            frame = cv2.flip(frame, 1)

            # Procesar el frame para encontrar la mano
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            self.frame_counter += 1

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Dibujar los landmarks en cada frame para una respuesta visual inmediata
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # AÑADIDO: Realizar predicción solo cada N frames para optimizar
                if self.frame_counter % self.prediction_interval == 0:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    
                    if landmarks.shape[0] == 63:
                        landmarks_scaled = self.scaler.transform([landmarks])
                        pred_probs = self.model.predict(landmarks_scaled)[0]
                        
                        pred_index = np.argmax(pred_probs)
                        self.last_confidence = pred_probs[pred_index]
                        
                        # AÑADIDO: Añadir predicción a la lista para suavizado
                        self.recent_predictions.append(pred_index)
                        
                        # AÑADIDO: Calcular la predicción más estable
                        if self.recent_predictions:
                            most_common_pred = max(set(self.recent_predictions), key=self.recent_predictions.count)
                            self.stable_prediction = self.le.inverse_transform([most_common_pred])[0].upper()

            # Mostrar la predicción estable en cada frame
            text = f"Letra: {self.stable_prediction} ({self.last_confidence:.1%})"
            cv2.rectangle(frame, (5, 5), (400, 50), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Deteccion en Tiempo Real", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._cleanup()

    def _cleanup(self):
        """Libera todos los recursos al salir."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("👋 Saliendo. ¡Recursos liberados!")

if __name__ == '__main__':
    try:
        predictor = CameraPredictor()
        predictor.run()
    except Exception as e:
        print(f"❌ Ocurrió un error crítico: {e}")