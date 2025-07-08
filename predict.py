import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from pathlib import Path

class SignPredictor:
    def __init__(self, model_dir='outputs'):
        """
        Inicializa el predictor cargando todos los artefactos del entrenamiento.
        """
        # AÑADIDO: La ruta base ahora es la carpeta del propio script para que siempre funcione
        self.base_path = Path(__file__).resolve().parent
        model_path = self.base_path / model_dir # Construir la ruta completa a la carpeta 'outputs'
        
        print(f"📦 Cargando artefactos desde el directorio: '{model_path}'")
        
        info_path = model_path / "model_info.pkl"
        if not info_path.exists():
            raise FileNotFoundError(f"No se encontró 'model_info.pkl' en '{model_path}'. "
                                    "Asegúrate de que el directorio del modelo es correcto y contiene los archivos de entrenamiento.")
        
        with open(info_path, "rb") as f:
            model_info = pickle.load(f)

        self.model = tf.keras.models.load_model(model_path / model_info['model_filename'])
        with open(model_path / model_info['label_encoder_filename'], "rb") as f:
            self.le = pickle.load(f)
        with open(model_path / model_info['scaler_filename'], "rb") as f:
            self.scaler = pickle.load(f)
            
        print("✅ Modelo y preprocesadores cargados exitosamente.")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.6
        )

    def predict_from_image(self, image_path):
        """
        Realiza una predicción sobre una sola imagen y la muestra.
        """
        # AÑADIDO: Construir la ruta completa a la imagen
        full_image_path = self.base_path / image_path
        
        img = cv2.imread(str(full_image_path))
        if img is None:
            print(f"❌ Error: No se pudo leer la imagen en la ruta: {full_image_path}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print("🖐️ No se detectó ninguna mano en la imagen.")
            cv2.imshow("Resultado", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        
        if landmarks.shape[0] != 63:
            print(f"⚠️ Se detectaron {landmarks.shape[0]} coordenadas en lugar de 63. No se puede predecir.")
            return
            
        X_input = self.scaler.transform([landmarks])
        prediction = self.model.predict(X_input)
        
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = self.le.inverse_transform([predicted_index])[0]

        print(f"LETRA PREDECIDA: '{predicted_label.upper()}' (Confianza: {confidence:.2%})")
        self.display_result(img, hand_landmarks, predicted_label, confidence)

    def display_result(self, image, hand_landmarks, label, confidence):
        """
        Dibuja los landmarks y la predicción en la imagen y la muestra.
        """
        self.mp_drawing.draw_landmarks(
            image, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
        )
        text = f"Prediccion: {label.upper()} ({confidence:.1%})"
        cv2.rectangle(image, (10, 30), (350, 70), (0, 0, 0), -1)
        cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Resultado de la Prediccion", image)
        print("\nPresiona cualquier tecla para cerrar la ventana de la imagen...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    try:
        # ------------------------------------------------------------------
        # AQUÍ PONES LA RUTA RELATIVA DE LA IMAGEN QUE QUIERES PROBAR
        # La ruta es relativa a la carpeta 'innovasigns'
        # ------------------------------------------------------------------
        imagen_a_probar = "dataset/test/B/7487b589-a185-4ba0-a417-f8f64db3a03d.jpg"
        
        # El directorio del modelo por defecto es 'outputs'
        predictor = SignPredictor(model_dir="outputs")
        predictor.predict_from_image(imagen_a_probar)
        
    except FileNotFoundError as e:
        print(f"❌ Error al cargar los archivos: {e}")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()