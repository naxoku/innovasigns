import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import pickle
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import lecun_normal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings
warnings.filterwarnings('ignore')

# Verificar versión de TensorFlow
print(f"🔧 TensorFlow version: {tf.__version__}")
print(f"🔧 Keras version: {tf.keras.__version__}")

# Configurar TensorFlow para usar GPU si está disponible
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU disponible: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        print(f"⚠️ Error configurando GPU: {e}")
else:
    print("💻 Usando CPU para entrenamiento")

class SignLanguageTrainer:
    def __init__(self):
        self.dataset_paths = [
            r"C:\Users\naxok\OneDrive\Desktop\Planificacion\PYTHON\dataset\datasets\train",
            r"C:\Users\naxok\OneDrive\Desktop\Planificacion\PYTHON\data"
        ]
        self.target_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        self.csv_filename = "landmarks_train_filtered.csv"
        self.model_filename = "sign_language_model_filtered.keras"
        self.label_encoder_filename = "label_encoder_filtered.pkl"
        self.scaler_filename = "scaler_filtered.pkl"
        
        # Configurar MediaPipe con parámetros optimizados para entrenamiento
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks(self, image_path):
        """Extrae landmarks de una imagen de mano"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Error al leer imagen: {image_path}")
                return None
                
            # Mejorar la calidad de la imagen antes del procesamiento
            img = cv2.resize(img, (224, 224))  # Tamaño estándar
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return landmarks
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error procesando {image_path}: {str(e)}")
            return None
    
    def extract_and_save_landmarks(self):
        """Extrae landmarks de todas las imágenes y los guarda en CSV"""
        print("🔍 Extrayendo landmarks de las imágenes...")
        
        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['label'] + [f'{coord}{i}' for i in range(1, 22) for coord in ('x','y','z')]
            writer.writerow(header)

            total = 0
            successful = 0
            sin_mano = 0
            class_counts = {cls: 0 for cls in self.target_classes}

            for base_path in self.dataset_paths:
                if not os.path.exists(base_path):
                    print(f"⚠️ Ruta no encontrada: {base_path}")
                    continue
                    
                for label_folder in os.listdir(base_path):
                    if label_folder.lower() not in self.target_classes:
                        continue
                        
                    label_path = os.path.join(base_path, label_folder)
                    if not os.path.isdir(label_path):
                        continue

                    print(f"📁 Procesando clase: {label_folder}")
                    
                    for filename in os.listdir(label_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            label = label_folder.lower()
                            img_path = os.path.join(label_path, filename)
                            landmarks = self.extract_landmarks(img_path)
                            
                            total += 1
                            if landmarks and len(landmarks) == 63:
                                writer.writerow([label] + landmarks)
                                successful += 1
                                class_counts[label] += 1
                            else:
                                sin_mano += 1

        print(f"\n📊 Resumen de extracción:")
        print(f"Total imágenes procesadas: {total}")
        print(f"Landmarks extraídos exitosamente: {successful}")
        print(f"Imágenes sin mano detectada: {sin_mano}")
        print(f"Tasa de éxito: {(successful/total)*100:.1f}%")
        
        print(f"\n📈 Distribución por clase:")
        for cls, count in class_counts.items():
            print(f"  {cls.upper()}: {count} imágenes")
            
        # Verificar clases faltantes
        missing_classes = [cls for cls, count in class_counts.items() if count == 0]
        if missing_classes:
            print(f"⚠️ Clases sin datos: {', '.join(missing_classes)}")
            
        return successful > 0
    
    def augment_landmarks_df(self, df, target_size=15, noise_factors=[0.005, 0.01, 0.015]):
        """Aumenta datos con múltiples niveles de ruido"""
        print("🔄 Aumentando datos...")
        
        augmented_rows = []
        for label in df['label'].unique():
            current = df[df['label'] == label]
            n_current = len(current)
            
            if n_current < target_size:
                needed = target_size - n_current
                print(f"  Clase {label}: {n_current} -> {target_size} (+{needed})")
                
                for i in range(needed):
                    # Usar diferentes factores de ruido para mayor variabilidad
                    noise_std = noise_factors[i % len(noise_factors)]
                    sample = current.sample(1)
                    landmark_vals = sample.iloc[:, 1:].values.astype(float)
                    
                    # Aplicar ruido gaussiano
                    noisy = landmark_vals + np.random.normal(0, noise_std, size=landmark_vals.shape)
                    
                    # Aplicar pequeñas rotaciones simuladas (opcional)
                    if np.random.random() > 0.7:  # 30% de probabilidad
                        rotation_factor = np.random.uniform(-0.02, 0.02)
                        noisy[:, ::3] += rotation_factor  # Afectar coordenadas X
                    
                    new_row = [label] + noisy.flatten().tolist()
                    augmented_rows.append(new_row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows, columns=df.columns)
            return pd.concat([df, augmented_df], ignore_index=True)
        
        return df
    
    def prepare_data(self):
        """Prepara los datos para el entrenamiento"""
        print("📋 Preparando datos...")
        
        if not os.path.exists(self.csv_filename):
            print("❌ No se encontró el archivo CSV. Ejecutando extracción primero...")
            if not self.extract_and_save_landmarks():
                raise Exception("No se pudieron extraer landmarks")
        
        # Cargar datos
        df = pd.read_csv(self.csv_filename)
        print(f"📊 Datos cargados: {len(df)} filas")
        
        # Filtrar filas con todos los landmarks en 0
        df_clean = df[~(df.iloc[:, 1:] == 0).all(axis=1)]
        print(f"📊 Después de limpiar: {len(df_clean)} filas")
        
        print("\n📈 Distribución original:")
        print(df_clean['label'].value_counts().sort_index())
        
        # Verificar que tenemos datos para todas las clases objetivo
        available_classes = set(df_clean['label'].unique())
        missing_classes = set(self.target_classes) - available_classes
        
        if missing_classes:
            print(f"⚠️ Clases faltantes: {missing_classes}")
            # Filtrar target_classes para incluir solo las disponibles
            self.target_classes = [cls for cls in self.target_classes if cls in available_classes]
            print(f"🎯 Clases finales: {self.target_classes}")
        
        # Augmentar datos
        df_aug = self.augment_landmarks_df(df_clean, target_size=20)
        
        print("\n📈 Distribución después de aumentar:")
        print(df_aug['label'].value_counts().sort_index())
        
        return df_aug
    
    def build_model(self, input_shape, num_classes):
        """Construye el modelo de red neuronal"""
        model = Sequential([
            Input(shape=(input_shape,)),
            
            # Capa de entrada con normalización
            BatchNormalization(),
            Dense(512, activation='selu', kernel_initializer=lecun_normal()),
            AlphaDropout(0.1),
            
            # Capas ocultas
            Dense(256, activation='selu', kernel_initializer=lecun_normal()),
            BatchNormalization(),
            AlphaDropout(0.15),
            
            Dense(128, activation='selu', kernel_initializer=lecun_normal()),
            BatchNormalization(),
            AlphaDropout(0.15),
            
            Dense(64, activation='selu', kernel_initializer=lecun_normal()),
            AlphaDropout(0.1),
            
            # Capa de salida
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Entrena el modelo completo"""
        print("🚀 Iniciando entrenamiento...")
        
        # Preparar datos
        df_aug = self.prepare_data()
        
        # Separar características y etiquetas
        X = df_aug.drop('label', axis=1).values
        y = df_aug['label'].values
        
        # Codificar etiquetas
        le = LabelEncoder()
        le.fit(self.target_classes)  # Usar solo las clases disponibles
        
        y_enc = le.transform(y)
        y_cat = to_categorical(y_enc, num_classes=len(self.target_classes))
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"📊 Datos de entrenamiento: {X_train_scaled.shape}")
        print(f"📊 Datos de validación: {X_val_scaled.shape}")
        
        # Construir modelo
        model = self.build_model(input_shape=X_train_scaled.shape[1], 
                                num_classes=len(self.target_classes))
        
        print(f"🏗️ Modelo construido para {len(self.target_classes)} clases")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=15, 
                restore_best_weights=True, 
                monitor='val_accuracy',
                min_delta=0.001,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model_checkpoint.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar
        self.evaluate_model(model, X_val_scaled, y_val, le, history)
        
        # Guardar modelo y preprocessors
        self.save_model_and_preprocessors(model, le, scaler)
        
        # Cerrar MediaPipe
        self.hands.close()
        
        return model, le, scaler
    
    def evaluate_model(self, model, X_val, y_val, le, history):
        """Evalúa el modelo y muestra métricas"""
        print("\n📊 Evaluando modelo...")
        
        # Predicciones
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        # Métricas básicas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calcular top-3 accuracy manualmente
        top_3_accuracy = self.calculate_top_k_accuracy(y_pred_probs, y_true, k=3)
        
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ Top-3 Accuracy: {top_3_accuracy:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall: {recall:.4f}")
        print(f"✅ F1-Score: {f1:.4f}")
        
        # Reporte de clasificación
        print("\n📋 Reporte de clasificación:")
        target_names = [cls.upper() for cls in self.target_classes]
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.target_classes)))
        
        plt.figure(figsize=(15, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Matriz de Confusión - Validación")
        plt.tight_layout()
        plt.show()
        
        # Gráfico de entrenamiento
        self.plot_training_history(history)
    
    def calculate_top_k_accuracy(self, y_pred_probs, y_true, k=3):
        """Calcula top-k accuracy manualmente"""
        # Obtener los índices de las k predicciones más probables
        top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
        
        # Verificar si la clase verdadera está en las top-k predicciones
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def plot_training_history(self, history):
        """Muestra gráficos del historial de entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Entrenamiento')
        ax1.plot(history.history['val_accuracy'], label='Validación')
        ax1.set_title('Accuracy del Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Entrenamiento')
        ax2.plot(history.history['val_loss'], label='Validación')
        ax2.set_title('Loss del Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model_and_preprocessors(self, model, le, scaler):
        """Guarda el modelo y los preprocessors"""
        print("💾 Guardando modelo y preprocessors...")
        
        # Guardar modelo
        model.save(self.model_filename)
        print(f"✅ Modelo guardado: {self.model_filename}")
        
        # Guardar label encoder
        with open(self.label_encoder_filename, "wb") as f:
            pickle.dump(le, f)
        print(f"✅ Label encoder guardado: {self.label_encoder_filename}")
        
        # Guardar scaler
        with open(self.scaler_filename, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler guardado: {self.scaler_filename}")
        
        # Guardar información adicional
        info = {
            'target_classes': self.target_classes,
            'model_filename': self.model_filename,
            'label_encoder_filename': self.label_encoder_filename,
            'scaler_filename': self.scaler_filename,
            'input_shape': 63,
            'num_classes': len(self.target_classes)
        }
        
        with open("model_info.pkl", "wb") as f:
            pickle.dump(info, f)
        print("✅ Información del modelo guardada: model_info.pkl")

def main():
    """Función principal"""
    print("🎯 Iniciando entrenamiento del modelo de lengua de señas...")
    print("=" * 60)
    
    try:
        trainer = SignLanguageTrainer()
        
        # Verificar que las rutas de datos existen
        valid_paths = []
        for path in trainer.dataset_paths:
            if os.path.exists(path):
                valid_paths.append(path)
                print(f"✅ Ruta válida: {path}")
            else:
                print(f"⚠️ Ruta no encontrada: {path}")
        
        if not valid_paths:
            raise Exception("No se encontraron rutas de datos válidas")
        
        trainer.dataset_paths = valid_paths
        
        # Entrenar modelo
        model, le, scaler = trainer.train_model()
        
        print("\n" + "=" * 60)
        print("🎉 ¡Entrenamiento completado exitosamente!")
        print("=" * 60)
        
        # Mostrar información final
        print(f"📁 Archivos generados:")
        print(f"  - {trainer.model_filename}")
        print(f"  - {trainer.label_encoder_filename}")
        print(f"  - {trainer.scaler_filename}")
        print(f"  - model_info.pkl")
        print(f"  - best_model_checkpoint.keras")
        
        return model, le, scaler
        
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
        return None, None, None
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        print("💡 Sugerencias:")
        print("  - Verifica que las rutas de datos sean correctas")
        print("  - Asegúrate de tener suficientes imágenes para cada clase")
        print("  - Verifica que MediaPipe esté instalado correctamente")
        print("  - Comprueba que tienes suficiente memoria disponible")
        raise

if __name__ == "__main__":
    main()