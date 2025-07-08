import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import pickle
import os
import csv
from pathlib import Path
import warnings

# Importaciones de Scikit-learn organizadas y corregidas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, AlphaDropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.initializers import lecun_normal
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

# --- Verificación de Entorno ---
print(f"🔧 TensorFlow version: {tf.__version__}")
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
        # MODIFICADO: La ruta base ahora es la carpeta del propio script. ¡Esta es la corrección clave!
        self.base_path = Path(__file__).resolve().parent
        
        self.output_path = self.base_path / "outputs"
        self.output_path.mkdir(exist_ok=True) 

        self.dataset_paths = [
            self.base_path / "dataset/train",
            self.base_path / "data"
        ]
        
        self.csv_filename = self.output_path / "landmarks_train.csv"
        self.model_filename = self.output_path / "sign_language_model.keras"
        self.label_encoder_filename = self.output_path / "label_encoder.pkl"
        self.scaler_filename = self.output_path / "scaler.pkl"
        self.model_info_filename = self.output_path / "model_info.pkl"
        self.best_model_checkpoint = self.output_path / 'best_model.keras'

        self.training_params = {
            'test_size': 0.2,
            'random_state': 42,
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        self.augmentation_params = {
            'target_min_samples': 30,
            'noise_factors': [0.005, 0.01, 0.015]
        }
        
        # Abecedario en Lengua de Señas Chilena (excluyendo 'ñ' y 'z' que son dinámicas)
        self.target_classes = sorted(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 
                                      'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y'])
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, image_path):
        """Extrae landmarks de una imagen de mano."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️ Error al leer imagen: {image_path}")
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = [lm for lm_obj in hand.landmark for lm in (lm_obj.x, lm_obj.y, lm_obj.z)]
                return landmarks
            return None
        except Exception as e:
            print(f"❌ Error procesando {image_path}: {e}")
            return None

    def create_landmarks_file(self):
        """Extrae landmarks de todas las imágenes y los guarda en un archivo CSV."""
        print("🔍 Extrayendo landmarks de las imágenes...")
        
        processed_count = 0
        success_count = 0
        no_hand_count = 0
        class_counts = {cls: 0 for cls in self.target_classes}

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['label'] + [f'{coord}{i}' for i in range(21) for coord in ('x', 'y', 'z')]
            writer.writerow(header)

            for base_path in self.dataset_paths:
                if not base_path.exists():
                    print(f"⚠️ Ruta no encontrada, saltando: {base_path}")
                    continue
                
                print(f"📂 Procesando directorio: {base_path}")
                for label_folder in sorted(os.listdir(base_path)):
                    label = label_folder.lower()
                    if label not in self.target_classes:
                        continue
                    
                    label_path = base_path / label_folder
                    if not label_path.is_dir():
                        continue

                    # Busca imágenes .jpg, .jpeg y .png
                    for img_file in label_path.glob('*.[jp][pn]g'):
                        processed_count += 1
                        landmarks = self.extract_landmarks(img_file)
                        
                        if landmarks and len(landmarks) == 63:
                            writer.writerow([label] + landmarks)
                            success_count += 1
                            class_counts[label] += 1
                        else:
                            no_hand_count += 1
        
        print("\n📊 Resumen de extracción:")
        print(f"Total imágenes encontradas: {processed_count}")
        print(f"Landmarks extraídos exitosamente: {success_count}")
        print(f"Imágenes sin mano detectada: {no_hand_count}")
        if processed_count > 0:
            print(f"Tasa de éxito: {(success_count / processed_count) * 100:.1f}%")
        
        print("\n📈 Distribución por clase:")
        for cls, count in class_counts.items():
            print(f"   {cls.upper()}: {count} imágenes")
        
        missing_classes = [cls for cls, count in class_counts.items() if count == 0]
        if missing_classes:
            print(f"\n⚠️ ¡Atención! No se encontraron imágenes para las clases: {', '.join(missing_classes)}")
            
        return success_count > 0

    def augment_landmarks(self, df):
        """Aumenta los datos para balancear las clases con pocas muestras."""
        print("🔄 Aumentando datos para clases minoritarias...")
        
        augmented_rows = []
        target_size = self.augmentation_params['target_min_samples']
        noise_factors = self.augmentation_params['noise_factors']
        
        for label in df['label'].unique():
            current_df = df[df['label'] == label]
            n_current = len(current_df)
            
            if 0 < n_current < target_size:
                needed = target_size - n_current
                print(f"  Clase '{label}': {n_current} -> {target_size} (añadiendo {needed})")
                
                for i in range(needed):
                    noise_std = noise_factors[i % len(noise_factors)]
                    sample = current_df.sample(1)
                    landmark_vals = sample.iloc[:, 1:].values.astype(float)
                    
                    noise = np.random.normal(0, noise_std, size=landmark_vals.shape)
                    new_row = [label] + (landmark_vals + noise).flatten().tolist()
                    augmented_rows.append(new_row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows, columns=df.columns)
            return pd.concat([df, augmented_df], ignore_index=True)
        
        return df

    def prepare_data(self):
        """Carga, limpia y prepara los datos para el entrenamiento."""
        print("\n📋 Preparando datos...")
        
        # Siempre re-extraer para asegurar que los datos estén frescos
        if not self.create_landmarks_file():
            raise RuntimeError("Falló la extracción de landmarks. No se puede continuar.")
        
        df = pd.read_csv(self.csv_filename)
        print(f"Cargadas {len(df)} filas desde {self.csv_filename}")
        
        # Elimina filas donde todos los landmarks son 0 (error de extracción)
        df = df.loc[~(df.iloc[:, 1:] == 0).all(axis=1)]
        print(f"Después de limpiar datos inválidos: {len(df)} filas")
        
        if df.empty:
            return df # Retorna el DataFrame vacío para que el flujo principal lo maneje

        print("\n📈 Distribución de clases original:")
        print(df['label'].value_counts().sort_index())

        # Adaptar target_classes a las clases que realmente tienen datos
        available_classes = sorted(list(df['label'].unique()))
        if set(self.target_classes) != set(available_classes):
             print("\n⚠️ Actualizando lista de clases a las disponibles en el dataset.")
             self.target_classes = available_classes
        
        df_aug = self.augment_landmarks(df)
        
        print("\n📈 Distribución después de aumentar:")
        print(df_aug['label'].value_counts().sort_index())
        
        return df_aug

    def build_model(self, input_shape, num_classes):
        """Construye el modelo de red neuronal con una arquitectura auto-normalizada."""
        print(f"\n🏗️ Construyendo modelo para {num_classes} clases con input_shape ({input_shape},)")
        model = Sequential([
            Input(shape=(input_shape,)),
            BatchNormalization(),
            Dense(512, activation='selu', kernel_initializer=lecun_normal()),
            AlphaDropout(0.2),
            Dense(256, activation='selu', kernel_initializer=lecun_normal()),
            BatchNormalization(),
            AlphaDropout(0.2),
            Dense(128, activation='selu', kernel_initializer=lecun_normal()),
            BatchNormalization(),
            AlphaDropout(0.2),
            Dense(64, activation='selu', kernel_initializer=lecun_normal()),
            AlphaDropout(0.1),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.training_params['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        return model

    def train(self):
        """Orquesta todo el proceso de entrenamiento."""
        print("\n" + "="*60)
        print("🚀 INICIANDO PROCESO DE ENTRENAMIENTO 🚀")
        print("="*60)
        
        df = self.prepare_data()
        
        if df.empty:
            print("\n❌ No se encontraron datos válidos para entrenar. Abortando.")
            return

        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        # Asegurarse de que el número de clases sea el correcto
        y_cat = to_categorical(y_enc, num_classes=len(le.classes_))
        self.target_classes = list(le.classes_)

        scaler = StandardScaler()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cat, 
            test_size=self.training_params['test_size'], 
            random_state=self.training_params['random_state'], 
            stratify=y_enc
        )
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"\n📊 Datos de entrenamiento: {X_train_scaled.shape}")
        print(f"📊 Datos de validación: {X_val_scaled.shape}")
        
        model = self.build_model(input_shape=X_train_scaled.shape[1], num_classes=len(self.target_classes))
        
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy', verbose=1),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            ModelCheckpoint(filepath=self.best_model_checkpoint, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        print("\n💪 Entrenando el modelo...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=self.training_params['epochs'],
            batch_size=self.training_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        self.evaluate(model, X_val_scaled, y_val, le, history)
        
        self.save_artifacts(model, le, scaler)
        
        self.hands.close()
        print("\n" + "="*60)
        print("🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE! 🎉")
        print("="*60)

    def evaluate(self, model, X_val, y_val, le, history):
        """Evalúa el modelo, muestra métricas y guarda los gráficos."""
        print("\n📈 Evaluando el modelo final...")
        
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n✅ Accuracy final en validación: {accuracy:.4f}")
        
        print("\n📋 Reporte de clasificación:")
        target_names = [cls.upper() for cls in le.classes_]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        self.plot_training_history(history)
        self.plot_confusion_matrix(y_true, y_pred, le.classes_)

    def plot_training_history(self, history):
        """Muestra y guarda gráficos del historial de entrenamiento."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        ax1.plot(history.history['accuracy'], label='Entrenamiento')
        ax1.plot(history.history['val_accuracy'], label='Validación')
        ax1.set_title('Evolución del Accuracy')
        ax1.set_xlabel('Época'); ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Entrenamiento')
        ax2.plot(history.history['val_loss'], label='Validación')
        ax2.set_title('Evolución del Loss')
        ax2.set_xlabel('Época'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True)
        
        fig.suptitle('Resultados del Entrenamiento', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_path / "training_history.png")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """Muestra y guarda la matriz de confusión."""
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
        fig, ax = plt.subplots(figsize=(14, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[l.upper() for l in labels])
        disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
        ax.set_title("Matriz de Confusión - Validación")
        plt.tight_layout()
        plt.savefig(self.output_path / "confusion_matrix.png")
        plt.show()

    def save_artifacts(self, model, le, scaler):
        """Guarda el modelo final y los preprocesadores."""
        print(f"\n💾 Guardando artefactos en la carpeta: '{self.output_path}'")
        
        model.save(self.model_filename)
        print(f"✅ Modelo guardado: {self.model_filename}")
        
        with open(self.label_encoder_filename, "wb") as f:
            pickle.dump(le, f)
        print(f"✅ Label encoder guardado: {self.label_encoder_filename}")
        
        with open(self.scaler_filename, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler guardado: {self.scaler_filename}")
        
        info = {
            'target_classes': self.target_classes,
            'input_shape': model.input_shape[1],
            'model_filename': str(self.model_filename.name),
            'label_encoder_filename': str(self.label_encoder_filename.name),
            'scaler_filename': str(self.scaler_filename.name),
        }
        with open(self.model_info_filename, "wb") as f:
            pickle.dump(info, f)
        print(f"✅ Información del modelo guardada: {self.model_info_filename}")


if __name__ == "__main__":
    try:
        trainer = SignLanguageTrainer()
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento interrumpido por el usuario.")
    except Exception as e:
        print(f"\n❌ Se ha producido un error crítico: {e}")
        print("\n💡 Sugerencias:")
        print("  - Revisa la consola para asegurarte de que se están procesando imágenes.")
        print("  - Asegúrate de que las carpetas 'data' y 'dataset/train' existen y tienen imágenes.")
        print("  - Verifica que las dependencias (tensorflow, mediapipe, etc.) están instaladas.")