BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
EPOCHS = 30

import pandas as pd
import random
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data import carregar_dados
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Define a GPU específica para uso (GPU 0)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Ativa o crescimento de memória na GPU 0
        tf.config.experimental.set_memory_growth(gpus[0], True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configuradas.")

    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")

df_train_final = pd.read_csv('cbis_ddsm_dataset/csv/train.csv')
df_val = pd.read_csv('cbis_ddsm_dataset/csv/val.csv')
df_test_new = pd.read_csv('cbis_ddsm_dataset/csv/test.csv')

df_train_final['pathology_binary'] = df_train_final['pathology_binary'].astype(str)
df_val['pathology_binary'] = df_val['pathology_binary'].astype(str)
df_test_new['pathology_binary'] = df_test_new['pathology_binary'].astype(str)

df = pd.concat([df_train_final, df_val, df_test_new], ignore_index=True)
df_pairs = df[["patient_id", "left or right breast", "image view", "image file path", "pathology_binary"]]

# Cria duas colunas: uma para CC e uma para MLO da mesma mama
df_cc = df_pairs[df_pairs['image view'] == 'CC'].rename(columns={'image file path': 'path_cc'})
df_mlo = df_pairs[df_pairs['image view'] == 'MLO'].rename(columns={'image file path': 'path_mlo'})

# Junta CC com MLO para o mesmo paciente e lado
df_paired = pd.merge(df_cc, df_mlo, on=['patient_id', 'left or right breast', 'pathology_binary'])


df_train_pairs, df_val_pairs = train_test_split(
    df_paired,
    test_size=0.2,
    stratify=df_paired['pathology_binary'],
    random_state=42
)

class DualImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, target_size, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df.iloc[indexes]

        X_cc = np.array([self._load_image(row['path_cc']) for _, row in batch.iterrows()])
        X_mlo = np.array([self._load_image(row['path_mlo']) for _, row in batch.iterrows()])
        y = tf.keras.utils.to_categorical(batch['pathology_binary'], num_classes=2)

        return {'input_cc': X_cc, 'input_mlo': X_mlo}, y

    def _load_image(self, path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=self.target_size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return img




# Entrada dupla (CC e MLO)
input_cc = tf.keras.Input(shape=(224, 224, 3), name='input_cc')
input_mlo = tf.keras.Input(shape=(224, 224, 3), name='input_mlo')

# Compartilham a mesma DenseNet (mesmo backbone)
feature_extractor = tf.keras.applications.DenseNet121(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
feature_extractor.trainable = True

x_cc = feature_extractor(input_cc)
x_cc = layers.GlobalAveragePooling2D()(x_cc)

x_mlo = feature_extractor(input_mlo)
x_mlo = layers.GlobalAveragePooling2D()(x_mlo)

# Concatenar os embeddings
x = layers.Concatenate()([x_cc, x_mlo])
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=[input_cc, input_mlo], outputs=output)

train_gen = DualImageGenerator(df_train_pairs, batch_size=32, target_size=(224, 224))
val_gen = DualImageGenerator(df_val_pairs, batch_size=32, target_size=(224, 224), shuffle=False)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

h = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])

model.load_weights('best_model.keras')
val_gen = DualImageGenerator(df_val_pairs, batch_size=32, target_size=(224, 224), shuffle=False)

# Avaliação na validação
val_preds = model.predict(val_gen)
print(val_preds)
y_pred = np.argmax(val_preds, axis=1)

y_true = df_val_pairs['pathology_binary'].astype(int).values

# Relatório de classificação
classification_report = classification_report(y_true, y_pred)
# matriz de confusão
confusion = confusion_matrix(y_true, y_pred)
print("Classification Report:\n", classification_report)
