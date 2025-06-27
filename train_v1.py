BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
EPOCHS = 10
# initial_learning_rate = 1e-4


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

# Previne OOM na GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def preprocess_mammogram(img):
    # Converte de float para uint8 (padrão OpenCV)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = np.stack([img]*3, axis=-1)  # Volta para 3 canais
    img = img.astype('float32') / 255.0  # Reescala de volta pra 0-1
    return img

df_train, df_test = carregar_dados()

df_train = df_train[(df_train['pathology'] != 'BENIGN_WITHOUT_CALLBACK') & (df_train['abnormality type'] != 'calcification')]
df_test = df_test[(df_test['pathology'] != 'BENIGN_WITHOUT_CALLBACK') & (df_test['abnormality type'] != 'calcification')]

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
    # preprocessing_function=preprocess_mammogram,
    # rotation_range=10,
    # width_shift_range=0.1,
    # horizontal_flip=True
)

train_gen = datagen.flow_from_dataframe(
    dataframe=df_train,
    shuffle=True,
    x_col='image file path',
    y_col='pathology_binary',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='image file path',
    y_col='pathology_binary',
    target_size=TARGET_SIZE,
    batch_size=1,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Backbone pré-treinado
base_model = tf.keras.applications.ResNet152V2(
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True  # Descongela todas as camadas!

# Construindo a MLP head
inputs = tf.keras.Input(shape=INPUT_SHAPE)
x = base_model(inputs, training=True)  # Agora treinável!
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# Compilar com Focal Loss
model.compile(
        tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
        alpha=0.6,  # Peso para classe minoritária (malignas)
        gamma=2.0   # Foco em amostras difíceis
    ),
    metrics=['accuracy', 'precision', 'recall']
)
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

h = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Carregar os pesos do melhor modelo após o treino
model.load_weights('best_model.keras')

# Gerar previsões no conjunto de teste
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col='image file path',
    y_col='pathology_binary',
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

test_gen.reset()
y_test_pred_probs = model.predict(test_gen, verbose=1)
y_test_pred = (y_test_pred_probs > 0.5).astype(int).flatten()
y_test_true = test_gen.classes

# Classification report (Test)
print("Classification Report (Test):")
print(classification_report(y_test_true, y_test_pred))

# Confusion matrix (Test)
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test_true, y_test_pred))
