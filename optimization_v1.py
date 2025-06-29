import os
import json
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from data import carregar_dados  # adapte conforme sua estrutura real
from generator import DualImageGenerator  # adapte conforme necessário

# ==== CONFIG GLOBAL ====
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
FREEZE_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

# ===== DADOS ====
df_train_final = pd.read_csv('train.csv')
df_val = pd.read_csv('val.csv')
df_test_new = pd.read_csv('test.csv')

for df in [df_train_final, df_val, df_test_new]:
    df['pathology_binary'] = df['pathology_binary'].astype(str)

df = pd.concat([df_train_final, df_val, df_test_new], ignore_index=True)

df_pairs = df[["patient_id", "left or right breast", "image view", "image file path", "pathology_binary"]]
df_cc = df_pairs[df_pairs['image view'] == 'CC'].rename(columns={'image file path': 'path_cc'})
df_mlo = df_pairs[df_pairs['image view'] == 'MLO'].rename(columns={'image file path': 'path_mlo'})
df_paired = pd.merge(df_cc, df_mlo, on=['patient_id', 'left or right breast', 'pathology_binary'])

df_train_pairs, df_val_pairs = train_test_split(
    df_paired, test_size=0.2,
    stratify=df_paired['pathology_binary'],
    random_state=42
)

# ==== FUNÇÕES AUXILIARES ====
def get_next_run_path(backbone):
    base_path = f"resultados/{backbone}"
    os.makedirs(base_path, exist_ok=True)
    runs = [d for d in os.listdir(base_path) if d.startswith("run_")]
    run_id = len(runs) + 1
    path = os.path.join(base_path, f"run_{run_id}")
    os.makedirs(path)
    return path

def save_config(run_path, config):
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def save_metrics(run_path, y_true, y_pred, history):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    with open(os.path.join(run_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(run_path, "historico.json"), "w") as f:
        json.dump(history.history, f, indent=4)
    return metrics

def save_plot(run_path, history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_path, "acuracia_loss.png"))
    plt.close()

# ==== MODELO ====
def create_model(backbone_name, dense_units, dropout_rate, use_l2):
    input_cc = tf.keras.Input(shape=INPUT_SHAPE, name='input_cc')
    input_mlo = tf.keras.Input(shape=INPUT_SHAPE, name='input_mlo')

    if backbone_name == "densenet121":
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    elif backbone_name == "densenet169":
        base_model = tf.keras.applications.DenseNet169(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    else:
        raise ValueError("Backbone desconhecido")

    regularizer = regularizers.l2(1e-4) if use_l2 else None

    x_cc = base_model(input_cc)
    x_cc = layers.GlobalAveragePooling2D()(x_cc)

    x_mlo = base_model(input_mlo)
    x_mlo = layers.GlobalAveragePooling2D()(x_mlo)

    x = layers.Concatenate()([x_cc, x_mlo])
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizer)(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_cc, input_mlo], outputs=output)
    return model, base_model

# ==== OBJETIVO OPTUNA ====
def objective(trial):
    # Hiperparâmetros
    backbone = trial.suggest_categorical("backbone", ["densenet121", "densenet169"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    fine_tune_lr = trial.suggest_loguniform("fine_tune_lr", 1e-6, 5e-5)
    dense_units = trial.suggest_categorical("dense_units", [64, 128, 256])
    dropout_rate = trial.suggest_categorical("dropout", [0.3, 0.5])
    use_l2 = trial.suggest_categorical("use_l2", [True, False])
    use_augment = trial.suggest_categorical("augment", [True, False])

    run_path = get_next_run_path(backbone)
    save_config(run_path, {
        "backbone": backbone,
        "lr": lr,
        "fine_tune_lr": fine_tune_lr,
        "dense_units": dense_units,
        "dropout": dropout_rate,
        "l2": use_l2,
        "augment": use_augment,
        "batch_size": BATCH_SIZE
    })

    train_gen = DualImageGenerator(df_train_pairs, BATCH_SIZE, TARGET_SIZE, augment=use_augment)
    val_gen = DualImageGenerator(df_val_pairs, BATCH_SIZE, TARGET_SIZE, shuffle=False)

    # Criar modelo com base_model congelado
    model, base_model = create_model(backbone, dense_units, dropout_rate, use_l2)
    base_model.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Treinamento inicial (congelado)
    checkpoint = ModelCheckpoint(os.path.join(run_path, "model.keras"), monitor="val_loss", save_best_only=True)
    # early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history1 = model.fit(train_gen, validation_data=val_gen, epochs=FREEZE_EPOCHS, callbacks=[checkpoint], verbose=0)

    # Fine-tuning (descongelar)
    base_model.trainable = True
    print("Descongelando camadas do backbone...")
    model.compile(optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    print("Treinando modelo com backbone descongelado...")
    history2 = model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS, callbacks=[checkpoint], verbose=0)

    # Avaliação
    y_true = df_val['pathology_binary'].astype(int).values
    val_preds = model.predict(val_gen)
    y_pred = np.argmax(val_preds, axis=1)

    # Juntar histórico
    history1.history = {k: history1.history[k] + history2.history[k] for k in history1.history}
    metrics = save_metrics(run_path, y_true, y_pred, history1)
    save_plot(run_path, history1)

    return metrics["accuracy"]

# ==== EXECUÇÃO ====
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)  # ⬅ altere quantos testes quiser

    with open("resultados/resumo_study.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.params,
            "value": study.best_value
        }, f, indent=4)

    print("Melhor configuração:", study.best_trial.params)
    print("Melhor val_accuracy:", study.best_value)