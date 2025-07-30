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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

# Configuração GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configuradas.")
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")

# === CONFIGS ===
STORAGE_PATH = "sqlite:///resultados/optuna_study.db"
STUDY_NAME = "teste_mecanismos_atencao_simam_none"
# ==== CONFIG GLOBAL ====
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
FREEZE_EPOCHS = 5
FINE_TUNE_EPOCHS = 25
N_TRIALS = 10
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


class SimAM(tf.keras.layers.Layer):
    def __init__(self, e_lambda=1e-4, **kwargs):
        super(SimAM, self).__init__(**kwargs)
        self.e_lambda = e_lambda

    def call(self, x):
        # x: [B, H, W, C]
        # Transpor para [B, C, H, W] se necessário, dependendo do seu modelo
        # Calcula a média espacial por canal
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        # Desvio ao quadrado (x - mean)^2
        d = tf.square(x - mean)
        # Número de elementos espaciais (H * W - 1)
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        n = tf.cast(h * w - 1, tf.float32)
        # Variância espacial por canal (soma sem keepdims)
        var = tf.reduce_sum(d, axis=[1, 2], keepdims=True) / n
        # Equação do paper: attention = sigmoid( (x - mean)^2 / (4*(var + lambda)) + 0.5 )
        e_inv = d / (4.0 * (var + self.e_lambda)) + 0.5
        attention = tf.sigmoid(e_inv)
        return x * attention


def se_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([input_tensor, se])


def eca_block(input_tensor, k_size=3):
    filters = input_tensor.shape[-1]
    x = layers.GlobalAveragePooling2D()(input_tensor)
    x = layers.Reshape((filters, 1))(x)
    x = layers.Conv1D(1, kernel_size=k_size, padding='same', use_bias=False)(x)
    x = layers.Activation('sigmoid')(x)
    x = layers.Reshape((1, 1, filters))(x)
    return layers.Multiply()([input_tensor, x])


# Verificar se os paths existem
def check_paths(df):
    for _, row in df.iterrows():
        if not os.path.exists(row['path_cc']):
            raise FileNotFoundError(f"Arquivo não encontrado: {row['path_cc']}")
        if not os.path.exists(row['path_mlo']):
            raise FileNotFoundError(f"Arquivo não encontrado: {row['path_mlo']}")

check_paths(df_paired)

# df_train_pairs, df_test_pairs = train_test_split(
#     df_paired, test_size=0.2,
#     stratify=df_paired['pathology_binary'],
#     random_state=42
# )

# df_train_pairs, df_val_pairs = train_test_split(
#     df_train_pairs, test_size=0.125,
#     stratify=df_train_pairs['pathology_binary'],
#     random_state=42
# )

# #salvar os DataFrames para uso posterior
# df_train_pairs.to_csv('train_pairs.csv', index=False)
# df_val_pairs.to_csv('val_pairs.csv', index=False)
# df_test_pairs.to_csv('test_pairs.csv', index=False)
df_train_pairs = pd.read_csv('train_pairs.csv')
df_val_pairs = pd.read_csv('val_pairs.csv')
df_test_pairs = pd.read_csv('test_pairs.csv')

# ==== DUAL IMAGE GENERATOR ====
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

class DualImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, target_size, augment=False, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
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
        if self.augment:
            img = datagen.random_transform(img)
        return img

# ==== FUNÇÕES AUXILIARES ====
def get_next_run_path(backbone):
    base_path = f"resultados/{backbone}"
    os.makedirs(base_path, exist_ok=True)
    runs = [d for d in os.listdir(base_path) if d.startswith("run_")]
    run_id = len(runs) + 1
    path = os.path.join(base_path, f"run_{run_id}")
    os.makedirs(path, exist_ok=True)
    return path

def save_config(run_path, config):
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def save_metrics(run_path, y_true, y_pred, history):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary'),
        "recall": recall_score(y_true, y_pred, average='binary'),
        "f1_score": f1_score(y_true, y_pred, average='binary')
    }
    with open(os.path.join(run_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(run_path, "historico.json"), "w") as f:
        json.dump(history.history, f, indent=4)
    #matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    #usando seaborn para plotar a matriz de confusão e salvar em run_path
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benigno', 'Maligno'],
                yticklabels=['Benigno', 'Maligno'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.savefig(os.path.join(run_path, "matriz_confusao.png"))
    plt.close() 
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

def create_model(backbone_name, dense_units, dropout_rate, use_l2, use_dense_2=False, attention_module=None):
    input_cc = tf.keras.Input(shape=INPUT_SHAPE, name='input_cc')
    input_mlo = tf.keras.Input(shape=INPUT_SHAPE, name='input_mlo')

    if backbone_name == "densenet121":
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    elif backbone_name == "densenet169":
        base_model = tf.keras.applications.DenseNet169(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    elif backbone_name == "densenet201":
        base_model = tf.keras.applications.DenseNet201(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    else:
        raise ValueError("Backbone desconhecido")

    regularizer = regularizers.l2(1e-4) if use_l2 else None

    def apply_attention(x):
        if attention_module == "se":
            return se_block(x)
        elif attention_module == "eca":
            return eca_block(x)
        elif attention_module == "simam":
            return SimAM()(x)
        elif attention_module == "none":
            return x
        return x

    x_cc = base_model(input_cc)
    x_cc = apply_attention(x_cc)
    x_cc = layers.GlobalAveragePooling2D()(x_cc)

    x_mlo = base_model(input_mlo)
    x_mlo = apply_attention(x_mlo)
    x_mlo = layers.GlobalAveragePooling2D()(x_mlo)
    x = layers.Concatenate()([x_cc, x_mlo])
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizer)(x)
    x = layers.Dropout(dropout_rate)(x)
    if use_dense_2:
        x = layers.Dense(dense_units // 2, activation='relu', kernel_regularizer=regularizer)(x)
        x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_cc, input_mlo], outputs=output)
    return model, base_model

# ==== OBJETIVO OPTUNA ====
def objective(trial):
    # Hiperparâmetros
    backbone = trial.suggest_categorical("backbone", ["densenet121"])
    lr = trial.suggest_categorical("lr", [1e-3])
    fine_tune_lr = trial.suggest_categorical("fine_tune_lr", [1e-4])
    dense_units = trial.suggest_categorical("dense_units", [256])
    use_dense_2 = trial.suggest_categorical("use_dense_2", [False, False]) #sem usar duas camadas ocultas
    dropout_rate = trial.suggest_categorical("dropout", [0.4])
    use_l2 = trial.suggest_categorical("use_l2", [True])
    use_augment = trial.suggest_categorical("augment", [True])
    unfreeze_layers = trial.suggest_categorical("unfreeze_layers", [54])  # novo hiperparâmetro
    attention_module = trial.suggest_categorical("attention_module", ["none", "simam"])


    run_path = get_next_run_path(backbone)
    save_config(run_path, {
        "backbone": backbone,
        "lr": lr,
        "fine_tune_lr": fine_tune_lr,
        "dense_units": dense_units,
        "dropout": dropout_rate,
        "l2": use_l2,
        "augment": use_augment,
        "unfreeze_layers": unfreeze_layers,
        "batch_size": BATCH_SIZE,
        "use_dense_2": use_dense_2,
        "dense_units_2": dense_units//2 if use_dense_2 else "N/A",
        "attention_module": attention_module
    })
    print("\n" + "="*50)
    print(f"Trial {trial.number} - Config:")
    print(f"Backbone: {backbone}, LR: {lr:.1e}, Dense Units: {dense_units}")
    print(f"Dropout: {dropout_rate}, L2: {use_l2}, Augment: {use_augment}")
    print(f"Unfreeze Layers: {unfreeze_layers}, Use Dense 2: {use_dense_2}, Attention Module: {attention_module}")
    print("="*50 + "\n")


    train_gen = DualImageGenerator(df_train_pairs, BATCH_SIZE, TARGET_SIZE, augment=use_augment, shuffle=True)
    val_gen = DualImageGenerator(df_val_pairs, BATCH_SIZE, TARGET_SIZE, shuffle=False)
    test_gen = DualImageGenerator(df_test_pairs, BATCH_SIZE, TARGET_SIZE, shuffle=False)

    # Criar modelo
    model, base_model = create_model(backbone, dense_units, dropout_rate, use_l2, use_dense_2, attention_module=attention_module)
    
    # Congelar todo o backbone inicialmente
    base_model.trainable = False

    # Compilar com otimizador 1
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Callbacks
    checkpoint = ModelCheckpoint(os.path.join(run_path, "model_frozen.keras"), monitor="val_loss", save_best_only=True)
    # early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    print(f"==> Trial {trial.number} | Treinando com camadas congeladas...")
    history1 = model.fit(train_gen, validation_data=val_gen, epochs=FREEZE_EPOCHS, callbacks=[checkpoint], verbose=1)

    # Fine-tuning: descongelar parcialmente
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    # Carregar o melhor modelo do treino congelado
    # print(f"==> Carregando os melhores pesos do treino congelado...") # TODO PROVAVELMENTE PIOROU QUANDO CARREGA OS PESOS DO TREINO CONGELADO
    # model.load_weights(os.path.join(run_path, "model_frozen.keras"))
    print(f"==> Descongelando as últimas {unfreeze_layers} camadas e iniciando fine-tuning...")

    model.compile(optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    checkpoint = ModelCheckpoint(os.path.join(run_path, "model_finetuned.keras"), monitor="val_loss", save_best_only=True)
    history2 = model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS, callbacks=[checkpoint], verbose=1)

    # Carregar o melhor modelo do fine-tuning
    model.load_weights(os.path.join(run_path, "model_finetuned.keras"))
    print(f"==> Carregando o melhor modelo do fine-tuning...")
    # Avaliação
    y_true = df_test_pairs['pathology_binary'].astype(int).values
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)

    # Histórico completo
    full_history = {k: history1.history.get(k, []) + history2.history.get(k, []) for k in set(history1.history) | set(history2.history)}
    history1.history = full_history

    metrics = save_metrics(run_path, y_true, y_pred, history1)
    save_plot(run_path, history1)
    # Salvar resultado do trial em CSV
    result_row = {
        "trial": trial.number,
        "run_path": run_path,
        "backbone": backbone,
        "lr": lr,
        "fine_tune_lr": fine_tune_lr,
        "dense_units": dense_units,
        "dropout": dropout_rate,
        "use_l2": use_l2,
        "augment": use_augment,
        "unfreeze_layers": unfreeze_layers,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "use_dense_2": use_dense_2,
        "dense_units_2": dense_units//2 if use_dense_2 else "N/A",
        "attention_module": attention_module
    }
    csv_path = "resultados/todos_os_trials.csv"

    df_result = pd.DataFrame([result_row])

    # Garante que todas as colunas estejam presentes
    expected_columns = [
        "trial", "run_path", "backbone", "lr", "fine_tune_lr", "dense_units", "dropout",
        "use_l2", "augment", "unfreeze_layers", "accuracy", "precision", "recall", "f1_score",
        "use_dense_2", "dense_units_2", "attention_module"
    ]

    df_result = df_result.reindex(columns=expected_columns)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing.reindex(columns=expected_columns)
        df_all = pd.concat([df_existing, df_result], ignore_index=True)
        df_all.to_csv(csv_path, index=False)
    else:
        df_result.to_csv(csv_path, index=False)

    
    tf.keras.backend.clear_session()
    return metrics["accuracy"]

# ==== EXECUÇÃO ====
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, storage=STORAGE_PATH, load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)  # Teste com 3 trials inicialmente

    with open("resultados/resumo_study.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.params,
            "value": study.best_value
        }, f, indent=4)

    print("\nMelhor configuração encontrada:")
    print(study.best_trial.params)
    print(f"Melhor val_accuracy: {study.best_value:.4f}")