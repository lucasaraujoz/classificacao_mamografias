
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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