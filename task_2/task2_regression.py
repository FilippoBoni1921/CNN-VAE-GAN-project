import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from functools import partial
import time

def load_data():
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')

    regression_labels = labels[:, 0] + labels[:, 1] / 60

    shuffler = np.random.permutation(len(images))
    images = tf.convert_to_tensor(tf.reshape(images[shuffler], (18000, 150, 150, 1)))
    regression_labels = regression_labels[shuffler]

    cutoff = int(0.8 * len(images))
    X_train_full = images[:cutoff]
    y_train_full = regression_labels[:cutoff]
    X_test = images[cutoff:]
    y_test = regression_labels[cutoff:]

    cutoff = int(0.8 * len(X_train_full))
    X_train = X_train_full[:cutoff]
    y_train = y_train_full[:cutoff]
    X_valid = X_train_full[cutoff:]
    y_valid = y_train_full[cutoff:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def common_sense_error(y_true, y_pred):
    return tf.math.minimum(tf.abs(y_pred - y_true), tf.abs(12 - tf.abs(y_pred - y_true))) * 60


def train_network():
    DefaultConv2D = partial(keras.layers.Conv2D,
                            kernel_size=3, strides=1, activation='relu', padding="SAME")

    model = models.Sequential([
        DefaultConv2D(filters=64, kernel_size=7, strides=3, input_shape=[150, 150, 1]),
        keras.layers.MaxPooling2D(2),
        layers.BatchNormalization(),
        DefaultConv2D(filters=128, kernel_size=5, strides=2),
        DefaultConv2D(filters=128, kernel_size=5, strides=2),
        keras.layers.MaxPooling2D(2),
        layers.BatchNormalization(),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mae',
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=common_sense_error)

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")

    tensorboard_cb = callbacks.TensorBoard('logs/regression/' + run_id)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=20,
                        shuffle=True,
                        validation_data=(X_valid, y_valid),
                        callbacks=[tensorboard_cb])

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("results/regression/" + run_id + ".csv")


if __name__ == '__main__':
    train_network()

