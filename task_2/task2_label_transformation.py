import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from functools import partial
import pandas as pd
import time


def load_data():
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')

    # convert labels to angle on the unit circle in radians
    rad_labels = np.array([(h * (np.math.pi / 6), m * np.math.pi / 30) for h, m in labels])

    trig_labels = np.array([(np.math.sin(label[0]),
                             np.math.cos(label[0]),
                             np.math.sin(label[1]),
                             np.math.cos(label[1]))
                            for label in rad_labels])

    shuffler = np.random.permutation(len(images))
    images = tf.convert_to_tensor(tf.reshape(images[shuffler], (18000, 150, 150, 1)))
    trig_labels = trig_labels[shuffler]

    cutoff = int(0.8 * len(images))
    X_train_full = images[:cutoff]
    y_train_full = trig_labels[:cutoff]
    X_test = images[cutoff:]
    y_test = trig_labels[cutoff:]

    cutoff = int(0.8 * len(X_train_full))
    X_train = X_train_full[:cutoff]
    y_train = y_train_full[:cutoff]
    X_valid = X_train_full[cutoff:]
    y_valid = y_train_full[cutoff:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def to_angle(sin_values, cos_values):
    angles = tf.math.atan(sin_values / cos_values)
    addition = tf.map_fn(lambda x: np.math.pi if x < 0 else float(0), cos_values)

    return (angles + addition) % (2 * np.math.pi)


def common_sense_error(y_true, y_pred):
    hours_true = to_angle(y_true[:, 0], y_true[:, 1]) * (6 / np.math.pi)
    hours_pred = to_angle(y_pred[:, 0], y_pred[:, 1]) * (6 / np.math.pi)

    minutes_true = to_angle(y_true[:, 2], y_true[:, 3]) * (30 / np.math.pi)
    minutes_pred = to_angle(y_pred[:, 2], y_pred[:, 3]) * (30 / np.math.pi)

    total_true = hours_true * 60 + minutes_true
    total_pred = hours_pred * 60 + minutes_pred

    return tf.math.minimum(tf.abs(total_pred - total_true), tf.abs(720 - tf.abs(total_pred - total_true)))


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
        layers.Dense(128, activation='tanh'),
        #layers.Dropout(0.1),
        layers.Dense(64, activation='tanh'),
        #layers.Dropout(0.1),
        layers.Dense(32, activation='tanh'),
        #layers.Dropout(0.1),
        layers.Dense(4, activation='tanh')
    ])

    model.compile(loss='mae',
                  optimizer=keras.optimizers.Adam(learning_rate=0.002),
                  metrics=common_sense_error)

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")


    tensorboard_cb = callbacks.TensorBoard('logs/angle/' + run_id)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=20,
                        shuffle=True,
                        validation_data=(X_valid, y_valid),
                        callbacks=[tensorboard_cb])

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("results/angle/" + run_id + ".csv")


if __name__ == '__main__':
    train_network()
