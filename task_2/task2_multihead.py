import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from functools import partial
import time


def common_acc_class(y_true, y_pred):
    pred = tf.math.argmax(y_pred, axis=1)
    n = len(pred)
    y_true = tf.cast(y_true, dtype=tf.int64)
    y_true = tf.reshape(y_true, [n])
    return tf.math.minimum(tf.abs(pred - y_true), tf.abs(12 - tf.abs(pred - y_true)))

def custom_acc_reg(y_true,y_pred):
  
  acc = tf.math.minimum(tf.abs(y_pred-y_true),tf.abs(60 - tf.abs(y_pred-y_true)))

  return acc


def load_data():
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')

    hours = labels[:,0] 
    minutes = labels[:,1]

    shuffler = np.random.permutation(len(images))

    images = images[shuffler]
    hours = hours[shuffler]
    minutes = minutes[shuffler]

    X_train = images[0:int(0.65*len(images))]
    y_train_h = hours[0:int(0.65*len(images))]
    y_train_m = minutes[0:int(0.65*len(images))]

    X_valid = images[int(0.65*len(images)):int(0.8*len(images))]
    y_valid_h = hours[int(0.65*len(images)):int(0.8*len(images))]
    y_valid_m = minutes[int(0.65*len(images)):int(0.8*len(images))]

    X_test = images[int(0.8*len(images)):]
    y_test_h = hours[int(0.8*len(images)):]
    y_test_m = minutes[int(0.8*len(images)):]

    return X_train, y_train_h, y_train_m, X_valid, y_valid_h, y_valid_m, X_test, y_test_h, y_test_m


def train_network():
    DefaultConv2D = partial(keras.layers.Conv2D,
                            kernel_size=3, strides = 1, activation='relu', padding="SAME")

    input = keras.layers.Input(shape=(150, 150, 1))
    conv64 = DefaultConv2D(filters=64, kernel_size=7, strides=3)(input)
    maxpool64 = keras.layers.MaxPooling2D(2)(conv64)
    drop64= layers.Dropout(0.2)(maxpool64)
    batchnormal64 = layers.BatchNormalization()(drop64)
    conv128 = DefaultConv2D(filters=128, kernel_size=5, strides=2)(batchnormal64)
    conv128 = DefaultConv2D(filters=128, kernel_size=5, strides=2)(conv128)
    maxpool128 = keras.layers.MaxPooling2D(2)(conv128)
    drop128= layers.Dropout(0.2)(maxpool128)
    batchnormal128 = layers.BatchNormalization()(drop128)
    conv256 = DefaultConv2D(filters=256)(batchnormal128)
    conv256 = DefaultConv2D(filters=256)(conv256)
    maxpool256 = keras.layers.MaxPooling2D(2)(conv256)
    drop256 = layers.Dropout(0.1)(maxpool256)
    batchnormal256 = layers.BatchNormalization()(drop256)
    flat = layers.Flatten()(batchnormal256)

    reg_layer_1 = layers.Dense(128, activation='relu')(flat)
    reg_layer_1_drop = layers.Dropout(0.5)(reg_layer_1)
    reg_layer_2 = layers.Dense(64, activation='relu')(reg_layer_1)
    reg_layer_2_drop = layers.Dropout(0.5)(reg_layer_2)
    reg_layer_3 = layers.Dense(16, activation='relu')(reg_layer_2)
    reg_layer_3_drop = layers.Dropout(0.2)(reg_layer_3)
    reg_output = layers.Dense(1, activation='linear', name='reg_output')(reg_layer_3)

    class_layer_1 = layers.Dense(128, activation='relu')(flat)
    class_layer_1_drop = layers.Dropout(0.5)(class_layer_1)
    class_layer_2 = layers.Dense(64, activation='relu')(class_layer_1)
    class_layer_2_drop = layers.Dropout(0.3)(class_layer_2)
    class_output = layers.Dense(12, activation='softmax', name='class_output')(class_layer_2)

    model = keras.models.Model(inputs=input, outputs=[class_output, reg_output])

    model.compile(loss=['sparse_categorical_crossentropy', 'mae'],
        loss_weights=[0.83, 0.17], 
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics={'class_output':common_acc_class, 'reg_output': custom_acc_reg})

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")

    tensorboard_cb = callbacks.TensorBoard('logs/multi/' + run_id)

    X_train, y_train_h, y_train_m, X_valid, y_valid_h, y_valid_m, X_test, y_test_h, y_test_m = load_data()

    history = model.fit(X_train, [y_train_h, y_train_m],
                    batch_size=36,
                    epochs=20,
                    shuffle=True,
                    validation_data=(X_valid, [y_valid_h, y_valid_m]),
                    callbacks=[tensorboard_cb])

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("results/multi/" + run_id + ".csv")


if __name__ == '__main__':
    train_network()

