from tensorflow import keras
from keras import models, layers
import numpy as np
import pickle


def make_model(nodes=None,
               dropout=0.2,
               activation='relu',
               bn=False,
               init="glorot_uniform",
               regularization=None,
               optimizer=keras.optimizers.SGD,
               lr=0.01):
    if not nodes:
        nodes = [128, 128]
    model = models.Sequential()
    model.add(layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=(28, 28, 1), kernel_initializer=init, kernel_regularizer=regularization))
    if bn:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=init, kernel_regularizer=regularization))
    model.add(layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=init, kernel_regularizer=regularization))
    if bn:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=init, kernel_regularizer=regularization))
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=init, kernel_regularizer=regularization))
    if bn:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())

    for node in nodes:
        model.add(layers.Dense(node, activation=activation, kernel_initializer=init, kernel_regularizer=regularization))
        if dropout:
            model.add(layers.Dropout(dropout))
        if bn:
            model.add(layers.BatchNormalization())

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer(learning_rate=lr),
                  metrics=['accuracy'])

    return model


def run_experiment(name, params, n_runs=5, epochs=20):
    results = {
        'loss': np.array([]).reshape(0, epochs),
        'accuracy': np.array([]).reshape(0, epochs),
        'val_loss': np.array([]).reshape(0, epochs),
        'val_accuracy': np.array([]).reshape(0, epochs)
    }
    for _ in range(n_runs):
        model = make_model(**params)
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_data=(X_valid, y_valid))

        for metric, res in history.history.items():
            results[metric] = np.vstack([results[metric], res])

    end_result = {}
    for metric, res in results.items():
        means = res.mean(axis=0)
        stddev = res.std(axis=0)

        end_result[f"{metric}_mean"] = means
        end_result[f"{metric}_std"] = stddev

    with open(f"cnn_result/{name}.pickle", 'wb') as f:
        pickle.dump(end_result, f, protocol=pickle.HIGHEST_PROTOCOL)


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255

configs = {
    'default':
        {'nodes': [128, 64],
        'dropout': 0.2,
        'activation': 'relu',
        'bn': False,
        'init': "glorot_uniform",
        'regularization': None,
        'optimizer': keras.optimizers.SGD,
        'lr': 0.01}
}

for name, params in configs.items():
    run_experiment(name, params)
