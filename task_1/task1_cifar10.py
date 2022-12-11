from tensorflow import keras
from keras import models, layers
import numpy as np
import pickle

cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255

# MLP1
results = {
    'loss': np.array([]).reshape(0, 50),
    'accuracy': np.array([]).reshape(0, 50),
    'val_loss': np.array([]).reshape(0, 50),
    'val_accuracy': np.array([]).reshape(0, 50)
}
for _ in range(5):
    mlp1 = models.Sequential([
        layers.Flatten(input_shape=[32, 32, 3]),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    mlp1.compile(loss='sparse_categorical_crossentropy',
                 optimizer=keras.optimizers.SGD(learning_rate=0.01),
                 metrics=['accuracy'])

    history = mlp1.fit(X_train, y_train,
                        epochs=50,
                        validation_data=(X_valid, y_valid))

    for metric, res in history.history.items():
        results[metric] = np.vstack([results[metric], res])

end_result = {}
for metric, res in results.items():
    means = res.mean(axis=0)
    stddev = res.std(axis=0)

    end_result[f"{metric}_mean"] = means
    end_result[f"{metric}_std"] = stddev

with open(f"mlp1.pickle", 'wb') as f:
    pickle.dump(end_result, f, protocol=pickle.HIGHEST_PROTOCOL)

# MLP2
results = {
    'loss': np.array([]).reshape(0, 50),
    'accuracy': np.array([]).reshape(0, 50),
    'val_loss': np.array([]).reshape(0, 50),
    'val_accuracy': np.array([]).reshape(0, 50)
}
for _ in range(5):
    mlp2 = models.Sequential([
        layers.Flatten(input_shape=[32, 32, 3]),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    mlp2.compile(loss='sparse_categorical_crossentropy',
                 optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
                 metrics=['accuracy'])

    history = mlp2.fit(X_train, y_train,
                        epochs=50,
                        validation_data=(X_valid, y_valid))

    for metric, res in history.history.items():
        results[metric] = np.vstack([results[metric], res])

end_result = {}
for metric, res in results.items():
    means = res.mean(axis=0)
    stddev = res.std(axis=0)

    end_result[f"{metric}_mean"] = means
    end_result[f"{metric}_std"] = stddev

with open(f"mlp2.pickle", 'wb') as f:
    pickle.dump(end_result, f, protocol=pickle.HIGHEST_PROTOCOL)

# CNN1
results = {
    'loss': np.array([]).reshape(0, 50),
    'accuracy': np.array([]).reshape(0, 50),
    'val_loss': np.array([]).reshape(0, 50),
    'val_accuracy': np.array([]).reshape(0, 50)
}
for _ in range(5):
    cnn1 = models.Sequential([
        layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=(32, 32, 3),
                      kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform'),
        layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform'),
        layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='tanh'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(64, activation='tanh'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])

    cnn1.compile(loss='sparse_categorical_crossentropy',
                 optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 metrics=['accuracy'])

    history = cnn1.fit(X_train, y_train,
                       epochs=20,
                       validation_data=(X_valid, y_valid))

    for metric, res in history.history.items():
        results[metric] = np.vstack([results[metric], res])

end_result = {}
for metric, res in results.items():
    means = res.mean(axis=0)
    stddev = res.std(axis=0)

    end_result[f"{metric}_mean"] = means
    end_result[f"{metric}_std"] = stddev

with open(f"cifar_cnn1.pickle", 'wb') as f:
    pickle.dump(end_result, f, protocol=pickle.HIGHEST_PROTOCOL)