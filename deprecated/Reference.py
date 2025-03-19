import pandas as pd
import numpy as np

# Wczytywanie danych
X_train = pd.read_csv('train/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv('train/y_train.txt', delim_whitespace=True, header=None)
X_test = pd.read_csv('test/X_test.txt', delim_whitespace=True, header=None)
y_test = pd.read_csv('test/y_test.txt', delim_whitespace=True, header=None)

# Wczytanie nazw kolumn
features = pd.read_csv('features.txt', delim_whitespace=True, header=None)
X_train.columns = features[1]
X_test.columns = features[1]

# Wczytanie etykiet aktywności
activity_labels = pd.read_csv('activity_labels.txt', delim_whitespace=True, header=None, index_col=0)
print(X_train.shape)  # (7352, 561) - 7352 przykładów, 561 cech
print(y_train.shape)  # (7352, 1) - 7352 etykiet aktywności

print(activity_labels)  # Wyświetlenie nazw aktywności
# Połączenie zbiorów
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
from tensorflow.keras import to_categorical

# One-hot encoding etykiet
y = y - 1  # Etykiety zaczynają się od 1, przesuwamy do 0
y_categorical = to_categorical(y, num_classes=6)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Tworzenie modelu
model = Sequential([
    Dense(256, activation='relu', input_shape=(561,)),  # 561 cech
    Dropout(0.5),  # Regularyzacja Dropout
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')  # Wyjście dla 6 klas
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Wyświetlenie architektury modelu
model.summary()
# Trenowanie sieci
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val, y_val))
# Ewaluacja modelu
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f"Dokładność na zbiorze walidacyjnym: {test_accuracy:.2f}")
import matplotlib.pyplot as plt

# Wizualizacja dokładności
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

# Wizualizacja funkcji straty
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
import tensorflow as tf

# Lista dostępnych GPU
print("GPU dostępne:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Ustaw użycie pierwszego GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')

        # Ogranicz pamięć GPU (opcjonalnie)
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from kerastuner.tuners import RandomSearch

# Funkcja budująca model z hiperparametrami
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(6, activation='softmax'))  # 6 klas wyjściowych
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [0.001, 0.0001])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# RandomSearch z KerasTuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Ile konfiguracji przetestować
    executions_per_trial=1,  # Ile razy trenować każdą konfigurację
    directory='tuner_logs',  # Gdzie zapisać wyniki
    project_name='human_activity_recognition'
)

# Przeszukiwanie hiperparametrów
tuner.search(X_train, y_categorical, validation_split=0.2, epochs=10, batch_size=32)

# Pobierz najlepszy model
best_hyperparams = tuner.get_best_hyperparameters(1)[0]
print(best_hyperparams.values)
import multiprocessing as mp

# Funkcja testująca jeden model
def train_and_evaluate_model(params):
    model = Sequential([
        Dense(params['units'], activation='relu', input_shape=(561,)),
        Dropout(params['dropout']),
        Dense(6, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_categorical, epochs=params['epochs'], batch_size=32, verbose=0)
    return max(history.history['accuracy'])

# Lista konfiguracji do przetestowania
param_grid = [
    {'units': 64, 'dropout': 0.2, 'learning_rate': 0.001, 'epochs': 10},
    {'units': 128, 'dropout': 0.3, 'learning_rate': 0.0001, 'epochs': 15},
    {'units': 32, 'dropout': 0.1, 'learning_rate': 0.01, 'epochs': 5}
]

# Zrównoleglenie
with mp.Pool(processes=3) as pool:
    results = pool.map(train_and_evaluate_model, param_grid)

print("Wyniki:", results)
import tensorflow as tf

# Lista dostępnych GPU
gpus = tf.config.list_physical_devices('GPU')

# Przypisz różne modele do różnych GPU
for i, params in enumerate(param_grid):
    with tf.device(f'/GPU:{i % len(gpus)}'):
        train_and_evaluate_model(params)
