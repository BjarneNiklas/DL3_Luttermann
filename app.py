""" import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

# Funktion zur Datengenerierung
def generate_data(N):
    x_vals = np.random.uniform(-2, 2, N)
    y_vals = 0.5 * (x_vals + 0.8) * (x_vals + 1.8) * (x_vals - 0.2) * (x_vals - 0.3) * (x_vals - 1.9) + 1
    return x_vals, y_vals

# Gaussian Noise hinzufügen
def add_noise(y_vals, variance):
    noise = np.random.normal(0, np.sqrt(variance), len(y_vals))
    return y_vals + noise

# Daten generieren und aufteilen
N = 100
x_vals, y_vals = generate_data(N)
train_x, train_y = x_vals[:N//2], y_vals[:N//2]
test_x, test_y = x_vals[N//2:], y_vals[N//2:]

# Daten mit Rauschen
train_y_noisy = add_noise(train_y, 0.05)
test_y_noisy = add_noise(test_y, 0.05)

# Funktion zur Erstellung des Modells
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=[1]),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')
    return model

# Modell ohne Rauschen trainieren
def train_clean_model():
    model = create_model()
    model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=0)
    train_loss = model.evaluate(train_x, train_y, verbose=0)
    test_loss = model.evaluate(test_x, test_y, verbose=0)
    return model, train_loss, test_loss

# Modell mit Rauschen trainieren (Best-Fit)
def train_best_fit_model():
    model = create_model()
    model.fit(train_x, train_y_noisy, epochs=200, batch_size=32, verbose=0)
    train_loss = model.evaluate(train_x, train_y_noisy, verbose=0)
    test_loss = model.evaluate(test_x, test_y_noisy, verbose=0)
    return model, train_loss, test_loss

# Modell mit Rauschen trainieren (Overfit)
def train_overfit_model():
    model = create_model()
    model.fit(train_x, train_y_noisy, epochs=1000, batch_size=32, verbose=0)
    train_loss = model.evaluate(train_x, train_y_noisy, verbose=0)
    test_loss = model.evaluate(test_x, test_y_noisy, verbose=0)
    return model, train_loss, test_loss

# Vorhersagen plotten
def plot_predictions(model, x, y, title):
    plt.figure()
    plt.scatter(x, y, label='Data')
    preds = model.predict(x)
    plt.plot(x, preds, color='red', label='Model Predictions')
    plt.title(title)
    plt.legend()
    plt.show()

# Gradio Interface
def visualize_models():
    # Modelle trainieren
    clean_model, clean_train_loss, clean_test_loss = train_clean_model()
    best_model, best_train_loss, best_test_loss = train_best_fit_model()
    overfit_model, overfit_train_loss, overfit_test_loss = train_overfit_model()

    # Plots generieren
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plot_predictions(clean_model, train_x, train_y, f'Clean Model (Train Loss: {clean_train_loss:.4f}, Test Loss: {clean_test_loss:.4f})')

    plt.subplot(2, 2, 2)
    plot_predictions(clean_model, test_x, test_y, f'Clean Model (Train Loss: {clean_train_loss:.4f}, Test Loss: {clean_test_loss:.4f})')

    plt.subplot(2, 2, 3)
    plot_predictions(best_model, train_x, train_y_noisy, f'Best Fit Model (Train Loss: {best_train_loss:.4f}, Test Loss: {best_test_loss:.4f})')

    plt.subplot(2, 2, 4)
    plot_predictions(overfit_model, test_x, test_y_noisy, f'Overfit Model (Train Loss: {overfit_train_loss:.4f}, Test Loss: {overfit_test_loss:.4f})')

    plt.tight_layout()
    plt.show()

# Gradio App
gr.Interface(
    fn=visualize_models, 
    inputs=[], 
    outputs="plot",
    title="FFNN Regression with TensorFlow.js",
    description="Visualize the training and test results of different FFNN models."
).launch()"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objects import Scatter, Line

# Funktion y(x)
def y(x):
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Datenerzeugung
N = 100
x = np.random.uniform(-2, 2, N)
y = y(x)

train_x = x[:50]
train_y = y[:50]

test_x = x[50:]
test_y = y[50:]

noise = np.random.normal(0, 0.05, N)

train_y_noisy = train_y + noise
test_y_noisy = test_y + noise

# Modellarchitektur
model = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])

# Modellkompilierung
model.compile(loss='mse', optimizer='adam', learning_rate=0.01)

# Training des Modells
model.fit(train_x[:, np.newaxis], train_y_noisy, epochs=100, batch_size=32)

# Evaluation und Visualisierung
train_loss = model.evaluate(train_x[:, np.newaxis], train_y_noisy)
test_loss = model.evaluate(test_x[:, np.newaxis], test_y_noisy)

print('Train Loss:', train_loss)
print('Test Loss:', test_loss)

# Plotten der Daten, des Loss und der Vorhersagen
plt.scatter(x, y, label='Original Data')
plt.scatter(x, y_noisy, label='Noisy Data')

y_unnoisy_pred = model.predict(x[:, np.newaxis])
plt.plot(x, y_unnoisy_pred, label='Un-Noisy Prediction')

y_best_pred = model.predict(train_x[:, np.newaxis])
plt.plot(train_x, y_best_pred, label='Best Model Prediction')

y_overfit_pred = model.predict(train_x[:, np.newaxis])
plt.plot(train_x, y_overfit_pred, label='Overfit Model Prediction')

plt.legend()
plt.show()

# Diskussion
# Hier die Ergebnisse und Beobachtungen diskutieren, insbesondere das Phänomen des Overfittings.

# Dokumentation
# Code, Bibliotheken und Ergebnisse dokumentieren.

