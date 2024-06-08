import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gradio as gr

# Funktion zur Datengenerierung
def generate_data(N):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    return x, y

# Funktion zum Verrauschen der Daten
def add_noise(y, variance):
    noise = np.random.normal(0, np.sqrt(variance), y.shape)
    return y + noise

# Daten generieren
N = 100
x, y = generate_data(N)
x_train, x_test = np.split(x, 2)
y_train, y_test = np.split(y, 2)

# Daten verrauschen
noise_variance = 0.05
y_train_noisy = add_noise(y_train, noise_variance)
y_test_noisy = add_noise(y_test, noise_variance)

### 2. Modelltraining

# Funktion zur Erstellung des Modells
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')
    return model

# Funktion zum Trainieren des Modells
def train_model(model, x_train, y_train, epochs):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return history.history['loss']

# Modelle trainieren
model_clean = create_model()
loss_clean = train_model(model_clean, x_train, y_train, 100)

model_best_fit = create_model()
loss_best_fit = train_model(model_best_fit, x_train, y_train_noisy, 100)

model_overfit = create_model()
loss_overfit = train_model(model_overfit, x_train, y_train_noisy, 200)

# Vorhersagen generieren
y_pred_clean_train = model_clean.predict(x_train).flatten()
y_pred_clean_test = model_clean.predict(x_test).flatten()

y_pred_best_fit_train = model_best_fit.predict(x_train).flatten()
y_pred_best_fit_test = model_best_fit.predict(x_test).flatten()

y_pred_overfit_train = model_overfit.predict(x_train).flatten()
y_pred_overfit_test = model_overfit.predict(x_test).flatten()

### 3. Visualisierung und Gradio-Interface

# Funktion zum Plotten der Daten
def plot_data():
    fig, axs = plt.subplots(5, 2, figsize=(12, 20))

    # R1: Datensätze
    axs[0, 0].scatter(x_train, y_train, color='blue', label='Train (clean)')
    axs[0, 0].scatter(x_test, y_test, color='red', label='Test (clean)')
    axs[0, 0].legend()
    axs[0, 0].set_title("Datensätze ohne Rauschen")

    axs[0, 1].scatter(x_train, y_train_noisy, color='blue', label='Train (noisy)')
    axs[0, 1].scatter(x_test, y_test_noisy, color='red', label='Test (noisy)')
    axs[0, 1].legend()
    axs[0, 1].set_title("Datensätze mit Rauschen")

    # R2: Vorhersage des Modells ohne Rauschen
    axs[1, 0].scatter(x_train, y_train, color='blue', label='Train (clean)')
    axs[1, 0].plot(x_train, y_pred_clean_train, color='green', label='Prediction')
    axs[1, 0].legend()
    axs[1, 0].set_title(f"Train Loss: {loss_clean[-1]:.4f}")

    axs[1, 1].scatter(x_test, y_test, color='red', label='Test (clean)')
    axs[1, 1].plot(x_test, y_pred_clean_test, color='green', label='Prediction')
    axs[1, 1].legend()
    axs[1, 1].set_title(f"Test Loss: {tf.keras.losses.mean_squared_error(y_test, y_pred_clean_test).numpy():.4f}")

    # R3: Vorhersage des Best-Fit-Modells
    axs[2, 0].scatter(x_train, y_train_noisy, color='blue', label='Train (noisy)')
    axs[2, 0].plot(x_train, y_pred_best_fit_train, color='green', label='Prediction')
    axs[2, 0].legend()
    axs[2, 0].set_title(f"Train Loss: {loss_best_fit[-1]:.4f}")

    axs[2, 1].scatter(x_test, y_test_noisy, color='red', label='Test (noisy)')
    axs[2, 1].plot(x_test, y_pred_best_fit_test, color='green', label='Prediction')
    axs[2, 1].legend()
    axs[2, 1].set_title(f"Test Loss: {tf.keras.losses.mean_squared_error(y_test_noisy, y_pred_best_fit_test).numpy():.4f}")

    # R4: Vorhersage des Overfit-Modells
    axs[3, 0].scatter(x_train, y_train_noisy, color='blue', label='Train (noisy)')
    axs[3, 0].plot(x_train, y_pred_overfit_train, color='green', label='Prediction')
    axs[3, 0].legend()
    axs[3, 0].set_title(f"Train Loss: {loss_overfit[-1]:.4f}")

    axs[3, 1].scatter(x_test, y_test_noisy, color='red', label='Test (noisy)')
    axs[3, 1].plot(x_test, y_pred_overfit_test, color='green', label='Prediction')
    axs[3, 1].legend()
    axs[3, 1].set_title(f"Test Loss: {tf.keras.losses.mean_squared_error(y_test_noisy, y_pred_overfit_test).numpy():.4f}")

    # Display the plots
    plt.tight_layout()
    plt.show()

# Gradio Interface
iface = gr.Interface(fn=plot_data, inputs=[], outputs="plot")
iface.launch()





"""import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.subplots as sp
import gradio as gr

# Generierung der Daten
def generate_data(N=100):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    return x, y

# Hinzufügen von Rauschen
def add_noise(y, variance=0.05):
    noise = np.random.normal(0, np.sqrt(variance), y.shape)
    return y + noise

# Daten generieren
x, y = generate_data()
x_train, x_test = x[:50], x[50:]
y_train, y_test = y[:50], y[50:]

# Daten mit Rauschen
y_train_noisy = add_noise(y_train)
y_test_noisy = add_noise(y_test)

# Modell definieren und trainieren
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Training des Modells ohne Rauschen
model_unverrauscht = create_model()
model_unverrauscht.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Best-Fit)
model_best_fit = create_model()
model_best_fit.fit(x_train, y_train_noisy, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Over-Fit)
model_over_fit = create_model()
model_over_fit.fit(x_train, y_train_noisy, epochs=1000, batch_size=32, verbose=0)

# Visualisierung der Ergebnisse
def plot_results():
    # Originalfunktion
    x_line = np.linspace(-2, 2, 100)
    y_line = 0.5 * (x_line + 0.8) * (x_line + 1.8) * (x_line - 0.2) * (x_line - 0.3) * (x_line - 1.9) + 1

    # Vorhersagen
    y_train_pred_unverrauscht = model_unverrauscht.predict(x_train).flatten()
    y_test_pred_unverrauscht = model_unverrauscht.predict(x_test).flatten()

    y_train_pred_best_fit = model_best_fit.predict(x_train).flatten()
    y_test_pred_best_fit = model_best_fit.predict(x_test).flatten()

    y_train_pred_over_fit = model_over_fit.predict(x_train).flatten()
    y_test_pred_over_fit = model_over_fit.predict(x_test).flatten()

    # Plotly Visualisierung
    fig = sp.make_subplots(rows=4, cols=2, subplot_titles=(
        'Train Data (No Noise)', 'Train Data (With Noise)', 
        'Test Data (No Noise)', 'Test Data (With Noise)',
        'Train Predictions (Unverrauscht)', 'Train Predictions (With Noise)',
        'Test Predictions (Unverrauscht)', 'Test Predictions (With Noise)'
    ))

    # R1: Die Datensätze
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data (No Noise)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data (No Noise)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Train Data (With Noise)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Test Data (With Noise)'), row=2, col=2)

    # R2: Die Vorhersage des Modells, das ohne Rauschen trainiert wurde
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_unverrauscht, mode='markers', name='Train Predictions (Unverrauscht)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_unverrauscht, mode='markers', name='Test Predictions (Unverrauscht)'), row=4, col=1)

    # R3: Die Vorhersage Ihres besten Modells
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_best_fit, mode='markers', name='Train Predictions (Best-Fit)'), row=3, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_best_fit, mode='markers', name='Test Predictions (Best-Fit)'), row=4, col=2)

    # R4: Die Vorhersage Ihres Overfit-Modells
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_over_fit, mode='markers', name='Train Predictions (Over-Fit)'), row=3, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_over_fit, mode='markers', name='Test Predictions (Over-Fit)'), row=4, col=2)

    fig.update_layout(height=1000, width=1200, title_text="Regression mit FFNN und TensorFlow.js")
    
    return fig

# Gradio Interface
def interactive_plot():
    fig = plot_results()
    return fig

interface = gr.Interface(fn=interactive_plot, inputs=[], outputs=gr.Plot())

interface.launch()"""



"""import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import gradio as gr

# Generierung der Daten
def generate_data(N=100):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    return x, y

# Hinzufügen von Rauschen
def add_noise(y, variance=0.05):
    noise = np.random.normal(0, np.sqrt(variance), y.shape)
    return y + noise

# Daten generieren
x, y = generate_data()
x_train, x_test = x[:50], x[50:]
y_train, y_test = y[:50], y[50:]

# Daten mit Rauschen
y_train_noisy = add_noise(y_train)
y_test_noisy = add_noise(y_test)

# Modell definieren und trainieren
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Training des Modells ohne Rauschen
model_unverrauscht = create_model()
model_unverrauscht.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Best-Fit)
model_best_fit = create_model()
model_best_fit.fit(x_train, y_train_noisy, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Over-Fit)
model_over_fit = create_model()
model_over_fit.fit(x_train, y_train_noisy, epochs=1000, batch_size=32, verbose=0)

# Visualisierung der Ergebnisse
def plot_results(x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy, model_unverrauscht, model_best_fit, model_over_fit):
    # Originalfunktion
    x_line = np.linspace(-2, 2, 100)
    y_line = 0.5 * (x_line + 0.8) * (x_line + 1.8) * (x_line - 0.2) * (x_line - 0.3) * (x_line - 1.9) + 1

    # Vorhersagen
    y_train_pred_unverrauscht = model_unverrauscht.predict(x_train).flatten()
    y_test_pred_unverrauscht = model_unverrauscht.predict(x_test).flatten()

    y_train_pred_best_fit = model_best_fit.predict(x_train).flatten()
    y_test_pred_best_fit = model_best_fit.predict(x_test).flatten()

    y_train_pred_over_fit = model_over_fit.predict(x_train).flatten()
    y_test_pred_over_fit = model_over_fit.predict(x_test).flatten()

    # Plotly Visualisierung
    fig = go.Figure()

    # Originalfunktion plot
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Original Function'))

    # Noisy training data plot
    fig.add_trace(go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Noisy Train Data'))

    # Noisy test data plot
    fig.add_trace(go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Noisy Test Data'))

    # Train predictions plot
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_unverrauscht, mode='markers', name='Train Predictions (Unverrauscht)'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_unverrauscht, mode='markers', name='Test Predictions (Unverrauscht)'))

    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_best_fit, mode='markers', name='Train Predictions (Best-Fit)'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_best_fit, mode='markers', name='Test Predictions (Best-Fit)'))

    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_over_fit, mode='markers', name='Train Predictions (Over-Fit)'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_over_fit, mode='markers', name='Test Predictions (Over-Fit)'))

    fig.update_layout(title='Regression mit FFNN und TensorFlow.js',
                      xaxis_title='x',
                      yaxis_title='y',
                      legend_title='Legend')
    
    return fig

# Gradio Interface
def interactive_plot():
    fig = plot_results(x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy, model_unverrauscht, model_best_fit, model_over_fit)
    return fig

interface = gr.Interface(fn=interactive_plot, inputs=[], outputs=gr.Plot())

interface.launch()"""
