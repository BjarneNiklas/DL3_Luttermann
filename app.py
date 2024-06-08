import numpy as np
import plotly.graph_objs as go
import gradio as gr
from tensorflow import keras
from tensorflow.keras import layers

# Daten generieren
def generate_data(N, noise_variance, x_min, x_max):
    x_values = np.linspace(x_min, x_max, N)
    y_values = 0.5 * (x_values + 0.8) * (x_values + 1.8) * (x_values - 0.2) * (x_values - 0.3) * (x_values - 1.9) + 1
    noise = np.random.normal(0, np.sqrt(noise_variance), y_values.shape)
    y_noisy = y_values + noise
    return x_values, y_values, y_noisy

# Modell definieren
def build_model(hidden_layers, neurons_per_layer):
    model = keras.Sequential()
    model.add(layers.Dense(neurons_per_layer, activation='relu', input_shape=[1]))
    for _ in range(hidden_layers - 1):
        model.add(layers.Dense(neurons_per_layer, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# Daten aufteilen
def split_data(x_values, y_values, y_noisy):
    indices = np.arange(x_values.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[:N//2]
    test_indices = indices[N//2:]
    return (x_values[train_indices], y_values[train_indices], y_noisy[train_indices]), (x_values[test_indices], y_values[test_indices], y_noisy[test_indices])

# Plots erstellen
def create_plots(train_data, test_data, model, title):
    train_x, train_y, train_y_noisy = train_data
    test_x, test_y, test_y_noisy = test_data
    
    # Trainingsdaten
    train_trace_clean = go.Scatter(x=train_x, y=train_y, mode='markers', name='Train Clean', marker=dict(color='rgba(255, 0, 0, .8)'))
    train_trace_noisy = go.Scatter(x=train_x, y=train_y_noisy, mode='markers', name='Train Noisy', marker=dict(color='rgba(255, 165, 0, .8)'))
    
    # Testdaten
    test_trace_clean = go.Scatter(x=test_x, y=test_y, mode='markers', name='Test Clean', marker=dict(color='rgba(0, 255, 0, .8)'))
    test_trace_noisy = go.Scatter(x=test_x, y=test_y_noisy, mode='markers', name='Test Noisy', marker=dict(color='rgba(0, 0, 255, .8)'))
    
    # Vorhersagen
    x_range = np.linspace(train_x.min(), train_x.max(), 200)
    predictions = model.predict(x_range)
    prediction_trace = go.Scatter(x=x_range, y=predictions.flatten(), mode='lines', name='Prediction', line=dict(color='black'))
    
    # Plots zusammenfügen
    fig = go.Figure(data=[train_trace_clean, train_trace_noisy, test_trace_clean, test_trace_noisy, prediction_trace])
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    
    return fig

# Gradio Interface
def gradio_interface(model, train_data, test_data):
    def predict(x):
        return model.predict(np.array([x])).flatten()[0]
    
    demo = gr.Interface(fn=predict, inputs='number', outputs='number')
    return demo

# Hauptfunktion
def main(N, noise_variance, x_min, x_max, hidden_layers, neurons_per_layer, learning_rate, epochs, overfit_epochs, batch_size):
    # Daten generieren
    x, y, y_noisy = generate_data(N, noise_variance, x_min, x_max)
    
    # Daten aufteilen
    train_data, test_data = split_data(x, y, y_noisy)
    
    # Modelle erstellen und trainieren
    clean_model = build_model(hidden_layers, neurons_per_layer)
    clean_model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size, verbose=0)
    
    best_fit_model = build_model(hidden_layers, neurons_per_layer)
    best_fit_model.fit(train_data[0], train_data[2], epochs=epochs*2, batch_size=batch_size, verbose=0)
    
    overfit_model = build_model(hidden_layers, neurons_per_layer)
    overfit_model.fit(train_data[0], train_data[2], epochs=overfit_epochs, batch_size=batch_size, verbose=0)
    
    # Plots erstellen
    clean_fig = create_plots(train_data, test_data, clean_model, 'Clean Model Predictions')
    best_fit_fig = create_plots(train_data, test_data, best_fit_model, 'Best Fit Model Predictions')
    overfit_fig = create_plots(train_data, test_data, overfit_model, 'Overfit Model Predictions')
    
    clean_fig.show()
    best_fit_fig.show()
    overfit_fig.show()
    
    # Gradio Interface starten
    demo = gradio_interface(clean_model, train_data, test_data)
    demo.launch()

# Gradio Interface für Einstellungen
iface = gr.Interface(
    fn=main,
    inputs=[
        gr.Slider(50, 500, value=100, label="Data Points (N)"),
        gr.Slider(0.01, 1.0, value=0.05, label="Noise Variance (V)"),
        gr.Slider(-5, 5, value=-2, step=0.1, label="X Min"),
        gr.Slider(0, 10, value=2, step=0.1, label="X Max"),
        gr.Slider(1, 10, value=2, label="Hidden Layers"),
        gr.Slider(10, 200, value=100, label="Neurons per Layer"),
        gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Learning Rate"),
        gr.Slider(10, 1000, value=100, label="Epochs"),
        gr.Slider(10, 1000, value=500, label="Overfit Epochs"),
        gr.Slider(1, 128, value=32, label="Batch Size")
    ],
    outputs=[]
)
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
