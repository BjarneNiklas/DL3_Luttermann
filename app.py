import gradio as gr
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go

# Funktion zur Datengenerierung
def generate_data(N=100, V=0.05, X_min=-2, X_max=2):
    x_values = np.sort(np.random.uniform(X_min, X_max, N))
    y_values = 0.5 * (x_values + 0.8) * (x_values + 1.8) * (x_values - 0.2) * (x_values - 0.3) * (x_values - 1.9) + 1
    y_values_noisy = y_values + np.random.normal(0, np.sqrt(V), N)
    
    # Zufällige, gleichmäßige Aufteilung der Daten in Trainings- und Testdaten
    indices = np.random.permutation(N)
    split_index = N // 2
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    
    x_train, y_train = x_values[train_indices], y_values[train_indices]
    x_test, y_test = x_values[test_indices], y_values[test_indices]
    y_noisy_train, y_noisy_test = y_values_noisy[train_indices], y_values_noisy[test_indices]
    
    return (x_train, y_train), (x_test, y_test), (x_train, y_noisy_train), (x_test, y_noisy_test)

# Funktion zur Erstellung und Training des Modells
def create_and_train_model(train_data, hidden_layers, neurons_per_layer, learning_rate, epochs, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(1,)))
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    x_train, y_train = train_data
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model

# Funktion zur Vorhersage und Plot-Generierung
def plot_predictions(x_train, y_train, x_test, y_test, model, title):
    y_pred_train = model.predict(x_train).flatten()
    y_pred_test = model.predict(x_test).flatten()
    
    trace_train_data = go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data', marker=dict(color='blue'))
    trace_test_data = go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data', marker=dict(color='red'))
    trace_train_pred = go.Scatter(x=x_train, y=y_pred_train, mode='lines', name='Train Prediction', line=dict(color='blue', dash='dash'))
    trace_test_pred = go.Scatter(x=x_test, y=y_pred_test, mode='lines', name='Test Prediction', line=dict(color='red', dash='dash'))

    layout = go.Layout(title=title, xaxis={'title': 'x'}, yaxis={'title': 'y'})
    fig = go.Figure(data=[trace_train_data, trace_test_data, trace_train_pred, trace_test_pred], layout=layout)
    
    return fig

# Funktion zur Berechnung des Loss
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Hauptfunktion zur Verarbeitung der Eingaben und Rückgabe der Plots
def main(N, V, X_min, X_max, hidden_layers, neurons_per_layer, learning_rate, epochs, overfit_epochs, batch_size):
    # Daten generieren
    (x_train, y_train), (x_test, y_test), (x_train_noisy, y_train_noisy), (x_test_noisy, y_test_noisy) = generate_data(N, V, X_min, X_max)
    
    # Plot für noiseless Data
    noiseless_plot = go.Figure()
    noiseless_plot.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data', marker=dict(color='blue')))
    noiseless_plot.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data', marker=dict(color='red')))
    noiseless_plot.update_layout(title="Noiseless Datasets", xaxis_title='x', yaxis_title='y')

    # Plot für noisy Data
    noisy_plot = go.Figure()
    noisy_plot.add_trace(go.Scatter(x=x_train_noisy, y=y_train_noisy, mode='markers', name='Train Data', marker=dict(color='blue')))
    noisy_plot.add_trace(go.Scatter(x=x_test_noisy, y=y_test_noisy, mode='markers', name='Test Data', marker=dict(color='red')))
    noisy_plot.update_layout(title="Noisy Datasets", xaxis_title='x', yaxis_title='y')
    
    # Modell für noiseless Data trainieren
    model_noiseless = create_and_train_model((x_train, y_train), hidden_layers, neurons_per_layer, learning_rate, epochs, batch_size)
    noiseless_predictions_plot = plot_predictions(x_train, y_train, x_test, y_test, model_noiseless, "Unnoisy Model Predictions")

    # Modell für noisy Data trainieren (Best-Fit)
    model_noisy_best_fit = create_and_train_model((x_train_noisy, y_train_noisy), hidden_layers, neurons_per_layer, learning_rate, 200, batch_size)  # Epochen auf 200 gesetzt
    best_fit_predictions_plot = plot_predictions(x_train_noisy, y_train_noisy, x_test_noisy, y_test_noisy, model_noisy_best_fit, "Best-Fit Predictions")
    
    # Modell für Overfitting mit Noisy Data trainieren
    model_noisy_overfit = create_and_train_model((x_train_noisy, y_train_noisy), hidden_layers, neurons_per_layer, learning_rate, overfit_epochs, batch_size)
    overfit_predictions_plot = plot_predictions(x_train_noisy, y_train_noisy, x_test_noisy, y_test_noisy, model_noisy_overfit, "Overfit Predictions")
    
    return noiseless_plot, noisy_plot, noiseless_predictions_plot, best_fit_predictions_plot, overfit_predictions_plot

# Gradio Interface
interface = gr.Interface(
    fn=main,
    inputs=[
        gr.Slider(50, 500, value=100, label="Data Points (N)"),
        gr.Slider(0.01, 1.0, value=0.05, label="Noise Variance (V)"),
        gr.Slider(-5, 0, value=-2, step=0.1, label="X Min"),
        gr.Slider(0, 5, value=2, step=0.1, label="X Max"),
        gr.Slider(1, 10, value=2, label="Hidden Layers"),
        gr.Slider(10, 200, value=100, label="Neurons per Layer"),
        gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Learning Rate"),
        gr.Slider(10, 1000, value=100, label="Epochs"),
        gr.Slider(10, 1000, value=500, label="Overfit Epochs"),
        gr.Slider(1, 128, value=32, label="Batch Size")
    ],
    outputs=[
        gr.Plot(label="Noiseless Datasets"),
        gr.Plot(label="Noisy Datasets"),
        gr.Plot(label="Unnoisy Model Predictions"),
        gr.Plot(label="Best-Fit Predictions"),
        gr.Plot(label="Overfit Predictions")
    ],
    title="FFNN Regression with TensorFlow and Plotly",
    description="Train and visualize regression models using Feed-Forward Neural Network (FFNN) on noisy and noiseless data."
)

# Start Gradio Interface
if __name__ == "__main__":
    interface.launch()




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
