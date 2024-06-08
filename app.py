import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

# Function definition
def true_function(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Define model architecture
def build_model(hidden_layers=2, neurons_per_layer=100, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_shape=(1,)))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def train_and_visualize(data_points, noise_variance, hidden_layers, neurons_per_layer, learning_rate, epochs, batch_size):
    # Generate data
    X = np.random.uniform(-2, 2, data_points)
    Y = true_function(X)

    # Split into train and test datasets
    train_size = data_points // 2
    indices = np.random.permutation(data_points)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Add Gaussian noise to the output
    Y_train_noisy = Y_train + np.random.normal(0, np.sqrt(noise_variance), train_size)
    Y_test_noisy = Y_test + np.random.normal(0, np.sqrt(noise_variance), data_points - train_size)

    # Train model on noiseless data
    model_noiseless = build_model(hidden_layers, neurons_per_layer, learning_rate)
    model_noiseless.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Train model on noisy data (best-fit)
    model_best_fit = build_model(hidden_layers, neurons_per_layer, learning_rate)
    model_best_fit.fit(X_train, Y_train_noisy, epochs=epochs, batch_size=batch_size, verbose=0)

    # Train model on noisy data (overfit)
    model_overfit = build_model(hidden_layers, neurons_per_layer, learning_rate)
    model_overfit.fit(X_train, Y_train_noisy, epochs=epochs * 10, batch_size=batch_size, verbose=0)

    # Evaluate models
    Y_pred_noiseless_train = model_noiseless.predict(X_train).flatten()
    Y_pred_noiseless_test = model_noiseless.predict(X_test).flatten()
    Y_pred_best_fit_train = model_best_fit.predict(X_train).flatten()
    Y_pred_best_fit_test = model_best_fit.predict(X_test).flatten()
    Y_pred_overfit_train = model_overfit.predict(X_train).flatten()
    Y_pred_overfit_test = model_overfit.predict(X_test).flatten()

    # Create subplots
    fig = make_subplots(rows=3, cols=2, subplot_titles=(
        'Noiseless Train', 'Noiseless Test', 'Best-Fit Train', 'Best-Fit Test', 'Overfit Train', 'Overfit Test'))

    # Plot results for noiseless model
    fig.add_trace(go.Scatter(x=X_train, y=Y_train, mode='markers', name='Train Data', marker=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_train, y=Y_pred_noiseless_train, mode='lines', name='Model Prediction', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_test, y=Y_test, mode='markers', name='Test Data', marker=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=X_test, y=Y_pred_noiseless_test, mode='lines', name='Model Prediction', line=dict(color='red')), row=1, col=2)

    # Plot results for best-fit model
    fig.add_trace(go.Scatter(x=X_train, y=Y_train_noisy, mode='markers', name='Train Data', marker=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=X_train, y=Y_pred_best_fit_train, mode='lines', name='Model Prediction', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=X_test, y=Y_test_noisy, mode='markers', name='Test Data', marker=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=X_test, y=Y_pred_best_fit_test, mode='lines', name='Model Prediction', line=dict(color='red')), row=2, col=2)

    # Plot results for overfit model
    fig.add_trace(go.Scatter(x=X_train, y=Y_train_noisy, mode='markers', name='Train Data', marker=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=X_train, y=Y_pred_overfit_train, mode='lines', name='Model Prediction', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=X_test, y=Y_test_noisy, mode='markers', name='Test Data', marker=dict(color='blue')), row=3, col=2)
    fig.add_trace(go.Scatter(x=X_test, y=Y_pred_overfit_test, mode='lines', name='Model Prediction', line=dict(color='red')), row=3, col=2)

    fig.update_layout(title='Model Predictions', showlegend=False, height=900)

    return fig

iface = gr.Interface(
    fn=train_and_visualize,
    inputs=[
        gr.components.Slider(50, 500, step=10, value=100, label='Data Points (N)'),
        gr.components.Slider(0.01, 0.2, step=0.01, value=0.05, label='Noise Variance (V)'),
        gr.components.Slider(1, 5, step=1, value=2, label='Hidden Layers'),
        gr.components.Slider(10, 200, step=10, value=100, label='Neurons per Layer'),
        gr.components.Number(value=0.01, label='Learning Rate'),
        gr.components.Slider(10, 1000, step=10, value=100, label='Epochs'),
        gr.components.Slider(16, 128, step=16, value=32, label='Batch Size')
    ],
    outputs=gr.components.Plot(label='Model Predictions'),
    title='FFNN Regression with TensorFlow',
    description='An interactive tool for training and visualizing Feed-Forward Neural Networks for regression tasks.'
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
