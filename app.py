import numpy as np
import plotly.graph_objects as go
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the true function
def true_function(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Generate datasets
def generate_data(N, noise_variance, x_min, x_max):
    x = np.random.uniform(x_min, x_max, N)
    y = true_function(x)
    x_train, x_test = x[:N//2], x[N//2:]
    y_train, y_test = y[:N//2], y[N//2:]

    noise = np.random.normal(0, noise_variance**0.5, y_train.shape)
    y_train_noisy = y_train + noise
    y_test_noisy = y_test + noise
    
    return x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy

# Define the neural network model
def create_model():
    model = Sequential([
        Dense(100, activation='relu', input_dim=1),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Train the model
def train_model(x_train, y_train, epochs):
    model = create_model()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

# Plot data and predictions
def plot_data(x_train, y_train, x_test, y_test, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data'))
    fig.update_layout(title=title)
    return fig

def plot_predictions(model, x_train, y_train, x_test, y_test, title):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data'))
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred[:, 0], mode='markers', name='Train Prediction'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred[:, 0], mode='markers', name='Test Prediction'))
    fig.update_layout(title=title)
    return fig

def main(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit):
    # Generate data
    x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy = generate_data(N, noise_variance, x_min, x_max)
    
    # Unnoisy model
    model_unnoisy = train_model(x_train, y_train, epochs_unnoisy)
    unnoisy_plot_train = plot_predictions(model_unnoisy, x_train, y_train, x_train, y_train, "Unnoisy Model - Train Data")
    unnoisy_plot_test = plot_predictions(model_unnoisy, x_train, y_train, x_test, y_test, "Unnoisy Model - Test Data")
    
    # Best-fit model
    model_best = train_model(x_train, y_train_noisy, epochs_best)
    best_fit_plot_train = plot_predictions(model_best, x_train, y_train_noisy, x_train, y_train_noisy, "Best-Fit Model - Train Data")
    best_fit_plot_test = plot_predictions(model_best, x_train, y_train_noisy, x_test, y_test_noisy, "Best-Fit Model - Test Data")
    
    # Overfit model
    model_overfit = train_model(x_train, y_train_noisy, epochs_overfit)
    overfit_plot_train = plot_predictions(model_overfit, x_train, y_train_noisy, x_train, y_train_noisy, "Overfit Model - Train Data")
    overfit_plot_test = plot_predictions(model_overfit, x_train, y_train_noisy, x_test, y_test_noisy, "Overfit Model - Test Data")
    
    # Data plots
    noiseless_plot = plot_data(x_train, y_train, x_test, y_test, "Noiseless Datasets")
    noisy_plot = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, "Noisy Datasets")
    
    return noiseless_plot, noisy_plot, unnoisy_plot_train, unnoisy_plot_test, best_fit_plot_train, best_fit_plot_test, overfit_plot_train, overfit_plot_test

# Gradio Interface
interface = gr.Interface(
    fn=main,
    inputs=[
        gr.Slider(50, 200, step=1, value=100, label="Data Points (N)"),
        gr.Slider(0.01, 0.1, step=0.01, value=0.05, label="Noise Variance (V)"),
        gr.Slider(-3, -1, step=0.1, value=-2.0, label="X Min"),
        gr.Slider(1, 3, step=0.1, value=2.0, label="X Max"),
        gr.Slider(50, 200, step=1, value=100, label="Epochs for Unnoisy Model"),
        gr.Slider(100, 300, step=1, value=200, label="Epochs for Best-Fit Model"),
        gr.Slider(300, 600, step=1, value=500, label="Epochs for Overfit Model")
    ],
    outputs=[
        gr.Plot(), gr.Plot(),
        gr.Plot(), gr.Plot(),
        gr.Plot(), gr.Plot(),
        gr.Plot(), gr.Plot()
    ],
    layout="horizontal"
)

interface.launch()
