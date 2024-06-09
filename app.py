import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import gradio as gr

# Define the true function
def true_function(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Generate datasets
def generate_data(N, noise_variance, x_min, x_max):
    x = np.random.uniform(x_min, x_max, N)
    y = true_function(x)
    x_train, x_test = x[:N//2], x[N//2:]
    y_train, y_test = y[:N//2], y[N//2:]

    noise_train = np.random.normal(0, noise_variance**0.5, y_train.shape)
    noise_test = np.random.normal(0, noise_variance**0.5, y_test.shape)
    y_train_noisy = y_train + noise_train
    y_test_noisy = y_test + noise_test
    
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
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model, history.history['loss'][-1]

# Plot data and predictions
def plot_data(x_train, y_train, x_test, y_test, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data', marker=dict(color='red')))
    fig.update_layout(title=title)
    return fig

def plot_predictions(model, x_train, y_train, x_test, y_test, title):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred[:, 0], mode='lines', name='Train Prediction', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred[:, 0], mode='lines', name='Test Prediction', line=dict(color='red')))
    fig.update_layout(title=title)
    return fig

def main(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit):
    # Generate data
    x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy = generate_data(N, noise_variance, x_min, x_max)
    
    # Noiseless data plot
    noiseless_plot = plot_data(x_train, y_train, x_test, y_test, "Noiseless Datasets")
    
    # Noisy data plot
    noisy_plot = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, "Noisy Datasets")
    
    # Unnoisy model
    model_unnoisy, loss_unnoisy = train_model(x_train, y_train, epochs_unnoisy)
    unnoisy_plot_train = plot_predictions(model_unnoisy, x_train, y_train, x_train, y_train, "Unnoisy Model - Train Data")
    unnoisy_plot_test = plot_predictions(model_unnoisy, x_train, y_train, x_test, y_test, "Unnoisy Model - Test Data")
    loss_unnoisy_test = model_unnoisy.evaluate(x_test, y_test, verbose=0)
    
    # Best-fit model
    model_best, loss_best = train_model(x_train, y_train_noisy, epochs_best)
    best_fit_plot_train = plot_predictions(model_best, x_train, y_train_noisy, x_train, y_train_noisy, "Best-Fit Model - Train Data")
    best_fit_plot_test = plot_predictions(model_best, x_train, y_train_noisy, x_test, y_test_noisy, "Best-Fit Model - Test Data")
    loss_best_test = model_best.evaluate(x_test, y_test_noisy, verbose=0)
    
    # Overfit model
    model_overfit, loss_overfit = train_model(x_train, y_train_noisy, epochs_overfit)
    overfit_plot_train = plot_predictions(model_overfit, x_train, y_train_noisy, x_train, y_train_noisy, "Overfit Model - Train Data")
    overfit_plot_test = plot_predictions(model_overfit, x_train, y_train_noisy, x_test, y_test_noisy, "Overfit Model - Test Data")
    loss_overfit_test = model_overfit.evaluate(x_test, y_test_noisy, verbose=0)
    
    return (noiseless_plot, noisy_plot, 
            unnoisy_plot_train, f"Train Loss: {loss_unnoisy:.4f} | Test Loss: {loss_unnoisy_test:.4f}", unnoisy_plot_test, 
            best_fit_plot_train, f"Train Loss: {loss_best:.4f} | Test Loss: {loss_best_test:.4f}", best_fit_plot_test, 
            overfit_plot_train, f"Train Loss: {loss_overfit:.4f} | Test Loss: {loss_overfit_test:.4f}", overfit_plot_test)

# Gradio Interface
inputs = [
    gr.Slider(50, 200, step=1, value=100, label="Data Points (N)"),
    gr.Slider(0.01, 0.1, step=0.01, value=0.05, label="Noise Variance (V)"),
    gr.Slider(-3, -1, step=0.1, value=-2.0, label="X Min"),
    gr.Slider(1, 3, step=0.1, value=2.0, label="X Max"),
    gr.Slider(50, 500, step=10, value=100, label="Epochs (Unnoisy Model)"),
    gr.Slider(50, 500, step=10, value=200, label="Epochs (Best-Fit Model)"),
    gr.Slider(50, 500, step=10, value=500, label="Epochs (Overfit Model)")
]

outputs = [
    gr.Plot(label="Noiseless Datasets"),
    gr.Plot(label="Noisy Datasets"),
    gr.Plot(label="Unnoisy Model - Train Data"),
    gr.Textbox(label="Unnoisy Model Losses"),
    gr.Plot(label="Unnoisy Model - Test Data"),
    gr.Plot(label="Best-Fit Model - Train Data"),
    gr.Textbox(label="Best-Fit Model Losses"),
    gr.Plot(label="Best-Fit Model - Test Data"),
    gr.Plot(label="Overfit Model - Train Data"),
    gr.Textbox(label="Overfit Model Losses"),
    gr.Plot(label="Overfit Model - Test Data")
]

def wrapper(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit):
    return main(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit)

# Define the interface
with gr.Blocks() as demo:
    gr.Markdown("## Regression with Feed-Forward Neural Network")
    with gr.Row():
        with gr.Column():
            for input_widget in inputs:
                input_widget.render()
            gr.Button("Generate Data and Train Models").click(wrapper, inputs, outputs)
    
    with gr.Row():
        with gr.Column():
            outputs[0].render()
        with gr.Column():
            outputs[1].render()
    
    with gr.Row():
        with gr.Column():
            outputs[2].render()
            outputs[3].render()
        with gr.Column():
            outputs[4].render()
    
    with gr.Row():
        with gr.Column():
            outputs[5].render()
            outputs[6].render()
        with gr.Column():
            outputs[7].render()
    
    with gr.Row():
        with gr.Column():
            outputs[8].render()
            outputs[9].render()
        with gr.Column():
            outputs[10].render()
    
demo.launch()
