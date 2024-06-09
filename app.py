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

# Plot data
def plot_data(x_train, y_train, x_test, y_test, title, show_true_function, x_min, x_max):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data', marker=dict(color='red')))
    if show_true_function:
        x_range = np.linspace(x_min, x_max, 1000)
        y_true = true_function(x_range)
        fig.add_trace(go.Scatter(x=x_range, y=y_true, mode='lines', name='True Function', line=dict(color='green')))
    fig.update_layout(title=title)
    return fig

# Plot predictions
def plot_predictions(x, y, model, title, show_true_function, x_min, x_max, is_train):
    x_range = np.linspace(x_min, x_max, 1000)
    y_pred = model.predict(x_range).flatten()
    
    fig = go.Figure()
    if is_train:
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Train Data', marker=dict(color='blue')))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Test Data', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Prediction', line=dict(color='green')))
    if show_true_function:
        y_true = true_function(x_range)
        fig.add_trace(go.Scatter(x=x_range, y=y_true, mode='lines', name='True Function', line=dict(color='orange')))
    fig.update_layout(title=title)
    return fig

def main(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit, show_true_function):
    # Generate data
    x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy = generate_data(N, noise_variance, x_min, x_max)
    
    # Noiseless data plot
    noiseless_plot = plot_data(x_train, y_train, x_test, y_test, "Noiseless Datasets", show_true_function, x_min, x_max)
    
    # Noisy data plot
    noisy_plot = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, "Noisy Datasets", show_true_function, x_min, x_max)
    
    # Unnoisy model
    model_unnoisy, loss_unnoisy = train_model(x_train, y_train, epochs_unnoisy)
    unnoisy_plot_train = plot_predictions(x_train, y_train, model_unnoisy, "Unnoisy Model - Train Data", show_true_function, x_min, x_max, True)
    unnoisy_plot_test = plot_predictions(x_test, y_test, model_unnoisy, "Unnoisy Model - Test Data", show_true_function, x_min, x_max, False)
    loss_unnoisy_test = model_unnoisy.evaluate(x_test, y_test, verbose=0)
    
    # Best-fit model
    model_best, loss_best = train_model(x_train, y_train_noisy, epochs_best)
    best_fit_plot_train = plot_predictions(x_train, y_train_noisy, model_best, "Best-Fit Model - Train Data", show_true_function, x_min, x_max, True)
    best_fit_plot_test = plot_predictions(x_test, y_test_noisy, model_best, "Best-Fit Model - Test Data", show_true_function, x_min, x_max, False)
    loss_best_test = model_best.evaluate(x_test, y_test_noisy, verbose=0)
    
    # Overfit model
    model_overfit, loss_overfit = train_model(x_train, y_train_noisy, epochs_overfit)
    overfit_plot_train = plot_predictions(x_train, y_train_noisy, model_overfit, "Overfit Model - Train Data", show_true_function, x_min, x_max, True)
    overfit_plot_test = plot_predictions(x_test, y_test_noisy, model_overfit, "Overfit Model - Test Data", show_true_function, x_min, x_max, False)
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
    gr.Slider(50, 500, step=10, value=500, label="Epochs (Overfit Model)"),
    gr.Checkbox(label="Show True Function", value=True)
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

def wrapper(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit, show_true_function):
    return main(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit, show_true_function)

def update_true_function(show_true_function):
    N = inputs[0].value
    noise_variance = inputs[1].value
    x_min = inputs[2].value
    x_max = inputs[3].value
    epochs_unnoisy = inputs[4].value
    epochs_best = inputs[5].value
    epochs_overfit = inputs[6].value
    return main(N, noise_variance, x_min, x_max, epochs_unnoisy, epochs_best, epochs_overfit, show_true_function)

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

    inputs[7].change(fn=update_true_function, inputs=inputs[7], outputs=[outputs[0], outputs[1], outputs[2], outputs[4], outputs[5], outputs[7], outputs[8], outputs[10]])

demo.launch()
