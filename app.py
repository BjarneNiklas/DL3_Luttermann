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
def create_model(num_layers, neurons_per_layer, activation):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation=activation, input_dim=1))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation=activation))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Train the model
def train_model(x_train, y_train, epochs, model):
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
def plot_predictions(x, y, model, title, show_true_function, x_min, x_max, data_type):
    x_range = np.linspace(x_min, x_max, 1000)
    y_pred = model.predict(x_range).flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{data_type} Data', marker=dict(color='blue' if data_type == 'Train' else 'red')))
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Prediction', line=dict(color='green')))
    if show_true_function:
        y_true = true_function(x_range)
        fig.add_trace(go.Scatter(x=x_range, y=y_true, mode='lines', name='True Function', line=dict(color='orange')))
    fig.update_layout(title=title)
    return fig

# Main function to handle training and plotting
def main(N, noise_variance, x_min, x_max, num_layers, neurons_per_layer, activation, epochs_unnoisy, epochs_best_fit, epochs_overfit, show_true_function):
    # Generate data
    x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy = generate_data(N, noise_variance, x_min, x_max)
    
    # Noiseless data plot
    noiseless_plot = plot_data(x_train, y_train, x_test, y_test, "Noiseless Datasets", show_true_function, x_min, x_max)
    
    # Noisy data plot
    noisy_plot = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, "Noisy Datasets", show_true_function, x_min, x_max)
    
    # Unnoisy model
    model_unnoisy = create_model(num_layers, neurons_per_layer, activation)
    model_unnoisy, loss_unnoisy = train_model(x_train, y_train, epochs_unnoisy, model_unnoisy)
    unnoisy_plot_train = plot_predictions(x_train, y_train, model_unnoisy, "Unnoisy Model - Train Data", show_true_function, x_min, x_max, "Train")
    unnoisy_plot_test = plot_predictions(x_test, y_test, model_unnoisy, "Unnoisy Model - Test Data", show_true_function, x_min, x_max, "Test")
    loss_unnoisy_test = model_unnoisy.evaluate(x_test, y_test, verbose=0)
    
    # Best-fit model
    model_best_fit = create_model(num_layers, neurons_per_layer, activation)
    model_best_fit, loss_best_fit = train_model(x_train, y_train_noisy, epochs_best_fit, model_best_fit)
    best_fit_plot_train = plot_predictions(x_train, y_train_noisy, model_best_fit, "Best-Fit Model - Train Data", show_true_function, x_min, x_max, "Train")
    best_fit_plot_test = plot_predictions(x_test, y_test_noisy, model_best_fit, "Best-Fit Model - Test Data", show_true_function, x_min, x_max, "Test")
    loss_best_fit_test = model_best_fit.evaluate(x_test, y_test_noisy, verbose=0)
    
    # Overfit model
    model_overfit = create_model(num_layers, neurons_per_layer, activation)
    model_overfit, loss_overfit = train_model(x_train, y_train_noisy, epochs_overfit, model_overfit)
    overfit_plot_train = plot_predictions(x_train, y_train_noisy, model_overfit, "Overfit Model - Train Data", show_true_function, x_min, x_max, "Train")
    overfit_plot_test = plot_predictions(x_test, y_test_noisy, model_overfit, "Overfit Model - Test Data", show_true_function, x_min, x_max, "Test")
    loss_overfit_test = model_overfit.evaluate(x_test, y_test_noisy, verbose=0)
    
    return (noiseless_plot, noisy_plot, 
            unnoisy_plot_train, f"Train Loss: {loss_unnoisy:.4f} | Test Loss: {loss_unnoisy_test:.4f}", unnoisy_plot_test, 
            best_fit_plot_train, f"Train Loss: {loss_best_fit:.4f} | Test Loss: {loss_best_fit_test:.4f}", best_fit_plot_test, 
            overfit_plot_train, f"Train Loss: {loss_overfit:.4f} | Test Loss: {loss_overfit_test:.4f}", overfit_plot_test)

# Gradio Interface
inputs = [
    gr.Slider(50, 200, step=1, value=100, label="Data Points (N)"),
    gr.Slider(0.01, 0.1, step=0.01, value=0.05, label="Noise Variance (V)"),
    gr.Slider(-3, -1, step=0.1, value=-2.0, label="X Min"),
    gr.Slider(1, 3, step=0.1, value=2.0, label="X Max"),
    gr.Slider(1, 5, step=1, value=2, label="Number of Hidden Layers"),
    gr.Slider(50, 200, step=10, value=100, label="Neurons per Hidden Layer"),
    gr.Dropdown(["relu", "sigmoid", "tanh"], value="relu", label="Activation Function"),
    gr.Slider(50, 500, step=10, value=100, label="Epochs (Unnoisy Model)"),
    gr.Slider(50, 500, step=10, value=200, label="Epochs (Best-Fit Model)"),
    gr.Slider(50, 500, step=10, value=500, label="Epochs (Overfit Model)"),
    gr.Checkbox(value=True, label="Show True Function")
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

def wrapper(*args):
    return main(*args)

demo = gr.Blocks()
with demo:
    gr.Markdown("## Regression with Feed-Forward Neural Network")
    with gr.Row():
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
