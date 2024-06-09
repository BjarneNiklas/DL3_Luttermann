import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import gradio as gr

# Step 1: Generate Data
def generate_data(noise_variance=0.05):
    np.random.seed(0)
    x = np.random.uniform(-2, 2, 100)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

    noise = np.random.normal(0, noise_variance**0.5, size=y.shape)
    y_noisy = y + noise

    idx = np.random.permutation(len(x))
    train_idx, test_idx = idx[:50], idx[50:]
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_noisy, y_test_noisy = y_noisy[train_idx], y_noisy[test_idx]

    return (x_train, y_train), (x_test, y_test), (x_train, y_train_noisy), (x_test, y_test_noisy)

# Step 2: Define the Model
def create_model():
    model = Sequential([
        Dense(100, activation='relu', input_shape=(1,)),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Step 3: Train Models
def train_model(x_train, y_train, epochs):
    model = create_model()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model, history

# Step 4: Plotting
def plot_data(x_train, y_train, x_test, y_test, x_train_noisy, y_train_noisy, x_test_noisy, y_test_noisy):
    fig = make_subplots(rows=4, cols=2, subplot_titles=[
        "Original Data", "Noisy Data", "Unnoised Model Train", "Unnoised Model Test",
        "Best Fit Model Train", "Best Fit Model Test", "Overfit Model Train", "Overfit Model Test"
    ])

    # Original Data
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data'), row=1, col=1)
    
    # Noisy Data
    fig.add_trace(go.Scatter(x=x_train_noisy, y=y_train_noisy, mode='markers', name='Train Data'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_test_noisy, y=y_test_noisy, mode='markers', name='Test Data'), row=1, col=2)

    return fig

# Step 5: Creating Gradio Interface
def gradio_interface(noise_variance, epochs_best, epochs_overfit):
    (x_train, y_train), (x_test, y_test), (x_train_noisy, y_train_noisy), (x_test_noisy, y_test_noisy) = generate_data(noise_variance)
    
    # Train models
    model_unnoised, _ = train_model(x_train, y_train, 200)
    model_best, _ = train_model(x_train_noisy, y_train_noisy, epochs_best)
    model_overfit, _ = train_model(x_train_noisy, y_train_noisy, epochs_overfit)
    
    # Predictions
    y_train_pred_unnoised = model_unnoised.predict(x_train)
    y_test_pred_unnoised = model_unnoised.predict(x_test)
    y_train_pred_best = model_best.predict(x_train_noisy)
    y_test_pred_best = model_best.predict(x_test_noisy)
    y_train_pred_overfit = model_overfit.predict(x_train_noisy)
    y_test_pred_overfit = model_overfit.predict(x_test_noisy)
    
    # Plotting
    fig = plot_data(x_train, y_train, x_test, y_test, x_train_noisy, y_train_noisy, x_test_noisy, y_test_noisy)
    
    # Unnoised Model Predictions
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_unnoised.flatten(), mode='markers', name='Train Predictions'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_unnoised.flatten(), mode='markers', name='Test Predictions'), row=2, col=2)
    
    # Best Fit Model Predictions
    fig.add_trace(go.Scatter(x=x_train_noisy, y=y_train_pred_best.flatten(), mode='markers', name='Train Predictions'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_test_noisy, y=y_test_pred_best.flatten(), mode='markers', name='Test Predictions'), row=3, col=2)
    
    # Overfit Model Predictions
    fig.add_trace(go.Scatter(x=x_train_noisy, y=y_train_pred_overfit.flatten(), mode='markers', name='Train Predictions'), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_test_noisy, y=y_test_pred_overfit.flatten(), mode='markers', name='Test Predictions'), row=4, col=2)
    
    return fig

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            noise_variance = gr.Slider(minimum=0, maximum=1, value=0.05, label="Noise Variance")
            epochs_best = gr.Slider(minimum=100, maximum=500, value=200, step=10, label="Epochs for Best Fit Model")
            epochs_overfit = gr.Slider(minimum=100, maximum=500, value=500, step=10, label="Epochs for Overfit Model")
            button = gr.Button("Run")

        with gr.Column():
            output = gr.Plot(label="Results")

    button.click(fn=gradio_interface, inputs=[noise_variance, epochs_best, epochs_overfit], outputs=output)

demo.launch()
