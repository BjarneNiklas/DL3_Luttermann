import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import gradio as gr
from plotly.subplots import make_subplots

# Function to generate data
def generate_data(N=100, noise_variance=0.05):
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    y_noisy = y + np.random.normal(0, np.sqrt(noise_variance), N)
    
    indices = np.random.permutation(N)
    train_idx, test_idx = indices[:N//2], indices[N//2:]
    data = {
        "train": {"x": x[train_idx], "y": y[train_idx], "y_noisy": y_noisy[train_idx]},
        "test": {"x": x[test_idx], "y": y[test_idx], "y_noisy": y_noisy[test_idx]}
    }
    return data

# Function to create the model
def create_ffnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)  # Linear activation by default
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

# Function to train models and return predictions
def train_models(data, epochs_best=200, epochs_overfit=500):
    model_clean = create_ffnn()
    model_clean.fit(data["train"]["x"], data["train"]["y"], epochs=100, batch_size=32, verbose=0)
    
    model_best = create_ffnn()
    model_best.fit(data["train"]["x"], data["train"]["y_noisy"], epochs=epochs_best, batch_size=32, verbose=0)
    
    model_overfit = create_ffnn()
    model_overfit.fit(data["train"]["x"], data["train"]["y_noisy"], epochs=epochs_overfit, batch_size=32, verbose=0)
    
    results = {
        "clean": {"train": model_clean.predict(data["train"]["x"]), "test": model_clean.predict(data["test"]["x"])},
        "best": {"train": model_best.predict(data["train"]["x"]), "test": model_best.predict(data["test"]["x"])},
        "overfit": {"train": model_overfit.predict(data["train"]["x"]), "test": model_overfit.predict(data["test"]["x"])},
    }
    
    losses = {
        "clean_train": model_clean.evaluate(data["train"]["x"], data["train"]["y"], verbose=0),
        "clean_test": model_clean.evaluate(data["test"]["x"], data["test"]["y"], verbose=0),
        "best_train": model_best.evaluate(data["train"]["x"], data["train"]["y_noisy"], verbose=0),
        "best_test": model_best.evaluate(data["test"]["x"], data["test"]["y_noisy"], verbose=0),
        "overfit_train": model_overfit.evaluate(data["train"]["x"], data["train"]["y_noisy"], verbose=0),
        "overfit_test": model_overfit.evaluate(data["test"]["x"], data["test"]["y_noisy"], verbose=0),
    }
    
    return results, losses

# Function to plot the results
def plot_results(data, results):
    fig = make_subplots(rows=4, cols=2, subplot_titles=[
        "Clean Data (Train/Test)", "Noisy Data (Train/Test)",
        "Model Prediction (Clean, Train)", "Model Prediction (Clean, Test)",
        "Model Prediction (Best-Fit, Train)", "Model Prediction (Best-Fit, Test)",
        "Model Prediction (Overfit, Train)", "Model Prediction (Overfit, Test)"
    ])
    
    fig.add_trace(go.Scatter(x=data["train"]["x"], y=data["train"]["y"], mode='markers', name='Train (Clean)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["test"]["x"], y=data["test"]["y"], mode='markers', name='Test (Clean)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["train"]["x"], y=data["train"]["y_noisy"], mode='markers', name='Train (Noisy)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=data["test"]["x"], y=data["test"]["y_noisy"], mode='markers', name='Test (Noisy)'), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=data["train"]["x"], y=results["clean"]["train"].flatten(), mode='markers', name='Pred (Clean, Train)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data["test"]["x"], y=results["clean"]["test"].flatten(), mode='markers', name='Pred (Clean, Test)'), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=data["train"]["x"], y=results["best"]["train"].flatten(), mode='markers', name='Pred (Best-Fit, Train)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data["test"]["x"], y=results["best"]["test"].flatten(), mode='markers', name='Pred (Best-Fit, Test)'), row=3, col=2)
    
    fig.add_trace(go.Scatter(x=data["train"]["x"], y=results["overfit"]["train"].flatten(), mode='markers', name='Pred (Overfit, Train)'), row=4, col=1)
    fig.add_trace(go.Scatter(x=data["test"]["x"], y=results["overfit"]["test"].flatten(), mode='markers', name='Pred (Overfit, Test)'), row=4, col=2)
    
    fig.update_layout(height=1200, width=1200, title_text="FFNN Regression Results")
    return fig

# Gradio interface
def interface(epochs_best=200, epochs_overfit=500):
    data = generate_data()
    results, losses = train_models(data, epochs_best, epochs_overfit)
    fig = plot_results(data, results)
    
    loss_text = f"""
    Clean Model: Train Loss = {losses['clean_train']:.4f}, Test Loss = {losses['clean_test']:.4f}\n
    Best-Fit Model: Train Loss = {losses['best_train']:.4f}, Test Loss = {losses['best_test']:.4f}\n
    Overfit Model: Train Loss = {losses['overfit_train']:.4f}, Test Loss = {losses['overfit_test']:.4f}
    """
    
    return fig, loss_text

# Gradio app
gr.Interface(
    fn=interface,
    inputs=[
        gr.Slider(50, 500, value=200, step=10, label="Epochs for Best-Fit Model"),
        gr.Slider(50, 1000, value=500, step=10, label="Epochs for Overfit Model")
    ],
    outputs=[
        gr.Plot(label="FFNN Regression Results"),
        gr.Textbox(label="Loss Values")
    ],
    layout="vertical",
    title="FFNN Regression with Plotly and Gradio",
    description="Train and visualize FFNN models on noisy and clean data."
).launch()