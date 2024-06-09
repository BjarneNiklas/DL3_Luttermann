import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import gradio as gr
from plotly.subplots import make_subplots

# Data generation function
def generate_data(N, noise_variance):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    y_noisy = y + np.random.normal(0, np.sqrt(noise_variance), N)
    
    # Split into train and test sets
    idx = np.random.permutation(N)
    train_idx, test_idx = idx[:N//2], idx[N//2:]
    x_train, y_train, y_train_noisy = x[train_idx], y[train_idx], y_noisy[train_idx]
    x_test, y_test, y_test_noisy = x[test_idx], y[test_idx], y_noisy[test_idx]
    
    return (x_train, y_train, y_train_noisy), (x_test, y_test, y_test_noisy)

# Define the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)  # Linear activation
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse')
    return model

# Train and evaluate models
def train_models(x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy, epochs_best, epochs_overfit):
    # Model on clean data
    model_clean = create_model()
    history_clean = model_clean.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    # Best-fit model on noisy data
    model_best = create_model()
    history_best = model_best.fit(x_train, y_train_noisy, epochs=epochs_best, batch_size=32, verbose=0)
    
    # Overfit model on noisy data
    model_overfit = create_model()
    history_overfit = model_overfit.fit(x_train, y_train_noisy, epochs=epochs_overfit, batch_size=32, verbose=0)
    
    # Evaluate models
    loss_clean_train = model_clean.evaluate(x_train, y_train, verbose=0)
    loss_clean_test = model_clean.evaluate(x_test, y_test, verbose=0)
    
    loss_best_train = model_best.evaluate(x_train, y_train_noisy, verbose=0)
    loss_best_test = model_best.evaluate(x_test, y_test_noisy, verbose=0)
    
    loss_overfit_train = model_overfit.evaluate(x_train, y_train_noisy, verbose=0)
    loss_overfit_test = model_overfit.evaluate(x_test, y_test_noisy, verbose=0)
    
    # Predictions
    y_pred_clean_train = model_clean.predict(x_train)
    y_pred_clean_test = model_clean.predict(x_test)
    
    y_pred_best_train = model_best.predict(x_train)
    y_pred_best_test = model_best.predict(x_test)
    
    y_pred_overfit_train = model_overfit.predict(x_train)
    y_pred_overfit_test = model_overfit.predict(x_test)
    
    return {
        'history_clean': history_clean,
        'history_best': history_best,
        'history_overfit': history_overfit,
        'losses': {
            'clean_train': loss_clean_train,
            'clean_test': loss_clean_test,
            'best_train': loss_best_train,
            'best_test': loss_best_test,
            'overfit_train': loss_overfit_train,
            'overfit_test': loss_overfit_test
        },
        'predictions': {
            'clean_train': y_pred_clean_train,
            'clean_test': y_pred_clean_test,
            'best_train': y_pred_best_train,
            'best_test': y_pred_best_test,
            'overfit_train': y_pred_overfit_train,
            'overfit_test': y_pred_overfit_test
        }
    }

# Visualization function
def plot_results(data, results):
    (x_train, y_train, y_train_noisy), (x_test, y_test, y_test_noisy) = data
    preds = results['predictions']
    
    fig = make_subplots(rows=4, cols=2, subplot_titles=[
        "Clean Data (Train/Test)", "Noisy Data (Train/Test)",
        "Model Prediction (Clean, Train)", "Model Prediction (Clean, Test)",
        "Model Prediction (Best-Fit, Train)", "Model Prediction (Best-Fit, Test)",
        "Model Prediction (Overfit, Train)", "Model Prediction (Overfit, Test)"
    ])
    
    # Row 1: Clean and Noisy Data
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train (Clean)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test (Clean)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Train (Noisy)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Test (Noisy)'), row=1, col=2)
    
    # Row 2: Model Prediction on Clean Data
    fig.add_trace(go.Scatter(x=x_train, y=preds['clean_train'].flatten(), mode='markers', name='Pred (Clean, Train)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=preds['clean_test'].flatten(), mode='markers', name='Pred (Clean, Test)'), row=2, col=2)
    
    # Row 3: Model Prediction on Noisy Data (Best-Fit)
    fig.add_trace(go.Scatter(x=x_train, y=preds['best_train'].flatten(), mode='markers', name='Pred (Best-Fit, Train)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=preds['best_test'].flatten(), mode='markers', name='Pred (Best-Fit, Test)'), row=3, col=2)
    
    # Row 4: Model Prediction on Noisy Data (Overfit)
    fig.add_trace(go.Scatter(x=x_train, y=preds['overfit_train'].flatten(), mode='markers', name='Pred (Overfit, Train)'), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=preds['overfit_test'].flatten(), mode='markers', name='Pred (Overfit, Test)'), row=4, col=2)
    
    fig.update_layout(height=1000, width=1200, title_text="FFNN Regression Results")
    
    return fig

# Gradio interface
def interface(epochs_best=200, epochs_overfit=500):
    data = generate_data(100, 0.05)
    results = train_models(*data, epochs_best, epochs_overfit)
    fig = plot_results(data, results)
    
    losses = results['losses']
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