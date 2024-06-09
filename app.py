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
def create_model(num_layers, neurons_per_layer):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_dim=1))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
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
    y_pred_points = model.predict(x).flatten()  # Vorhersagen für die tatsächlichen Datenpunkte

    if data_type == "Train":
        x_range = np.linspace(x_min, x_max, 1000)
        y_pred = model.predict(x_range).flatten()  # Vorhersagelinie basierend auf Trainingsdaten
    else:
        x_range = x  # Verwende die Testdaten für die x-Werte
        y_pred = y_pred_points  # Verwende die Vorhersagen für die Testdaten als Vorhersagelinie

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{data_type} Data', marker=dict(color='blue' if data_type == 'Train' else 'red')))
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Prediction Line', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x, y=y_pred_points, mode='markers', name='Prediction Points', marker=dict(color='green')))

    if show_true_function:
        y_true = true_function(x_range)
        fig.add_trace(go.Scatter(x=x_range, y=y_true, mode='lines', name='True Function', line=dict(color='orange')))
    fig.update_layout(title=title)
    return fig

# Main function to handle training and plotting
def main(N, noise_variance, x_min, x_max, num_layers, neurons_per_layer, epochs_unnoisy, epochs_best_fit, epochs_overfit, show_true_function):
    # Generate data
    x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy = generate_data(N, noise_variance, x_min, x_max)

    # Noiseless data plot
    noiseless_plot = plot_data(x_train, y_train, x_test, y_test, "Noiseless Datasets", show_true_function, x_min, x_max)

    # Noisy data plot
    noisy_plot = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, "Noisy Datasets", show_true_function, x_min, x_max)

    # Unnoisy model
    model_unnoisy = create_model(num_layers, neurons_per_layer)
    model_unnoisy, loss_unnoisy_train = train_model(x_train, y_train, epochs_unnoisy, model_unnoisy)
    unnoisy_plot_train = plot_predictions(x_train, y_train, model_unnoisy, "Unnoisy Model - Train Data", show_true_function, x_min, x_max, "Train")
    unnoisy_plot_test = plot_predictions(x_test, y_test, model_unnoisy, "Unnoisy Model - Test Data", show_true_function, x_min, x_max, "Test")
    loss_unnoisy_test = model_unnoisy.evaluate(x_test, y_test, verbose=0)

    # Best-fit model
    model_best_fit = create_model(num_layers, neurons_per_layer)
    model_best_fit, loss_best_fit_train = train_model(x_train, y_train_noisy, epochs_best_fit, model_best_fit)
    best_fit_plot_train = plot_predictions(x_train, y_train_noisy, model_best_fit, "Best-Fit Model - Train Data", show_true_function, x_min, x_max, "Train")
    best_fit_plot_test = plot_predictions(x_test, y_test_noisy, model_best_fit, "Best-Fit Model - Test Data", show_true_function, x_min, x_max, "Test")
    loss_best_fit_test = model_best_fit.evaluate(x_test, y_test_noisy, verbose=0)

    # Overfit model
    model_overfit = create_model(num_layers, neurons_per_layer)
    model_overfit, loss_overfit_train = train_model(x_train, y_train_noisy, epochs_overfit, model_overfit)
    overfit_plot_train = plot_predictions(x_train, y_train_noisy, model_overfit, "Overfit Model - Train Data", show_true_function, x_min, x_max, "Train")
    overfit_plot_test = plot_predictions(x_test, y_test_noisy, model_overfit, "Overfit Model - Test Data", show_true_function, x_min, x_max, "Test")
    loss_overfit_test = model_overfit.evaluate(x_test, y_test_noisy, verbose=0)

    return (noiseless_plot, noisy_plot,
            unnoisy_plot_train, f"Train Loss: {loss_unnoisy_train:.4f} | Test Loss: {loss_unnoisy_test:.4f}", unnoisy_plot_test,
            best_fit_plot_train, f"Train Loss: {loss_best_fit_train:.4f} | Test Loss: {loss_best_fit_test:.4f}", best_fit_plot_test,
            overfit_plot_train, f"Train Loss: {loss_overfit_train:.4f} | Test Loss: {loss_overfit_test:.4f}", overfit_plot_test)

# Discussion and Documentation
discussion = """
## Discussion

The task of regression using a feed-forward neural network (FFNN) is a fundamental problem in machine learning. By training the FFNN on noisy data generated from an unknown ground truth function, we simulate real-world scenarios where the underlying function is not explicitly known, and the data is subject to noise and uncertainties.

In this solution, we generate data points from a given function and add Gaussian noise to simulate real-world conditions. We then split the data into training and test sets, ensuring that the test set remains untouched during the training process to provide an unbiased evaluation of the model's performance.

Three models are trained: one on the clean data (no noise), one on the noisy data to achieve the best fit, and one on the noisy data with a higher number of epochs to demonstrate overfitting. By comparing the losses (mean squared error) on the training and test sets, we can observe the effects of overfitting and identify the model that generalizes best to unseen data.

The interactive Gradio interface allows users to adjust various parameters, such as the number of data points, noise variance, model architecture (number of layers and neurons per layer), and training epochs. This flexibility enables users to explore the impact of different settings on the regression task and gain a deeper understanding of the concepts involved.

## Technical Documentation

The solution is implemented using Python, TensorFlow for building and training the neural network models, and Plotly for interactive data visualization. The Gradio library is used to create the user interface and enable interactive parameter adjustments.

The main components of the code include:

1. Data generation and noise addition functions.
2. Model definition and training functions.
3. Plotting functions for visualizing data, predictions, and losses.
4. Gradio interface setup with input widgets and output plots.

The code follows a modular structure, with separate functions for each task, making it easy to maintain and extend. The interactive nature of the solution, facilitated by Gradio, allows users to experiment with different settings and observe their effects on the regression task in real-time.
"""

# Gradio Interface
data_settings = [
    gr.Slider(50, 200, step=1, value=100, label="Data Points (N)"),
    gr.Slider(0.01, 0.1, step=0.01, value=0.05, label="Noise Variance (V)"),
    gr.Slider(-3, -1, step=0.1, value=-2.0, label="X Min"),
    gr.Slider(1, 3, step=0.1, value=2.0, label="X Max")
]

model_settings = [
    gr.Slider(1, 5, step=1, value=2, label="Number of Hidden Layers"),
    gr.Slider(50, 200, step=10, value=100, label="Neurons per Hidden Layer"),
    gr.Slider(10, 500, step=10, value=100, label="Epochs (Unnoisy Model)"),
    gr.Slider(100, 500, step=10, value=200, label="Epochs (Best-Fit Model)"),
    gr.Slider(500, 2000, step=10, value=500, label="Epochs (Overfit Model)"),
    gr.Checkbox(value=True, label="Show True Function")
]

outputs = [
    gr.Plot(label="Noiseless Datasets"),
    gr.Plot(label="Noisy Datasets"),
    gr.Plot(label="Unnoisy Model - Train Data", visible=False),
    gr.Textbox(label="Unnoisy Model Losses", visible=False),
    gr.Plot(label="Unnoisy Model - Test Data", visible=False),
    gr.Plot(label="Best-Fit Model - Train Data", visible=False),
    gr.Textbox(label="Best-Fit Model Losses", visible=False),
    gr.Plot(label="Best-Fit Model - Test Data", visible=False),
    gr.Plot(label="Overfit Model - Train Data", visible=False),
    gr.Textbox(label="Overfit Model Losses", visible=False),
    gr.Plot(label="Overfit Model - Test Data", visible=False)
]

def generate_data_wrapper(*args):
    noiseless_plot, noisy_plot = generate_data(*args)
    return [noiseless_plot, noisy_plot] + [None] * 9  # Fülle die restlichen Ausgaben mit None

def train_models_wrapper(*args):
    return main(*args)

demo = gr.Blocks()

with demo:
    gr.Markdown("## Regression with Feed-Forward Neural Network")
    gr.Markdown(discussion)

    with gr.Column():
        gr.Markdown("### Data Settings")
        for input_widget in data_settings:
            input_widget.render()
        generate_data_btn = gr.Button("Generate Data")

    with gr.Column():
        gr.Markdown("### Model Training Settings")
        for input_widget in model_settings:
            input_widget.render()
        train_models_btn = gr.Button("Train Models")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Noiseless Datasets")
            outputs[0].render()
        with gr.Column():
            gr.Markdown("### Noisy Datasets")
            outputs[1].render()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Unnoisy Model - Train Data")
            outputs[2].render()
            outputs[3].render()
        with gr.Column():
            gr.Markdown("### Unnoisy Model - Test Data")
            outputs[4].render()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Best-Fit Model - Train Data")
            outputs[5].render()
            outputs[6].render()
        with gr.Column():
            gr.Markdown("### Best-Fit Model - Test Data")
            outputs[7].render()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Overfit Model - Train Data")
            outputs[8].render()
            outputs[9].render()
        with gr.Column():
            gr.Markdown("### Overfit Model - Test Data")
            outputs[10].render()

    generate_data_btn.click(generate_data_wrapper, inputs=data_settings, outputs=outputs)
    train_models_btn.click(train_models_wrapper, inputs=data_settings + model_settings, outputs=outputs)

demo.launch()
