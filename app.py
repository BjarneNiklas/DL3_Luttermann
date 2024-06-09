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

# Data Generation
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

# Model Definition
def create_model(num_layers, neurons_per_layer):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_dim=1))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Model Training
def train_model(x_train, y_train, epochs, model):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model, history.history['loss'][-1]

# Data Plotting
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

# Prediction Plotting
def plot_predictions(x, y, model, title, show_true_function, x_min, x_max, data_type):
    x_range = np.linspace(x_min, x_max, 1000)
    y_pred = model.predict(x_range).flatten()
    y_pred_points = model.predict(x).flatten()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{data_type} Data', marker=dict(color='blue' if data_type == 'Train' else 'red')))
    fig.add_trace(go.Scatter(x=x, y=y_pred_points, mode='markers', name='Prediction Points', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Prediction Line', line=dict(color='green')))
    if show_true_function:
        y_true = true_function(x_range)
        fig.add_trace(go.Scatter(x=x_range, y=y_true, mode='lines', name='True Function', line=dict(color='orange')))
    fig.update_layout(title=title)
    return fig

# Main Function
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
            unnoisy_plot_train, gr.Row([gr.Textbox(f"Train Loss: {loss_unnoisy_train:.4f}"), gr.Textbox(f"Test Loss: {loss_unnoisy_test:.4f}")]), unnoisy_plot_test,
            best_fit_plot_train, gr.Row([gr.Textbox(f"Train Loss: {loss_best_fit_train:.4f}"), gr.Textbox(f"Test Loss: {loss_best_fit_test:.4f}")]), best_fit_plot_test,
            overfit_plot_train, gr.Row([gr.Textbox(f"Train Loss: {loss_overfit_train:.4f}"), gr.Textbox(f"Test Loss: {loss_overfit_test:.4f}")]), overfit_plot_test)

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
inputs = [
    gr.Slider(50, 200, step=1, value=100, label="Data Points (N)"),
    gr.Slider(0.01, 0.1, step=0.01, value=0.05, label="Noise Variance (V)"),
    gr.Slider(-3, -1, step=0.1, value=-2.0, label="X Min"),
    gr.Slider(1, 3, step=0.1, value=2.0, label="X Max"),
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
    gr.Plot(label="Unnoisy Model - Train Data"),
    gr.Row(),
    gr.Plot(label="Unnoisy Model - Test Data"),
    gr.Plot(label="Best-Fit Model - Train Data"),
    gr.Row(),
    gr.Plot(label="Best-Fit Model - Test Data"),
    gr.Plot(label="Overfit Model - Train Data"),
    gr.Row(),
    gr.Plot(label="Overfit Model - Test Data")
]

def wrapper(*args):
    return main(*args)

def update_true_function(show_true_function):
    N = inputs[0].value
    noise_variance = inputs[1].value
    x_min = inputs[2].value
    x_max = inputs[3].value
    num_layers = inputs[4].value
    neurons_per_layer = inputs[5].value
    epochs_unnoisy = inputs[6].value
    epochs_best_fit = inputs[7].value
    epochs_overfit = inputs[8].value
    return main(N, noise_variance, x_min, x_max, num_layers, neurons_per_layer, epochs_unnoisy, epochs_best_fit, epochs_overfit, show_true_function)

demo = gr.Blocks()

with demo:
    gr.Markdown("## Regression with Feed-Forward Neural Network")
    gr.Markdown(discussion)

    gr.Markdown("### Data Generation")
    with gr.Column():
        for input_widget in inputs[:4]:
            input_widget.render()

    gr.Markdown("### Model Definition")
    with gr.Column():
        for input_widget in inputs[4:6]:
            input_widget.render()

    gr.Markdown("### Model Training")
    with gr.Column():
        for input_widget in inputs[6:9]:
            input_widget.render()

    gr.Markdown("### Visualization")
    with gr.Column():
        inputs[9].render()

    gr.Button("Generate Data and Train Models").click(wrapper, inputs, outputs)

    with gr.Row():
        with gr.Column():
            outputs[0].render()
        with gr.Column():
            outputs[1].render()

    with gr.Row():
        with gr.Column():
            outputs[2].render()
        with gr.Column():
            outputs[3].render()
        with gr.Column():
            outputs[4].render()

    with gr.Row():
        with gr.Column():
            outputs[5].render()
        with gr.Column():
            outputs[6].render()
        with gr.Column():
            outputs[7].render()

    with gr.Row():
        with gr.Column():
            outputs[8].render()
        with gr.Column():
            outputs[9].render()
        with gr.Column():
            outputs[10].render()

    inputs[9].change(fn=update_true_function, inputs=inputs, outputs=[outputs[0], outputs[1], outputs[2], outputs[4], outputs[5], outputs[7], outputs[8], outputs[10]])

demo.launch()
