import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.subplots as sp
import gradio as gr

# Function to generate data
def generate_data(N=100, noise_variance=0.05):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    noise = np.random.normal(0, np.sqrt(noise_variance), N)
    y_noisy = y + noise
    return x, y, y_noisy

# Function to create model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Function to train model
def train_model(x_train, y_train, epochs):
    model = create_model()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

# Function to calculate MSE
def calculate_loss(y_true, y_pred):
    return np.mean((y_true - y_pred.squeeze())**2)

# Function to create plots
def create_plot(x, y, y_noisy, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='True', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y_noisy, mode='markers', name='Noisy', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=x, y=y_pred.squeeze(), mode='lines', name='Predicted', line=dict(color='green')))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    return fig

# Function to train models and generate plots
def train_and_plot(N, noise_variance, epochs_best_fit, epochs_overfit):
    x, y, y_noisy = generate_data(N, noise_variance)
    indices = np.random.permutation(N)
    train_indices, test_indices = indices[:N//2], indices[N//2:]
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    y_noisy_train, y_noisy_test = y_noisy[train_indices], y_noisy[test_indices]

    # Train models
    model_clean = train_model(x_train, y_train, epochs_best_fit)
    model_best_fit = train_model(x_train, y_noisy_train, epochs_best_fit)
    model_overfit = train_model(x_train, y_noisy_train, epochs_overfit)

    # Predictions
    y_pred_clean_train = model_clean.predict(x_train)
    y_pred_clean_test = model_clean.predict(x_test)
    y_pred_best_fit_train = model_best_fit.predict(x_train)
    y_pred_best_fit_test = model_best_fit.predict(x_test)
    y_pred_overfit_train = model_overfit.predict(x_train)
    y_pred_overfit_test = model_overfit.predict(x_test)

    # Calculate loss
    loss_clean_train = calculate_loss(y_train, y_pred_clean_train)
    loss_clean_test = calculate_loss(y_test, y_pred_clean_test)
    loss_best_fit_train = calculate_loss(y_noisy_train, y_pred_best_fit_train)
    loss_best_fit_test = calculate_loss(y_noisy_test, y_pred_best_fit_test)
    loss_overfit_train = calculate_loss(y_noisy_train, y_pred_overfit_train)
    loss_overfit_test = calculate_loss(y_noisy_test, y_pred_overfit_test)

    # Create plots
    fig1 = create_plot(x, y, y_noisy, y, "Data without noise")
    fig2 = create_plot(x, y, y_noisy, y_noisy, "Data with noise")
    fig3 = create_plot(x_train, y_train, y_noisy_train, y_pred_clean_train, f"Model without noise (Train) - MSE: {loss_clean_train:.4f}")
    fig4 = create_plot(x_test, y_test, y_noisy_test, y_pred_clean_test, f"Model without noise (Test) - MSE: {loss_clean_test:.4f}")
    fig5 = create_plot(x_train, y_train, y_noisy_train, y_pred_best_fit_train, f"Best-Fit Model (Train) - MSE: {loss_best_fit_train:.4f}")
    fig6 = create_plot(x_test, y_test, y_noisy_test, y_pred_best_fit_test, f"Best-Fit Model (Test) - MSE: {loss_best_fit_test:.4f}")
    fig7 = create_plot(x_train, y_train, y_noisy_train, y_pred_overfit_train, f"Overfit Model (Train) - MSE: {loss_overfit_train:.4f}")
    fig8 = create_plot(x_test, y_test, y_noisy_test, y_pred_overfit_test, f"Overfit Model (Test) - MSE: {loss_overfit_test:.4f}")

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# FFNN Regression with Noise and Overfitting")
        with gr.Row():
            N = gr.Slider(50, 200, value=100, step=1, label="Number of Data Points")
            noise_variance = gr.Slider(0.01, 0.1, value=0.05, step=0.01, label="Noise Variance")
            epochs_best_fit = gr.Slider(50, 500, value=200, step=10, label="Epochs (Best-Fit)")
            epochs_overfit = gr.Slider(50, 500, value=500, step=10, label="Epochs (Overfit)")
        
        plot1 = gr.Plot(label="Data without noise")
        plot2 = gr.Plot(label="Data with noise")
        plot3 = gr.Plot(label="Model without noise (Train)")
        plot4 = gr.Plot(label="Model without noise (Test)")
        plot5 = gr.Plot(label="Best-Fit Model (Train)")
        plot6 = gr.Plot(label="Best-Fit Model (Test)")
        plot7 = gr.Plot(label="Overfit Model (Train)")
        plot8 = gr.Plot(label="Overfit Model (Test)")

        def update_plots(N, noise_variance, epochs_best_fit, epochs_overfit):
            fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = train_and_plot(N, noise_variance, epochs_best_fit, epochs_overfit)
            return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8

        update_btn = gr.Button("Train and Plot")
        update_btn.click(
            update_plots,
            inputs=[N, noise_variance, epochs_best_fit, epochs_overfit],
            outputs=[plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8]
        )
    
    return demo

demo = gradio_interface()
demo.launch()