import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
import gradio as gr

# Step 1: Generate Data
def generate_data(N=100, noise_variance=0.05):
    x = np.random.uniform(-2, 2, N)
    y_true = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    y_noisy = y_true + np.random.normal(0, np.sqrt(noise_variance), N)
    
    # Split into training and test sets
    train_indices = np.random.choice(N, N // 2, replace=False)
    test_indices = list(set(range(N)) - set(train_indices))
    
    x_train, y_train = x[train_indices], y_true[train_indices]
    x_test, y_test = x[test_indices], y_true[test_indices]
    
    y_train_noisy = y_noisy[train_indices]
    y_test_noisy = y_noisy[test_indices]
    
    return (x_train, y_train, y_train_noisy), (x_test, y_test, y_test_noisy)

# Step 2: Build FFNN Model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')
    return model

# Step 3: Train Models
def train_model(x_train, y_train, epochs=200):
    model = build_model()
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

# Step 4: Visualize Results
def plot_data(x_train, y_train, x_test, y_test, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data'))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    return fig

def plot_predictions(x, y_true, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_true, mode='markers', name='True Data'))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Predictions'))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    return fig

# Gradio Interface
def gradio_interface(N=100, noise_variance=0.05, epochs_best_fit=200, epochs_overfit=500):
    (x_train, y_train, y_train_noisy), (x_test, y_test, y_test_noisy) = generate_data(N, noise_variance)
    
    # Train models
    model_clean = train_model(x_train, y_train, epochs=epochs_best_fit)
    model_best_fit = train_model(x_train, y_train_noisy, epochs=epochs_best_fit)
    model_overfit = train_model(x_train, y_train_noisy, epochs=epochs_overfit)
    
    # Predictions
    y_train_pred_clean = model_clean.predict(x_train)
    y_test_pred_clean = model_clean.predict(x_test)
    
    y_train_pred_best = model_best_fit.predict(x_train)
    y_test_pred_best = model_best_fit.predict(x_test)
    
    y_train_pred_overfit = model_overfit.predict(x_train)
    y_test_pred_overfit = model_overfit.predict(x_test)
    
    # Plotting
    fig1 = plot_data(x_train, y_train, x_test, y_test, "Clean Data (Train/Test)")
    fig2 = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, "Noisy Data (Train/Test)")
    
    fig3 = plot_predictions(x_train, y_train, y_train_pred_clean, "Clean Model (Train)")
    fig4 = plot_predictions(x_test, y_test, y_test_pred_clean, "Clean Model (Test)")
    
    fig5 = plot_predictions(x_train, y_train_noisy, y_train_pred_best, "Best Fit Model (Train)")
    fig6 = plot_predictions(x_test, y_test_noisy, y_test_pred_best, "Best Fit Model (Test)")
    
    fig7 = plot_predictions(x_train, y_train_noisy, y_train_pred_overfit, "Overfit Model (Train)")
    fig8 = plot_predictions(x_test, y_test_noisy, y_test_pred_overfit, "Overfit Model (Test)")
    
    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]

# Gradio Layout
with gr.Blocks() as demo:
    with gr.Row():
        N = gr.Number(value=100, label="Number of Data Points")
        noise_variance = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.05, label="Noise Variance")
    with gr.Row():
        epochs_best_fit = gr.Number(value=200, label="Epochs for Best Fit")
        epochs_overfit = gr.Number(value=500, label="Epochs for Overfit")
    with gr.Row():
        generate_button = gr.Button("Generate and Train Models")
    with gr.Row():
        gr.Plot(gradio_interface, [N, noise_variance, epochs_best_fit, epochs_overfit])

    generate_button.click(gradio_interface, inputs=[N, noise_variance, epochs_best_fit, epochs_overfit], outputs=[gr.Plot]*8)

# Documentation
with gr.Blocks() as doc:
    gr.Markdown("# Documentation")
    gr.Markdown("### Technical")
    gr.Markdown("1. **TensorFlow.js**: Used for building and training neural network models.")
    gr.Markdown("2. **Plotly**: Used for visualizing data and predictions.")
    gr.Markdown("3. **Gradio**: Used for building the interactive web interface.")
    
    gr.Markdown("### Implementation")
    gr.Markdown("1. Data is generated based on the given mathematical function with and without noise.")
    gr.Markdown("2. Models are built and trained using TensorFlow.js with specified architecture and parameters.")
    gr.Markdown("3. Results are visualized using Plotly and displayed in a structured layout using Gradio.")

# Launch the Gradio interface and documentation
demo.launch()
doc.launch()
