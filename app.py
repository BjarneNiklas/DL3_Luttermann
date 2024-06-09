import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.subplots as sp
import gradio as gr

# Generierung der Daten
def generate_data(N=100, x_min=-2, x_max=2):
    x = np.random.uniform(x_min, x_max, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    return x, y

# Hinzufügen von Rauschen
def add_noise(y, variance=0.05):
    noise = np.random.normal(0, np.sqrt(variance), y.shape)
    return y + noise

# Modell definieren und trainieren
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Visualisierung der Ergebnisse
def plot_results(N, x_min, x_max, epochs_best_fit, epochs_over_fit):
    # Daten generieren
    x, y = generate_data(N, x_min, x_max)
    x_train, x_test = x[:N//2], x[N//2:]
    y_train, y_test = y[:N//2], y[N//2:]

    # Daten mit Rauschen
    y_train_noisy = add_noise(y_train)
    y_test_noisy = add_noise(y_test)

    # Training des Modells ohne Rauschen
    model_unverrauscht = create_model()
    model_unverrauscht.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Training des Modells mit Rauschen (Best-Fit)
    model_best_fit = create_model()
    model_best_fit.fit(x_train, y_train_noisy, epochs=epochs_best_fit, batch_size=32, verbose=0)

    # Training des Modells mit Rauschen (Over-Fit)
    model_over_fit = create_model()
    model_over_fit.fit(x_train, y_train_noisy, epochs=epochs_over_fit, batch_size=32, verbose=0)

    # Originalfunktion
    x_line = np.linspace(x_min, x_max, 100)
    y_line = 0.5 * (x_line + 0.8) * (x_line + 1.8) * (x_line - 0.2) * (x_line - 0.3) * (x_line - 1.9) + 1

    # Vorhersagen
    y_train_pred_unverrauscht = model_unverrauscht.predict(x_train).flatten()
    y_test_pred_unverrauscht = model_unverrauscht.predict(x_test).flatten()

    y_train_pred_best_fit = model_best_fit.predict(x_train).flatten()
    y_test_pred_best_fit = model_best_fit.predict(x_test).flatten()

    y_train_pred_over_fit = model_over_fit.predict(x_train).flatten()
    y_test_pred_over_fit = model_over_fit.predict(x_test).flatten()

    # Plotly Visualisierung
    fig = sp.make_subplots(rows=4, cols=2, subplot_titles=(
        'Train Data (No Noise)', 'Train Data (With Noise)',
        'Test Data (No Noise)', 'Test Data (With Noise)',
        'Train Predictions (Unverrauscht)', 'Train Predictions (With Noise)',
        'Test Predictions (Unverrauscht)', 'Test Predictions (With Noise)'
    ))

    # R1: Die Datensätze
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data (No Noise)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data (No Noise)'), row=2, col=1)

    fig.add_trace(go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Train Data (With Noise)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Test Data (With Noise)'), row=2, col=2)

    # R2: Die Vorhersage des Modells, das ohne Rauschen trainiert wurde
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_unverrauscht, mode='markers', name='Train Predictions (Unverrauscht)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_unverrauscht, mode='markers', name='Test Predictions (Unverrauscht)'), row=4, col=1)

    # R3: Die Vorhersage Ihres besten Modells
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_best_fit, mode='markers', name='Train Predictions (Best-Fit)'), row=3, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_best_fit, mode='markers', name='Test Predictions (Best-Fit)'), row=4, col=2)

    # R4: Die Vorhersage Ihres Overfit-Modells
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_over_fit, mode='markers', name='Train Predictions (Over-Fit)'), row=3, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_over_fit, mode='markers', name='Test Predictions (Over-Fit)'), row=4, col=2)

    fig.update_layout(height=1000, width=1200, title_text="Regression mit FFNN und TensorFlow.js")

    # Loss berechnen und anzeigen
    loss_train_unverrauscht = model_unverrauscht.evaluate(x_train, y_train, verbose=0)
    loss_test_unverrauscht = model_unverrauscht.evaluate(x_test, y_test, verbose=0)
    loss_train_best_fit = model_best_fit.evaluate(x_train, y_train_noisy, verbose=0)
    loss_test_best_fit = model_best_fit.evaluate(x_test, y_test_noisy, verbose=0)
    loss_train_over_fit = model_over_fit.evaluate(x_train, y_train_noisy, verbose=0)
    loss_test_over_fit = model_over_fit.evaluate(x_test, y_test_noisy, verbose=0)

    fig.add_annotation(x=0, y=0, xref="paper", yref="paper", text=f"Loss (Unverrauscht): Train={loss_train_unverrauscht:.4f}, Test={loss_test_unverrauscht:.4f}", showarrow=False)
    fig.add_annotation(x=0, y=-0.05, xref="paper", yref="paper", text=f"Loss (Best-Fit): Train={loss_train_best_fit:.4f}, Test={loss_test_best_fit:.4f}", showarrow=False)
    fig.add_annotation(x=0, y=-0.1, xref="paper", yref="paper", text=f"Loss (Over-Fit): Train={loss_train_over_fit:.4f}, Test={loss_test_over_fit:.4f}", showarrow=False)

    return fig

# Gradio Interface
data_settings = gr.Group([
    gr.Slider(1, 200, value=100, step=1, label="Number of Data Points"),
    gr.Slider(-3, 3, value=-2, label="X Min"),
    gr.Slider(-3, 3, value=2, label="X Max")
])

model_settings = gr.Group([
    gr.Slider(1, 1000, value=200, step=1, label="Epochs (Best-Fit)"),
    gr.Slider(1, 2000, value=500, step=1, label="Epochs (Over-Fit)")
])

interface = gr.Interface(fn=plot_results, inputs=[data_settings, model_settings], outputs=gr.Plot(), title="Regression mit FFNN und TensorFlow.js")

interface.launch()
