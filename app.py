import numpy as np

# Funktion zur Erzeugung der Daten
def generate_data(N=100, noise_variance=0.05):
    np.random.seed(42)  # Für Reproduzierbarkeit
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    
    # Aufteilen in Trainings- und Testdaten
    indices = np.random.permutation(N)
    train_idx, test_idx = indices[:N//2], indices[N//2:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    
    # Hinzufügen von Rauschen
    y_train_noisy = y_train + np.random.normal(0, np.sqrt(noise_variance), N//2)
    y_test_noisy = y_test + np.random.normal(0, np.sqrt(noise_variance), N//2)
    
    return (x_train, y_train, y_train_noisy), (x_test, y_test, y_test_noisy)

# Daten erzeugen
train_data, test_data = generate_data()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Modellarchitektur definieren
def create_model():
    model = Sequential([
        Dense(100, activation='relu', input_shape=(1,)),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# Modelle erstellen
model_unverrauscht = create_model()
model_best_fit = create_model()
model_overfit = create_model()

# Modelle trainieren
x_train, y_train, y_train_noisy = train_data
x_test, y_test, y_test_noisy = test_data

# Modell ohne Rauschen trainieren
history_unverrauscht = model_unverrauscht.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test), verbose=0)

# Modell mit Rauschen (Best-Fit) trainieren
history_best_fit = model_best_fit.fit(x_train, y_train_noisy, epochs=200, batch_size=32, validation_data=(x_test, y_test_noisy), verbose=0)

# Modell mit Rauschen (Over-Fit) trainieren
history_overfit = model_overfit.fit(x_train, y_train_noisy, epochs=500, batch_size=32, validation_data=(x_test, y_test_noisy), verbose=0)


import plotly.graph_objs as go
import gradio as gr

# Visualisierungsfunktion erstellen
def plot_results():
    # Datenplot
    fig_data = go.Figure()
    fig_data.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train (unverrauscht)', marker=dict(color='blue')))
    fig_data.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test (unverrauscht)', marker=dict(color='red')))
    fig_data.add_trace(go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Train (verrauscht)', marker=dict(color='green')))
    fig_data.add_trace(go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Test (verrauscht)', marker=dict(color='orange')))
    fig_data.update_layout(title='Daten', xaxis_title='x', yaxis_title='y', height=400)

    # Vorhersagenplot unverrauscht
    fig_pred_unverrauscht_train = go.Figure()
    fig_pred_unverrauscht_train.add_trace(go.Scatter(x=x_train, y=model_unverrauscht.predict(x_train), mode='markers', name='Train Predictions', marker=dict(color='blue')))
    fig_pred_unverrauscht_train.update_layout(title='Vorhersagen (Unverrauscht) - Training', xaxis_title='x', yaxis_title='y', height=400)
    
    fig_pred_unverrauscht_test = go.Figure()
    fig_pred_unverrauscht_test.add_trace(go.Scatter(x=x_test, y=model_unverrauscht.predict(x_test), mode='markers', name='Test Predictions', marker=dict(color='red')))
    fig_pred_unverrauscht_test.update_layout(title='Vorhersagen (Unverrauscht) - Test', xaxis_title='x', yaxis_title='y', height=400)
    
    # Vorhersagenplot Best-Fit
    fig_pred_best_fit_train = go.Figure()
    fig_pred_best_fit_train.add_trace(go.Scatter(x=x_train, y=model_best_fit.predict(x_train), mode='markers', name='Train Predictions', marker=dict(color='blue')))
    fig_pred_best_fit_train.update_layout(title='Vorhersagen (Best-Fit) - Training', xaxis_title='x', yaxis_title='y', height=400)
    
    fig_pred_best_fit_test = go.Figure()
    fig_pred_best_fit_test.add_trace(go.Scatter(x=x_test, y=model_best_fit.predict(x_test), mode='markers', name='Test Predictions', marker=dict(color='red')))
    fig_pred_best_fit_test.update_layout(title='Vorhersagen (Best-Fit) - Test', xaxis_title='x', yaxis_title='y', height=400)
    
    # Vorhersagenplot Over-Fit
    fig_pred_overfit_train = go.Figure()
    fig_pred_overfit_train.add_trace(go.Scatter(x=x_train, y=model_overfit.predict(x_train), mode='markers', name='Train Predictions', marker=dict(color='blue')))
    fig_pred_overfit_train.update_layout(title='Vorhersagen (Over-Fit) - Training', xaxis_title='x', yaxis_title='y', height=400)
    
    fig_pred_overfit_test = go.Figure()
    fig_pred_overfit_test.add_trace(go.Scatter(x=x_test, y=model_overfit.predict(x_test), mode='markers', name='Test Predictions', marker=dict(color='red')))
    fig_pred_overfit_test.update_layout(title='Vorhersagen (Over-Fit) - Test', xaxis_title='x', yaxis_title='y', height=400)
    
    return [fig_data, fig_pred_unverrauscht_train, fig_pred_unverrauscht_test, fig_pred_best_fit_train, fig_pred_best_fit_test, fig_pred_overfit_train, fig_pred_overfit_test]

# Gradio Interface
def build_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## Regression mit FFNN")
        with gr.Row():
            with gr.Column():
                fig_data = gr.Plot(plot_results()[0])
                fig_pred_unverrauscht_train = gr.Plot(plot_results()[1])
                fig_pred_best_fit_train = gr.Plot(plot_results()[3])
                fig_pred_overfit_train = gr.Plot(plot_results()[5])
            with gr.Column():
                fig_pred_unverrauscht_test = gr.Plot(plot_results()[2])
                fig_pred_best_fit_test = gr.Plot(plot_results()[4])
                fig_pred_overfit_test = gr.Plot(plot_results()[6])
                
        with gr.Row():
            gr.Markdown("## Loss (MSE)")
            gr.Textbox(f"Loss (Unverrauscht) - Train: {history_unverrauscht.history['loss'][-1]:.4f}, Test: {history_unverrauscht.history['val_loss'][-1]:.4f}")
            gr.Textbox(f"Loss (Best-Fit) - Train: {history_best_fit.history['loss'][-1]:.4f}, Test: {history_best_fit.history['val_loss'][-1]:.4f}")
            gr.Textbox(f"Loss (Over-Fit) - Train: {history_overfit.history['loss'][-1]:.4f}, Test: {history_overfit.history['val_loss'][-1]:.4f}")

    return demo

# Interface starten
demo = build_interface()
demo.launch()



"""import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
import gradio as gr

# Funktion zur Daten erzeugung
def generate_data(N, noise_var=0):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    noise = np.random.normal(0, noise_var, y.shape)
    y_noisy = y + noise
    return x, y, y_noisy

# Modell definieren
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

# Vorhersagen machen
def predict(model, x):
    return model.predict(x).flatten()

# Verlust berechnen
def calculate_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Visualisierung
def plot_data(x, y, y_noisy, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Unverrauscht'))
    fig.add_trace(go.Scatter(x=x, y=y_noisy, mode='markers', name='Verrauscht'))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    return fig

def plot_predictions(x_train, y_train, y_pred_train, x_test, y_test, y_pred_test, title_train, title_test):
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='True'))
    fig_train.add_trace(go.Scatter(x=x_train, y=y_pred_train, mode='markers', name='Predicted'))
    fig_train.update_layout(title=title_train, xaxis_title='x', yaxis_title='y')

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='True'))
    fig_test.add_trace(go.Scatter(x=x_test, y=y_pred_test, mode='markers', name='Predicted'))
    fig_test.update_layout(title=title_test, xaxis_title='x', yaxis_title='y')
    
    return fig_train, fig_test

# Gradio Interface
def visualize(N, noise_var, epochs_best_fit, epochs_overfit):
    x, y, y_noisy = generate_data(N, noise_var)
    train_indices = np.random.choice(np.arange(N), size=N//2, replace=False)
    test_indices = np.setdiff1d(np.arange(N), train_indices)

    x_train, y_train, y_train_noisy = x[train_indices], y[train_indices], y_noisy[train_indices]
    x_test, y_test, y_test_noisy = x[test_indices], y[test_indices], y_noisy[test_indices]

    model_unnoisy = build_model()
    model_unnoisy.fit(x_train, y_train, epochs=200, batch_size=32, verbose=0)

    model_best_fit = build_model()
    model_best_fit.fit(x_train, y_train_noisy, epochs=epochs_best_fit, batch_size=32, verbose=0)

    model_overfit = build_model()
    model_overfit.fit(x_train, y_train_noisy, epochs=epochs_overfit, batch_size=32, verbose=0)

    y_pred_train_unnoisy = predict(model_unnoisy, x_train)
    y_pred_test_unnoisy = predict(model_unnoisy, x_test)
    y_pred_train_best_fit = predict(model_best_fit, x_train)
    y_pred_test_best_fit = predict(model_best_fit, x_test)
    y_pred_train_overfit = predict(model_overfit, x_train)
    y_pred_test_overfit = predict(model_overfit, x_test)

    loss_train_unnoisy = calculate_loss(y_train, y_pred_train_unnoisy)
    loss_test_unnoisy = calculate_loss(y_test, y_pred_test_unnoisy)
    loss_train_best_fit = calculate_loss(y_train_noisy, y_pred_train_best_fit)
    loss_test_best_fit = calculate_loss(y_test_noisy, y_pred_test_best_fit)
    loss_train_overfit = calculate_loss(y_train_noisy, y_pred_train_overfit)
    loss_test_overfit = calculate_loss(y_test_noisy, y_pred_test_overfit)

    fig_data_unnoisy = plot_data(x, y, y, "Daten ohne Rauschen")
    fig_data_noisy = plot_data(x, y, y_noisy, "Daten mit Rauschen")

    fig_train_unnoisy, fig_test_unnoisy = plot_predictions(
        x_train, y_train, y_pred_train_unnoisy,
        x_test, y_test, y_pred_test_unnoisy,
        "Vorhersage ohne Rauschen - Training",
        "Vorhersage ohne Rauschen - Test"
    )

    fig_train_best_fit, fig_test_best_fit = plot_predictions(
        x_train, y_train_noisy, y_pred_train_best_fit,
        x_test, y_test_noisy, y_pred_test_best_fit,
        f"Vorhersage Best-Fit - Training (Loss: {loss_train_best_fit:.4f})",
        f"Vorhersage Best-Fit - Test (Loss: {loss_test_best_fit:.4f})"
    )

    fig_train_overfit, fig_test_overfit = plot_predictions(
        x_train, y_train_noisy, y_pred_train_overfit,
        x_test, y_test_noisy, y_pred_test_overfit,
        f"Vorhersage Overfit - Training (Loss: {loss_train_overfit:.4f})",
        f"Vorhersage Overfit - Test (Loss: {loss_test_overfit:.4f})"
    )

    return fig_data_unnoisy, fig_data_noisy, fig_train_unnoisy, fig_test_unnoisy, fig_train_best_fit, fig_test_best_fit, fig_train_overfit, fig_test_overfit

# Diskussion und Dokumentation
discussion = """
# Diskussion und Dokumentation
"""
## Aufgabenstellung
Wir sollten ein Feed-Forward Neural Network (FFNN) zur Regression einer unbekannten Funktion y(x) implementieren. Die Daten wurden verrauscht, um realistische Bedingungen zu simulieren. Zwei Modelle wurden trainiert: eins zur Vermeidung von Overfitting und eins zur Demonstration von Overfitting.

## Ergebnisse
1. **Modell ohne Rauschen:** Das Modell hat die unverrauschten Daten sehr gut gelernt und zeigt sowohl auf den Trainings- als auch auf den Testdaten einen geringen Loss.
2. **Best-Fit Modell:** Das Modell wurde mit verrauschten Daten trainiert und zeigt einen guten Kompromiss zwischen Trainings- und Test-Loss. Es generalisiert gut auf den Testdaten.
3. **Overfit Modell:** Dieses Modell wurde über eine längere Anzahl von Epochen trainiert und zeigt deutliches Overfitting. Der Trainings-Loss ist deutlich geringer als der Test-Loss.

## Fazit
Durch die Analyse der verschiedenen Modelle konnten wir die Auswirkungen von Overfitting und die Bedeutung der richtigen Anzahl von Trainingsepochen demonstrieren. Die Verwendung von Gradio und Plotly ermöglichte eine interaktive und anschauliche Darstellung der Ergebnisse.
"""
"""
# Gradio Interface erstellen
interface = gr.Interface(
    fn=visualize,
    inputs=[
        gr.Slider(minimum=50, maximum=500, step=50, value=100, label="Anzahl der Datenpunkte"),
        gr.Slider(minimum=0, maximum=0.1, step=0.01, value=0.05, label="Varianz des Rauschens"),
        gr.Slider(minimum=50, maximum=500, step=50, value=200, label="Epochen für Best-Fit Modell"),
        gr.Slider(minimum=50, maximum=500, step=50, value=500, label="Epochen für Overfit Modell")
    ],
    outputs=[
        gr.Plot(label="Daten ohne Rauschen"),
        gr.Plot(label="Daten mit Rauschen"),
        gr.Plot(label="Vorhersage ohne Rauschen - Training"),
        gr.Plot(label="Vorhersage ohne Rauschen - Test"),
        gr.Plot(label="Vorhersage Best-Fit - Training"),
        gr.Plot(label="Vorhersage Best-Fit - Test"),
        gr.Plot(label="Vorhersage Overfit - Training"),
        gr.Plot(label="Vorhersage Overfit - Test"),
    ],
    title="Regression mit Feed-Forward Neural Network (FFNN)",
    description=discussion
)

# Interface starten
interface.launch()"""

"""import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.subplots as sp
import gradio as gr

# Generierung der Daten
def generate_data(N=100):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    return x, y

# Hinzufügen von Rauschen
def add_noise(y, variance=0.05):
    noise = np.random.normal(0, np.sqrt(variance), y.shape)
    return y + noise

# Daten generieren
x, y = generate_data()
x_train, x_test = x[:50], x[50:]
y_train, y_test = y[:50], y[50:]

# Daten mit Rauschen
y_train_noisy = add_noise(y_train)
y_test_noisy = add_noise(y_test)

# Modell definieren und trainieren
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Training des Modells ohne Rauschen
model_unverrauscht = create_model()
model_unverrauscht.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Best-Fit)
model_best_fit = create_model()
model_best_fit.fit(x_train, y_train_noisy, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Over-Fit)
model_over_fit = create_model()
model_over_fit.fit(x_train, y_train_noisy, epochs=1000, batch_size=32, verbose=0)

# Visualisierung der Ergebnisse
def plot_results():
    # Originalfunktion
    x_line = np.linspace(-2, 2, 100)
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
    
    return fig

# Gradio Interface
def interactive_plot():
    fig = plot_results()
    return fig

interface = gr.Interface(fn=interactive_plot, inputs=[], outputs=gr.Plot())

interface.launch()"""



"""import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import gradio as gr

# Generierung der Daten
def generate_data(N=100):
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    return x, y

# Hinzufügen von Rauschen
def add_noise(y, variance=0.05):
    noise = np.random.normal(0, np.sqrt(variance), y.shape)
    return y + noise

# Daten generieren
x, y = generate_data()
x_train, x_test = x[:50], x[50:]
y_train, y_test = y[:50], y[50:]

# Daten mit Rauschen
y_train_noisy = add_noise(y_train)
y_test_noisy = add_noise(y_test)

# Modell definieren und trainieren
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Training des Modells ohne Rauschen
model_unverrauscht = create_model()
model_unverrauscht.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Best-Fit)
model_best_fit = create_model()
model_best_fit.fit(x_train, y_train_noisy, epochs=100, batch_size=32, verbose=0)

# Training des Modells mit Rauschen (Over-Fit)
model_over_fit = create_model()
model_over_fit.fit(x_train, y_train_noisy, epochs=1000, batch_size=32, verbose=0)

# Visualisierung der Ergebnisse
def plot_results(x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy, model_unverrauscht, model_best_fit, model_over_fit):
    # Originalfunktion
    x_line = np.linspace(-2, 2, 100)
    y_line = 0.5 * (x_line + 0.8) * (x_line + 1.8) * (x_line - 0.2) * (x_line - 0.3) * (x_line - 1.9) + 1

    # Vorhersagen
    y_train_pred_unverrauscht = model_unverrauscht.predict(x_train).flatten()
    y_test_pred_unverrauscht = model_unverrauscht.predict(x_test).flatten()

    y_train_pred_best_fit = model_best_fit.predict(x_train).flatten()
    y_test_pred_best_fit = model_best_fit.predict(x_test).flatten()

    y_train_pred_over_fit = model_over_fit.predict(x_train).flatten()
    y_test_pred_over_fit = model_over_fit.predict(x_test).flatten()

    # Plotly Visualisierung
    fig = go.Figure()

    # Originalfunktion plot
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Original Function'))

    # Noisy training data plot
    fig.add_trace(go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Noisy Train Data'))

    # Noisy test data plot
    fig.add_trace(go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Noisy Test Data'))

    # Train predictions plot
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_unverrauscht, mode='markers', name='Train Predictions (Unverrauscht)'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_unverrauscht, mode='markers', name='Test Predictions (Unverrauscht)'))

    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_best_fit, mode='markers', name='Train Predictions (Best-Fit)'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_best_fit, mode='markers', name='Test Predictions (Best-Fit)'))

    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_over_fit, mode='markers', name='Train Predictions (Over-Fit)'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_over_fit, mode='markers', name='Test Predictions (Over-Fit)'))

    fig.update_layout(title='Regression mit FFNN und TensorFlow.js',
                      xaxis_title='x',
                      yaxis_title='y',
                      legend_title='Legend')
    
    return fig

# Gradio Interface
def interactive_plot():
    fig = plot_results(x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy, model_unverrauscht, model_best_fit, model_over_fit)
    return fig

interface = gr.Interface(fn=interactive_plot, inputs=[], outputs=gr.Plot())

interface.launch()
"""