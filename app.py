
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objs as go
import gradio as gr

# Funktion zur Erzeugung der Daten
def generate_data(N, noise_var=0):
    np.random.seed(0)
    x = np.random.uniform(-2, 2, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    if noise_var > 0:
        y += np.random.normal(0, noise_var, y.shape)
    return x, y

# Funktion zur Modelldefinition
def build_model():
    model = Sequential([
        Dense(100, input_dim=1, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Funktion zur Visualisierung
def plot_data(x_train, y_train, x_test, y_test, y_pred_train, y_pred_test, title):
    trace1 = go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data')
    trace2 = go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data')
    trace3 = go.Scatter(x=x_train, y=y_pred_train, mode='lines', name='Train Prediction')
    trace4 = go.Scatter(x=x_test, y=y_pred_test, mode='lines', name='Test Prediction')
    
    layout = go.Layout(title=title, xaxis={'title': 'x'}, yaxis={'title': 'y'})
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    return fig

# Gradio App
def gradio_app(x_min, x_max, noise_var, epochs_best_fit, epochs_overfit):
    # Erzeuge Daten
    x, y = generate_data(100)
    x_train, y_train = x[:50], y[:50]
    x_test, y_test = x[50:], y[50:]
    y_train_noisy = y_train + np.random.normal(0, noise_var, y_train.shape)
    y_test_noisy = y_test + np.random.normal(0, noise_var, y_test.shape)
    
    # Trainiere Modelle
    model_clean = build_model()
    model_clean.fit(x_train, y_train, epochs=epochs_best_fit, batch_size=32, verbose=0)
    y_pred_train_clean = model_clean.predict(x_train).flatten()
    y_pred_test_clean = model_clean.predict(x_test).flatten()
    
    model_best_fit = build_model()
    model_best_fit.fit(x_train, y_train_noisy, epochs=epochs_best_fit, batch_size=32, verbose=0)
    y_pred_train_best_fit = model_best_fit.predict(x_train).flatten()
    y_pred_test_best_fit = model_best_fit.predict(x_test).flatten()
    
    model_overfit = build_model()
    model_overfit.fit(x_train, y_train_noisy, epochs=epochs_overfit, batch_size=32, verbose=0)
    y_pred_train_overfit = model_overfit.predict(x_train).flatten()
    y_pred_test_overfit = model_overfit.predict(x_test).flatten()
    
    # Erstelle Plots
    fig_clean_train = plot_data(x_train, y_train, x_test, y_test, y_pred_train_clean, y_pred_test_clean, 'Clean Data (Train)')
    fig_clean_test = plot_data(x_train, y_train, x_test, y_test, y_pred_train_clean, y_pred_test_clean, 'Clean Data (Test)')
    fig_best_fit_train = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, y_pred_train_best_fit, y_pred_test_best_fit, 'Best Fit (Train)')
    fig_best_fit_test = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, y_pred_train_best_fit, y_pred_test_best_fit, 'Best Fit (Test)')
    fig_overfit_train = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, y_pred_train_overfit, y_pred_test_overfit, 'Overfit (Train)')
    fig_overfit_test = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, y_pred_train_overfit, y_pred_test_overfit, 'Overfit (Test)')
    
    return fig_clean_train, fig_clean_test, fig_best_fit_train, fig_best_fit_test, fig_overfit_train, fig_overfit_test

# Gradio Interface
iface = gr.Interface(
    fn=gradio_app,
    inputs=[
        gr.Slider(-2, 2, step=0.1, label="X min", value=-2),
        gr.Slider(-2, 2, step=0.1, label="X max", value=2),
        gr.Slider(0, 0.1, step=0.01, label="Noise Variance", value=0.05),
        gr.Slider(1, 1000, step=1, label="Best Fit Epochs", value=200),
        gr.Slider(1, 1000, step=1, label="Overfit Epochs", value=500)
    ],
    outputs=[
        gr.Plot(label="Clean Data (Train)"),
        gr.Plot(label="Clean Data (Test)"),
        gr.Plot(label="Best Fit (Train)"),
        gr.Plot(label="Best Fit (Test)"),
        gr.Plot(label="Overfit (Train)"),
        gr.Plot(label="Overfit (Test)")
    ],
    layout="grid"
)

iface.launch()



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