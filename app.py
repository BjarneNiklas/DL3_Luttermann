import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import gradio as gr

# Funktion zur Generierung der Daten
def generate_data(N=100, x_min=-2, x_max=2):
    x = np.random.uniform(x_min, x_max, N)
    y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    return x, y

# Funktion zum Hinzufügen von Rauschen
def add_noise(y, variance=0.05):
    noise = np.random.normal(0, np.sqrt(variance), y.shape)
    return y + noise

# Funktion zur Vorbereitung der Trainings- und Testdaten
def prepare_data(N, variance, x_min, x_max):
    x, y = generate_data(N, x_min, x_max)
    x_train, x_test = x[:N//2], x[N//2:]
    y_train, y_test = y[:N//2], y[N//2:]
    y_train_noisy = add_noise(y_train, variance)
    y_test_noisy = add_noise(y_test, variance)
    return x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy

# Funktion zur Erstellung des Modells
def create_model(layers, neurons, learning_rate):
    model = tf.keras.Sequential()
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Funktion zum Trainieren der Modelle
def train_models(x_train, y_train, y_train_noisy, epochs, epochs_overfit, neurons, layers, learning_rate, batch_size):
    # Modell ohne Rauschen
    model_unverrauscht = create_model(layers, neurons, learning_rate)
    model_unverrauscht.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Modell mit Rauschen (Best-Fit)
    model_best_fit = create_model(layers, neurons, learning_rate)
    model_best_fit.fit(x_train, y_train_noisy, epochs=epochs, batch_size=batch_size, verbose=0)

    # Modell mit Rauschen (Over-Fit)
    model_over_fit = create_model(layers, neurons, learning_rate)
    model_over_fit.fit(x_train, y_train_noisy, epochs=epochs_overfit, batch_size=batch_size, verbose=0)

    return model_unverrauscht, model_best_fit, model_over_fit

# Funktion zur Visualisierung der Ergebnisse
def plot_results(x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy, model_unverrauscht, model_best_fit, model_over_fit):
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

    # Verlustmetriken
    train_loss_unverrauscht = model_unverrauscht.evaluate(x_train, y_train, verbose=0)
    test_loss_unverrauscht = model_unverrauscht.evaluate(x_test, y_test, verbose=0)
    train_loss_best_fit = model_best_fit.evaluate(x_train, y_train_noisy, verbose=0)
    test_loss_best_fit = model_best_fit.evaluate(x_test, y_test_noisy, verbose=0)
    train_loss_over_fit = model_over_fit.evaluate(x_train, y_train_noisy, verbose=0)
    test_loss_over_fit = model_over_fit.evaluate(x_test, y_test_noisy, verbose=0)

    # Plotly Visualisierung
    fig = make_subplots(rows=4, cols=2, subplot_titles=[
        'Training Data', 'Training Data (With Noise)',
        'Test Data', 'Test Data (With Noise)',
        'Training Predictions', 'Training Predictions (With Noise)',
        'Test Predictions', 'Test Predictions (With Noise)'
    ])

    # Zeile 1: Trainingsdaten und Testdaten
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Training Data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Training Data (With Noise)'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Test Data (With Noise)'), row=2, col=2)

    # Zeile 2: Vorhersagen auf Trainingsdaten
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_unverrauscht, mode='lines', name='Unverrauscht'), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_best_fit, mode='lines', name='Best-Fit'), row=3, col=2)
    fig.add_trace(go.Scatter(x=x_train, y=y_train_pred_over_fit, mode='lines', name='Over-Fit'), row=3, col=2)

    # Zeile 3: Vorhersagen auf Testdaten
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_unverrauscht, mode='lines', name='Unverrauscht'), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_best_fit, mode='lines', name='Best-Fit'), row=4, col=2)
    fig.add_trace(go.Scatter(x=x_test, y=y_test_pred_over_fit, mode='lines', name='Over-Fit'), row=4, col=2)

    # Verlustmetriken als Annotationen
    fig.add_annotation(x=0.5, y=1.15, xref='paper', yref='paper',
                       text=f"Train Loss (Unverrauscht): {train_loss_unverrauscht:.4f}<br>Test Loss (Unverrauscht): {test_loss_unverrauscht:.4f}",
                       showarrow=False, row=1, col=1)
    fig.add_annotation(x=0.5, y=1.15, xref='paper', yref='paper',
                       text=f"Train Loss (Best-Fit): {train_loss_best_fit:.4f}<br>Test Loss (Best-Fit): {test_loss_best_fit:.4f}",
                       showarrow=False, row=1, col=2)
    fig.add_annotation(x=0.5, y=1.15, xref='paper', yref='paper',
                       text=f"Train Loss (Over-Fit): {train_loss_over_fit:.4f}<br>Test Loss (Over-Fit): {test_loss_over_fit:.4f}",
                       showarrow=False, row=2, col=2)

    fig.update_layout(height=1000, width=1200, title_text="Regression mit Feed-Forward Neural Network")
    return fig

# Gradio Interface
def interactive_plot(N, variance, x_min, x_max, layers, neurons, learning_rate, epochs, epochs_overfit, batch_size):
    try:
        x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy = prepare_data(N, variance, x_min, x_max)
        model_unverrauscht, model_best_fit, model_over_fit = train_models(x_train, y_train, y_train_noisy, epochs, epochs_overfit, neurons, layers, learning_rate, batch_size)
        fig = plot_results(x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy, model_unverrauscht, model_best_fit, model_over_fit)
        return gr.Plot(fig)
    except Exception as e:
        return gr.Error(f"Ein Fehler ist aufgetreten: {str(e)}")

# Gradio Interface Konfiguration
data_inputs = [
    gr.Slider(50, 200, step=1, value=100, label="Anzahl Datenpunkte (N)"),
    gr.Slider(0.01, 0.1, step=0.01, value=0.05, label="Rauschvarianz (V)"),
    gr.Slider(-3.0, 3.0, step=0.1, value=-2.0, label="Minimaler X-Wert"),
    gr.Slider(-3.0, 3.0, step=0.1, value=2.0, label="Maximaler X-Wert")
]

model_inputs = [
    gr.Slider(1, 5, step=1, value=2, label="Anzahl versteckter Schichten"),
    gr.Slider(5, 100, step=5, value=20, label="Neuronen pro Schicht"),
    gr.Slider(0.0001, 0.01, step=0.0001, value=0.001, label="Lernrate"),
    gr.Slider(10, 100, step=10, value=50, label="Epochen"),
    gr.Slider(10, 200, step=10, value=100, label="Epochen für Over-Fit"),
    gr.Slider(1, 32, step=1, value=16, label="Batch-Größe")
]

interface = gr.Interface(interactive_plot, data_inputs + model_inputs, "plot",
                         title="Regression mit Feed-Forward Neural Network",
                         description="Passen Sie die Parameter an und klicken Sie auf 'Submit', um die Visualisierung zu generieren.")

interface.launch()






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

interface.launch()"""
