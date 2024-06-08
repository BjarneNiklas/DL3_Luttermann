import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
import gradio as gr

# Funktion zum Generieren der Daten
def generate_data(N=100, noise_variance=0.05, x_min=-2, x_max=2):
    x = np.random.uniform(x_min, x_max, N)
    y_true = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
    y_noisy = y_true + np.random.normal(0, np.sqrt(noise_variance), N)
    
    indices = np.random.permutation(N)
    train_indices = indices[:N // 2]
    test_indices = indices[N // 2:]
    
    x_train, y_train = x[train_indices], y_noisy[train_indices]
    x_test, y_test = x[test_indices], y_noisy[test_indices]
    
    return (x_train, y_train), (x_test, y_test), (x, y_true)

# Funktion zum Erstellen und Trainieren der Modelle
def create_and_train_model(x_train, y_train, hidden_layers, neurons_per_layer, learning_rate, epochs, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(1,)))
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error')
    
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, history

# Funktion zum Plotten der Daten
def plot_data(x_train, y_train, x_test, y_test, y_true, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=np.sort(x_train), y=y_true[np.argsort(x_train)], mode='lines', name='True Function', line=dict(color='green')))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    return fig

# Funktion zum Plotten der Vorhersagen
def plot_predictions(model, x_train, y_train, x_test, y_test, y_true, title):
    y_pred_train = model.predict(x_train).flatten()
    y_pred_test = model.predict(x_test).flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=np.sort(x_train), y=y_pred_train[np.argsort(x_train)], mode='lines', name='Pred Train', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=np.sort(x_test), y=y_pred_test[np.argsort(x_test)], mode='lines', name='Pred Test', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=np.sort(x_train), y=y_true[np.argsort(x_train)], mode='lines', name='True Function', line=dict(color='green')))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    return fig

def main(N=100, noise_variance=0.05, x_min=-2, x_max=2, hidden_layers=2, neurons_per_layer=100, learning_rate=0.01, epochs_best=100, epochs_overfit=1000, batch_size=32):
    (x_train, y_train), (x_test, y_test), (x, y_true) = generate_data(N, noise_variance, x_min, x_max)
    
    # Trainiere das Modell ohne Rauschen
    model_unnoisy, _ = create_and_train_model(x_train, y_true[train_indices], hidden_layers, neurons_per_layer, learning_rate, epochs_best, batch_size)
    
    # Trainiere die Modelle mit Rauschen
    model_best, _ = create_and_train_model(x_train, y_train, hidden_layers, neurons_per_layer, learning_rate, epochs_best, batch_size)
    model_overfit, _ = create_and_train_model(x_train, y_train, hidden_layers, neurons_per_layer, learning_rate, epochs_overfit, batch_size)
    
    fig_data_no_noise = plot_data(x_train, y_true[train_indices], x_test, y_true[test_indices], y_true, 'Noiseless Datasets')
    fig_data_with_noise = plot_data(x_train, y_train, x_test, y_test, y_true, 'Noisy Datasets')
    
    fig_pred_unnoisy = plot_predictions(model_unnoisy, x_train, y_true[train_indices], x_test, y_true[test_indices], y_true, 'Predictions (Unnoisy Model)')
    fig_pred_best = plot_predictions(model_best, x_train, y_train, x_test, y_test, y_true, 'Predictions (Best-Fit Model)')
    fig_pred_overfit = plot_predictions(model_overfit, x_train, y_train, x_test, y_test, y_true, 'Predictions (Overfit Model)')
    
    return fig_data_no_noise, fig_data_with_noise, fig_pred_unnoisy, fig_pred_best, fig_pred_overfit

# Gradio UI
inputs = [
    gr.Slider(minimum=10, maximum=500, value=100, label="Data Points (N)"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.05, label="Noise Variance (V)"),
    gr.Slider(minimum=-10.0, maximum=0.0, value=-2.0, label="X Min"),
    gr.Slider(minimum=0.0, maximum=10.0, value=2.0, label="X Max"),
    gr.Slider(minimum=1, maximum=10, value=2, label="Hidden Layers"),
    gr.Slider(minimum=10, maximum=500, value=100, label="Neurons per Layer"),
    gr.Slider(minimum=0.0001, maximum=0.1, value=0.01, label="Learning Rate"),
    gr.Slider(minimum=10, maximum=1000, value=200, label="Epochs"),
    gr.Slider(minimum=10, maximum=5000, value=500, label="Epochs (Overfit Model)"),
    gr.Slider(minimum=1, maximum=128, value=32, label="Batch Size")
]

outputs = [
    gr.Plot(label="Noiseless Datasets"),
    gr.Plot(label="Noisy Datasets"),
    gr.Plot(label="Predictions (Unnoisy Model)"),
    gr.Plot(label="Predictions (Best-Fit Model)"),
    gr.Plot(label="Predictions (Overfit Model)")
]

gr.Interface(fn=main, inputs=inputs, outputs=outputs, title="Regression with FFNN and TensorFlow.js", description="Train and evaluate FFNN models on synthetic data with and without noise.").launch()


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
