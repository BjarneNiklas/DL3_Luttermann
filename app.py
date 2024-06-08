import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import gradio as gr

# Datengenerierung
np.random.seed(42)
N = 100
x = np.random.uniform(-2, 2, N)
def y_function(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1
y = y_function(x)

# Aufteilung in Trainings- und Testdaten
indices = np.random.permutation(N)
train_idx, test_idx = indices[:N//2], indices[N//2:]
x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

# Hinzufügen von Rauschen
noise_variance = 0.05
y_train_noisy = y_train + np.random.normal(0, np.sqrt(noise_variance), y_train.shape)
y_test_noisy = y_test + np.random.normal(0, np.sqrt(noise_variance), y_test.shape)

# Modellarchitektur
def build_model():
    model = Sequential([
        Dense(100, activation='relu', input_shape=(1,)),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# Training des Modells ohne Rauschen
model_unverrauscht = build_model()
model_unverrauscht.fit(x_train, y_train, epochs=200, batch_size=32, verbose=0)

# Training des Best-Fit-Modells mit Rauschen
model_best_fit = build_model()
model_best_fit.fit(x_train, y_train_noisy, epochs=200, batch_size=32, verbose=0)

# Training des Overfit-Modells mit Rauschen
model_overfit = build_model()
model_overfit.fit(x_train, y_train_noisy, epochs=500, batch_size=32, verbose=0)

# Vorhersagen
y_pred_unverrauscht_train = model_unverrauscht.predict(x_train).flatten()
y_pred_unverrauscht_test = model_unverrauscht.predict(x_test).flatten()

y_pred_best_fit_train = model_best_fit.predict(x_train).flatten()
y_pred_best_fit_test = model_best_fit.predict(x_test).flatten()

y_pred_overfit_train = model_overfit.predict(x_train).flatten()
y_pred_overfit_test = model_overfit.predict(x_test).flatten()

# MSE-Berechnung
mse_unverrauscht_train = np.mean((y_pred_unverrauscht_train - y_train) ** 2)
mse_unverrauscht_test = np.mean((y_pred_unverrauscht_test - y_test) ** 2)

mse_best_fit_train = np.mean((y_pred_best_fit_train - y_train_noisy) ** 2)
mse_best_fit_test = np.mean((y_pred_best_fit_test - y_test_noisy) ** 2)

mse_overfit_train = np.mean((y_pred_overfit_train - y_train_noisy) ** 2)
mse_overfit_test = np.mean((y_pred_overfit_test - y_test_noisy) ** 2)

# Plots
def plot_data(x_train, y_train, x_test, y_test, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Train Data'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Test Data'))
    fig.update_layout(title=title)
    return fig

def plot_predictions(x, y_true, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_true, mode='markers', name='True Data'))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Predictions'))
    fig.update_layout(title=title)
    return fig

def create_dashboard():
    fig1 = plot_data(x_train, y_train, x_test, y_test, 'Original Data')
    fig2 = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, 'Noisy Data')

    fig3 = plot_predictions(x_train, y_train, y_pred_unverrauscht_train, 'Unverrauscht - Train')
    fig4 = plot_predictions(x_test, y_test, y_pred_unverrauscht_test, 'Unverrauscht - Test')

    fig5 = plot_predictions(x_train, y_train_noisy, y_pred_best_fit_train, 'Best Fit - Train')
    fig6 = plot_predictions(x_test, y_test_noisy, y_pred_best_fit_test, 'Best Fit - Test')

    fig7 = plot_predictions(x_train, y_train_noisy, y_pred_overfit_train, 'Overfit - Train')
    fig8 = plot_predictions(x_test, y_test_noisy, y_pred_overfit_test, 'Overfit - Test')

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8

# Gradio Interface
def gradio_interface():
    fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8 = create_dashboard()
    
    interface = gr.Interface(
        fn=create_dashboard, 
        inputs=[], 
        outputs=[
            gr.Plot(label="Original Data"),
            gr.Plot(label="Noisy Data"),
            gr.Plot(label="Unverrauscht - Train"),
            gr.Plot(label="Unverrauscht - Test"),
            gr.Plot(label="Best Fit - Train"),
            gr.Plot(label="Best Fit - Test"),
            gr.Plot(label="Overfit - Train"),
            gr.Plot(label="Overfit - Test"),
        ],
        title="Feed-Forward Neural Network Regression",
        description="Visualisierung der Regressionsergebnisse"
    )
    interface.launch()

if __name__ == "__main__":
    gradio_interface()

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
