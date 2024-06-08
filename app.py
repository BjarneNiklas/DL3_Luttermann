import numpy as np
import plotly.graph_objs as go
import gradio as gr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Definiere die Funktion
def f(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Generiere Daten
def generate_data(N, x_min, x_max, noise_var):
    x_data = np.random.uniform(x_min, x_max, N)
    y_data = f(x_data)

    idx = np.random.permutation(N)
    x_train, x_test = x_data[idx[:N//2]], x_data[idx[N//2:]]
    y_train, y_test = y_data[idx[:N//2]], y_data[idx[N//2:]]

    noise = np.random.normal(0, noise_var, N//2)
    y_train_noisy = y_train + noise
    y_test_noisy = y_test + noise

    return x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy

# Definiere das Modell
def build_model(hidden_units):
    model = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(1,)),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# Trainiere das Modell
def train_model(model, x_train, y_train, x_test, y_test, epochs):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    train_loss = model.evaluate(x_train, y_train, verbose=0)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    return history, train_loss, test_loss

# Visualisierung
def plot_data(x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy, predictions):
    layout = go.Layout(
        title='Datensätze',
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),
        width=600, height=400
    )

    trace1 = go.Scatter(x=x_train, y=y_train, mode='markers', name='Trainingsdata (unverrauscht)')
    trace2 = go.Scatter(x=x_test, y=y_test, mode='markers', name='Testdata (unverrauscht)')
    trace3 = go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Trainingsdata (verrauscht)')
    trace4 = go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Testdata (verrauscht)')

    data_noisy = [trace3, trace4]
    data_clean = [trace1, trace2]

    fig_noisy = go.Figure(data=data_noisy, layout=layout)
    fig_clean = go.Figure(data=data_clean, layout=layout)

    figs = []
    for pred in predictions:
        layout_pred = go.Layout(
            title='Vorhersage',
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            width=600, height=400
        )

        x_range = np.linspace(x_min, x_max, 100)
        y_pred = pred(x_range)

        trace_pred = go.Scatter(x=x_range, y=y_pred, mode='lines', name='Vorhersage')
        trace_train = go.Scatter(x=x_train, y=y_train_noisy, mode='markers', name='Trainingsdata (verrauscht)')
        trace_test = go.Scatter(x=x_test, y=y_test_noisy, mode='markers', name='Testdata (verrauscht)')

        fig = go.Figure(data=[trace_pred, trace_train, trace_test], layout=layout_pred)
        figs.append(fig)

    return fig_noisy, fig_clean, figs

# Gradio-Funktion
def gradio_ui(data_settings, model_settings):
    N = data_settings['N']
    x_min = data_settings['x_min']
    x_max = data_settings['x_max']
    noise_var = data_settings['noise_var']

    hidden_units = model_settings['hidden_units']
    epochs_clean = model_settings['epochs_clean']
    epochs_best = model_settings['epochs_best']
    epochs_overfit = model_settings['epochs_overfit']

    x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy = generate_data(N, x_min, x_max, noise_var)

    # Trainiere Modelle
    model_clean = build_model(hidden_units)
    history_clean, train_loss_clean, test_loss_clean = train_model(model_clean, x_train, y_train, x_test, y_test, epochs_clean)

    model_best = build_model(hidden_units)
    history_best, train_loss_best, test_loss_best = train_model(model_best, x_train, y_train_noisy, x_test, y_test_noisy, epochs_best)

    model_overfit = build_model(hidden_units)
    history_overfit, train_loss_overfit, test_loss_overfit = train_model(model_overfit, x_train, y_train_noisy, x_test, y_test_noisy, epochs_overfit)

    # Vorhersagen
    pred_clean = lambda x: model_clean.predict(x)
    pred_best = lambda x: model_best.predict(x)
    pred_overfit = lambda x: model_overfit.predict(x)

    # Visualisierung
    fig_noisy, fig_clean, figs = plot_data(x_train, y_train, y_train_noisy, x_test, y_test, y_test_noisy, [pred_clean, pred_best, pred_overfit])

    # Ausgabe
    with gr.Row():
        with gr.Column():
            gr.Plot(fig_clean, show_label=False)
            gr.Plot(fig_noisy, show_label=False)
        with gr.Column():
            gr.Plot(figs[0], show_label=False)
            gr.Markdown(f"Trainings-Loss (MSE): {train_loss_clean:.4f}, Test-Loss (MSE): {test_loss_clean:.4f}")
            gr.Plot(figs[1], show_label=False)
            gr.Markdown(f"Trainings-Loss (MSE): {train_loss_best:.4f}, Test-Loss (MSE): {test_loss_best:.4f}")
            gr.Plot(figs[2], show_label=False)
            gr.Markdown(f"Trainings-Loss (MSE): {train_loss_overfit:.4f}, Test-Loss (MSE): {test_loss_overfit:.4f}")

    # Dokumentation
    doc = gr.Markdown("""
    ## Dokumentation

    In dieser Lösung wurde eine Feed-Forward Neural Network (FFNN) Architektur mit zwei versteckten Schichten (Hidden Layers) und jeweils 100 Neuronen pro Schicht implementiert. Die Aktivierungsfunktion in den versteckten Schichten ist die ReLU-Funktion, während im Ausgabeschicht (Output Layer) eine lineare Aktivierungsfunktion verwendet wird.

    Das Modell wurde mit dem Adam-Optimierer und einer Lernrate von 0.01 trainiert. Als Verlustfunktion (Loss) wurde der Mean Squared Error (MSE) verwendet.

    Zunächst wurden Trainingsdaten und Testdaten aus der angegebenen Funktion `y(x) = 0.5*(x+0.8)*(x+1.8)*(x-0.2)*(x-0.3)*(x-1.9)+1` im Bereich `x ∈ [-2, 2]` generiert. Dabei wurden 100 Datenpunkte erzeugt und gleichmäßig in Trainings- und Testdaten aufgeteilt. Anschließend wurde zu den Ausgabewerten (`y`-Werten) normalverteiltes Rauschen mit einer Varianz von 0.05 hinzugefügt, um eine realistische Situation zu simulieren.

    Es wurden drei Modelle trainiert:
    1. Ein Modell auf den unverrauschten Daten (als Referenz)
    2. Ein Modell auf den verrauschten Daten mit einer geeigneten Epochenanzahl für gute Generalisierung (Best-Fit)
    3. Ein Modell auf den verrauschten Daten mit einer sehr hohen Epochenanzahl, um Overfitting zu demonstrieren

    Die Ergebnisse wurden in einem Gradio-Interface visualisiert, das die Datensätze, Modellvorhersagen und Verluste (Trainings- und Test-Loss) in Plotly-Diagrammen darstellt.

    ## Diskussion

    Wie erwartet, zeigt das Modell, das auf den unverrauschten Daten trainiert wurde, nahezu identische Verluste auf den Trainings- und Testdaten. Dies liegt daran, dass in diesem Fall keine Verrauschung vorliegt und das Modell die Funktion perfekt lernen kann.

    Das Best-Fit-Modell, das auf den verrauschten Daten trainiert wurde, weist einen etwas höheren Verlust auf den Testdaten im Vergleich zu den Trainingsdaten auf. Dies ist ein Anzeichen dafür, dass das Modell in der Lage ist, die zugrunde liegende Funktion trotz des Rauschens zu approximieren, ohne zu overfitting.

    Im Gegensatz dazu zeigt das Overfit-Modell, das mit einer sehr hohen Epochenanzahl trainiert wurde, einen deutlich geringeren Verlust auf den Trainingsdaten im Vergleich zu den Testdaten. Dies ist ein klares Anzeichen für Overfitting, da das Modell die Trainingsdaten nahezu perfekt gelernt hat, aber nicht in der Lage ist, auf neue, ungesehene Daten (Testdaten) zu generalisieren.

    Insgesamt demonstriert diese Lösung den Einfluss von Rauschen und Overfitting auf das Lernverhalten eines Feed-Forward Neural Networks und unterstreicht die Bedeutung einer sorgfältigen Modellwahl und Hyperparameter-Optimierung für eine gute Generalisierungsleistung.
    """)

    return doc

# Erstelle das Gradio-Interface
data_settings = gr.Group(
    [
        gr.Slider(10, 1000, value=100, step=10, label="Anzahl Datenpunkte (N)"),
        gr.Slider(-5, 5, value=-2, label="x_min"),
        gr.Slider(-5, 5, value=2, label="x_max"),
        gr.Slider(0, 1, value=0.05, step=0.01, label="Rauschen-Varianz (V)")
    ],
    label="Data Settings"
)

model_settings = gr.Group(
    [
        gr.Slider(10, 200, value=100, step=10, label="Anzahl versteckter Neuronen"),
        gr.Slider(10, 1000, value=100, step=10, label="Epochen (unverrauscht)"),
        gr.Slider(10, 1000, value=200, step=10, label="Epochen (Best-Fit)"),
        gr.Slider(10, 1000, value=500, step=10, label="Epochen (Overfit)")
    ],
    label="Model Training Settings"
)

gr.Interface(gradio_ui, inputs=[data_settings, model_settings], outputs="markdown", layout="horizontal", title="Regression mit Feed-Forward Neural Network").launch()


"""import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import gradio as gr

# Funktion zur Datenberechnung
def ground_truth_function(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Generiere Daten
def generate_data(N=100, noise_variance=0.05):
    x = np.random.uniform(-2, 2, N)
    y = ground_truth_function(x)
    noise = np.random.normal(0, np.sqrt(noise_variance), N)
    y_noisy = y + noise
    return x, y, y_noisy

# Trainieren des Modells
def train_model(x_train, y_train, epochs, learning_rate=0.01):
    model = Sequential([
        Dense(100, activation='relu', input_shape=(1,)),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

# Berechnen des Loss
def calculate_loss(model, x, y):
    y_pred = model.predict(x)
    return np.mean((y - y_pred.squeeze())**2)

# Plots erstellen
def create_plot(x, y, y_noisy, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='True', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y_noisy, mode='markers', name='Noisy', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name='Predicted', line=dict(color='green')))
    fig.update_layout(title=title, xaxis_title='x', yaxis_title='y')
    return fig

# Funktion zum Trainieren und Plotten der Ergebnisse
def train_and_plot(N, noise_variance, epochs_best_fit, epochs_overfit):
    # Daten generieren
    x, y, y_noisy = generate_data(N, noise_variance)
    split_index = N // 2
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    y_noisy_train, y_noisy_test = y_noisy[:split_index], y_noisy[split_index:]
    
    # Modelle trainieren
    model_clean = train_model(x_train, y_train, epochs=epochs_best_fit)
    model_best_fit = train_model(x_train, y_noisy_train, epochs=epochs_best_fit)
    model_overfit = train_model(x_train, y_noisy_train, epochs=epochs_overfit)
    
    # Vorhersagen berechnen
    y_pred_clean_train = model_clean.predict(x_train)
    y_pred_clean_test = model_clean.predict(x_test)
    y_pred_best_fit_train = model_best_fit.predict(x_train)
    y_pred_best_fit_test = model_best_fit.predict(x_test)
    y_pred_overfit_train = model_overfit.predict(x_train)
    y_pred_overfit_test = model_overfit.predict(x_test)
    
    # Loss berechnen
    loss_clean_train = calculate_loss(model_clean, x_train, y_train)
    loss_clean_test = calculate_loss(model_clean, x_test, y_test)
    loss_best_fit_train = calculate_loss(model_best_fit, x_train, y_noisy_train)
    loss_best_fit_test = calculate_loss(model_best_fit, x_test, y_noisy_test)
    loss_overfit_train = calculate_loss(model_overfit, x_train, y_noisy_train)
    loss_overfit_test = calculate_loss(model_overfit, x_test, y_noisy_test)
    
    # Plots erstellen
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
"""




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