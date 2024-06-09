import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import gradio as gr

# Definieren der Ground-Truth-Funktion
def original_function(x):
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1

# Generieren der Datensätze
def generate_data(N, noise_variance, x_min, x_max):
    x = np.random.uniform(x_min, x_max, N)
    y = original_function(x)
    x_train, x_test = x[:N//2], x[N//2:]
    y_train, y_test = y[:N//2], y[N//2:]
    noise_train = np.random.normal(0, noise_variance**0.5, y_train.shape)
    noise_test = np.random.normal(0, noise_variance**0.5, y_test.shape)
    y_train_noisy = y_train + noise_train
    y_test_noisy = y_test + noise_test

    return x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy

# Definieren des NNMs
def create_model(num_layers, neurons_per_layer):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_dim=1))
    for _ in range(num_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Trainieren des Modells
def train_model(x_train, y_train, epochs, model):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model, history.history['loss'][-1]

# Plot-Daten
def plot_data(x_train, y_train, x_test, y_test, title, show_original_function, x_min, x_max):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='Trainingsdaten', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='Testdaten', marker=dict(color='red')))
    if show_original_function:
        x_range = np.linspace(x_min, x_max, 1000)
        y_original = original_function(x_range)
        fig.add_trace(go.Scatter(x=x_range, y=y_original, mode='lines', name='y(x)', line=dict(color='green')))
    fig.update_layout(title=title)
    return fig

# Vorhersagen der Plots
def plot_predictions(x, y, model, title, show_original_function, x_min, x_max, data_type):
    x_range = np.linspace(x_min, x_max, 1000)
    y_pred = model.predict(x_range).flatten()
    y_pred_points = model.predict(x).flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{data_type}', marker=dict(color='blue' if data_type == 'Trainingsdaten' else 'red')))
    fig.add_trace(go.Scatter(x=x, y=y_pred_points, mode='markers', name='Predicted Points (Datenvorhersage)', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Predicted Curve (Hilfskurve)', line=dict(color='green')))
    if show_original_function:
        y_original = original_function(x_range)
        fig.add_trace(go.Scatter(x=x_range, y=y_original, mode='lines', name='GTF', line=dict(color='orange')))
    fig.update_layout(title=title)
    return fig

# Main Funktion für Training und Plotting
def main(N, noise_variance, x_min, x_max, num_layers, neurons_per_layer, epochs_unnoisy, epochs_best_fit, epochs_overfit, show_original_function):
    # Generate data
    x_train, y_train, x_test, y_test, y_train_noisy, y_test_noisy = generate_data(N, noise_variance, x_min, x_max)

    # Data plot ohne Rauschen
    noiseless_plot = plot_data(x_train, y_train, x_test, y_test, "Datensätze ohne Rauschen", show_original_function, x_min, x_max)

    # Data plot mit Rauschen
    noisy_plot = plot_data(x_train, y_train_noisy, x_test, y_test_noisy, "Datensätze mit Rauschen", show_original_function, x_min, x_max)

    # Modell ohne Rauschen
    model_unnoisy = create_model(num_layers, neurons_per_layer)
    model_unnoisy, loss_unnoisy_train = train_model(x_train, y_train, epochs_unnoisy, model_unnoisy)
    unnoisy_plot_train = plot_predictions(x_train, y_train, model_unnoisy, "Modell ohne Rauschen - Trainingsdaten", show_original_function, x_min, x_max, "Trainingsdaten")
    unnoisy_plot_test = plot_predictions(x_test, y_test, model_unnoisy, "Modell ohne Rauschen - Testdaten", show_original_function, x_min, x_max, "Testdaten")
    loss_unnoisy_test = model_unnoisy.evaluate(x_test, y_test, verbose=0)

    # Best-fit-Modell
    model_best_fit = create_model(num_layers, neurons_per_layer)
    model_best_fit, loss_best_fit_train = train_model(x_train, y_train_noisy, epochs_best_fit, model_best_fit)
    best_fit_plot_train = plot_predictions(x_train, y_train_noisy, model_best_fit, "Best-Fit-Modell - Trainingsdaten", show_original_function, x_min, x_max, "Trainingsdaten")
    best_fit_plot_test = plot_predictions(x_test, y_test_noisy, model_best_fit, "Best-Fit-Modell - Testdaten", show_original_function, x_min, x_max, "Testdaten")
    loss_best_fit_test = model_best_fit.evaluate(x_test, y_test_noisy, verbose=0)

    # Overfit-Modell
    model_overfit = create_model(num_layers, neurons_per_layer)
    model_overfit, loss_overfit_train = train_model(x_train, y_train_noisy, epochs_overfit, model_overfit)
    overfit_plot_train = plot_predictions(x_train, y_train_noisy, model_overfit, "Overfit-Modell - Trainingsdaten", show_original_function, x_min, x_max, "Trainingsdaten")
    overfit_plot_test = plot_predictions(x_test, y_test_noisy, model_overfit, "Overfit-Modell - Testdaten", show_original_function, x_min, x_max, "Testdaten")
    loss_overfit_test = model_overfit.evaluate(x_test, y_test_noisy, verbose=0)

    return (noiseless_plot, noisy_plot,
            unnoisy_plot_train, f"Train Loss: {loss_unnoisy_train:.4f} | Test Loss: {loss_unnoisy_test:.4f}", unnoisy_plot_test,
            best_fit_plot_train, f"Train Loss: {loss_best_fit_train:.4f} | Test Loss: {loss_best_fit_test:.4f}", best_fit_plot_test,
            overfit_plot_train, f"Train Loss: {loss_overfit_train:.4f} | Test Loss: {loss_overfit_test:.4f}", overfit_plot_test)


# Diskussion mit Experimenten und Resultaten
discussion = """
## Experimente, Resultate und Diskussion

Zunächst ist dies ein übliches reales Szenario: Mit dem FFNN können z. B. verrauschte Daten trainiert werden aus der für die Modelle unbekannten Ground-Truth-Funktion.\n
Deswegen generiere ich Datenpunkte aus einer gegebenen Funktion und füge Rauschen hinzu, um Bedingungen der echten Welt zu simulieren (Aufteilen in Trainings- und Testdaten). Im Rahmen des gesamten Trainings müssen die Testdaten unberührt bleiben.\n

Trainiert werden drei Modelle: eines mit den Daten ohne Rauschen (clean model), eines mit den verrauschten Daten für die beste Anpassung (Best-Fit-Modell) und eines zur Demonstration von Overfitting mit den verrauschten Daten mit einer höheren Anzahl von Trainings-Epochen.\n\n

Die meisten Parameter wurden so eingestellt wie vorgegeben (siehe Einstellmöglichkeiten unter "Datengenerierung" und "Modell-Definition" in der Lösung. Was angepasst wurde, sind die Trainings-Epochen für die verschiedenen Modelle, die im Folgenden kurz erläutert werden: \n


"""

# Gradio Interface
inputs = [
    gr.Slider(10, 250, step=1, value=100, label="Anzahl der Datenpunkte (N)"),
    gr.Slider(0.01, 0.2, step=0.01, value=0.05, label="Noise Variance (V)"),
    gr.Slider(-10, -1, step=0.1, value=-2.0, label="X Min"),
    gr.Slider(1, 10, step=0.1, value=2.0, label="X Max"),
    gr.Slider(1, 10, step=1, value=2, label="Anzahl der verborgenen Schichten (Hidden Layers)"),
    gr.Slider(10, 250, step=10, value=100, label="Anzahl der Neuronen pro verborgener Schicht"),
    gr.Slider(10, 500, step=10, value=150, label="Trainings-Epochen (Modell ohne Rauschen)"),
    gr.Slider(100, 500, step=10, value=200, label="Trainings-Epochen (Best-Fit-Modell)"),
    gr.Slider(500, 2500, step=10, value=500, label="Trainings-Epochen (Overfit-Modell)"),
    gr.Checkbox(value=True, label="Ground-Truth-Funktion y(x) anzeigen?")
]

outputs = [
    gr.Plot(label="Clean Datasets"),
    gr.Plot(label="Datensätze mit Rauschen"),
    gr.Plot(label="Clean Model - Trainingsdaten"),
    gr.Textbox(label="Losses (MSE) ohne Rauschen"),
    gr.Plot(label="Clean Model- Testdaten"),
    gr.Plot(label="Best-Fit-Modell - Trainingsdaten"),
    gr.Textbox(label="Losses (MSE) beim Best-Fit-Modell"),
    gr.Plot(label="Best-Fit-Modell - Testdaten"),
    gr.Plot(label="Overfit Model - Trainingsdaten"),
    gr.Textbox(label="Losses (MSE) beim Overfit-Modell"),
    gr.Plot(label="Overfit-Modell - Testdaten")
]

def wrapper(*args):
    return main(*args)

def update_original_function(show_original_function):
    N = inputs[0].value
    noise_variance = inputs[1].value
    x_min = inputs[2].value
    x_max = inputs[3].value
    num_layers = inputs[4].value
    neurons_per_layer = inputs[5].value
    epochs_unnoisy = inputs[6].value
    epochs_best_fit = inputs[7].value
    epochs_overfit = inputs[8].value
    return main(N, noise_variance, x_min, x_max, num_layers, neurons_per_layer, epochs_unnoisy, epochs_best_fit, epochs_overfit, show_original_function)

demo = gr.Blocks(css="footer{display:none !important}")

with demo:
    gr.Markdown("## Regression mit FFNN")

    gr.Markdown("### Datengenerierung")
    with gr.Column():
        for input_widget in inputs[:4]:
            input_widget.render()

    gr.Markdown("### Modell-Definition")
    with gr.Column():
        for input_widget in inputs[4:6]:
            input_widget.render()

    gr.Markdown("### Modell-Training")
    with gr.Column():
        for input_widget in inputs[6:9]:
            input_widget.render()

    gr.Markdown("### Ground-Truth-Funktion (GTF): y(x) = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1")
    with gr.Column():
        inputs[9].render()
    gr.Button("Daten generieren und Modelle trainieren").click(wrapper, inputs, outputs)
    with gr.Row():
        with gr.Column():
            outputs[0].render()
        with gr.Column():
            outputs[1].render()

    with gr.Row():
        with gr.Column():
            outputs[2].render()
        with gr.Column():
            outputs[4].render()
    with gr.Row():
        outputs[3].render()

    with gr.Row():
        with gr.Column():
            outputs[5].render()
        with gr.Column():
            outputs[7].render()
    with gr.Row():
        outputs[6].render()
    
    with gr.Row():
        with gr.Column():
            outputs[8].render()
        with gr.Column():
            outputs[10].render()
    with gr.Row():
        outputs[9].render()
    inputs[9].change(fn=update_original_function, inputs=inputs, outputs=[outputs[0], outputs[1], outputs[2], outputs[4], outputs[5], outputs[7], outputs[8], outputs[10]])

    gr.Markdown(discussion)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""# Technische Dokumentation
                Numpy: Für numerische Berechnungen (u. a. zum Generieren der zufälligen Datensätze und Setzen der x-Range im Diagramm\n
                Gradio: Zur Anzeige der UI-Elemente\n
                TensorFlow Keras: Mit dem ML-Framework wurde die Architektur des neuronalen Netzes definiert (u. a. Adam Optimization Algorithm, Activation Function, Neurons per Layer)\n
                Plotly: Zur Visualisierung der Diagramme\n\n
                Das Besondere an dieser Lösung ist die einfache Benutzerfreundlichkeit aufgrund des gewählten UI-Frameworks Gradio. Zudem ist sie einfach verständlich durch die populäre Python-Programmiersprache und dank der verschiedenen Bibliotheken beliebig erweiter- und wartbar. Denn die Codestruktur ist modular aufgebaut mit separaten kleineren Funktionen für jede Aufgabe. 
            
                ## Logik:
                Die Logik der Implementierung kann wie folgt unterteilt werden: \n
                1. Generieren der Datensätze (mithilfe der Ground-Truth-Funktion und Gaussian Noise)\n
                2. Data Plot ohne und mit Rauschen anzeigen \n
                3. Definieren des NNMs (linear, ReLU und MSE) \n
                4. Trainieren der drei Modelle \n
                5. Berechnen der Train und Test Losses (MSE) für die jeweiligen Modelle
                6. Setup der Gradio-UI mit den Widgets (Eingabedaten) und Output (Plots und Text)
            """)
        with gr.Column():
            gr.Markdown("""
                # Fachliche Dokumentation
        
                Es soll ein neuronales Netzwerk für eine Regressionsproblematik trainiert werden. Verschiedene Szenarien können damit simuliert werden.\n
                Zunächst wird die vorgegebene Ground-Truth-Funktion definiert. Von dieser können Datenpunkte unterschiedlicher Rauschlevel generiert werden.\n
                Im Rahmen der Aufgabe waren die unterschiedlichen Parameter (Möglichkeiten zum Trainieren und Anpassen der Modelle) überwiegend vorgegeben (siehe Aufgabenstellung).\n
                Die 100 Datenpunkte werden in Trainings- und Testdatensätze aufgeteilt (50:50). Drei Modelle werden trainiert: ein Modell auf den rauschfreien Daten,\n 
                ein Modell auf den verrauschten Daten zur bestmöglichen Anpassung (Best-Fit-Modell) und ein Modell mit einer erhöhten Anzahl von Trainings-Epochen, um Overfitting zu beobachten (Overfit-Modell).\n\n

                Durch den Vergleich der MSE der Trainings- und Testdaten kann Overfitting beobachtet werden und geprüft werden, welches Modell am besten generalisiert.\n
                Die Gradio-UI ermöglicht es den Benutzern, verschiedene Parameter anzupassen, wie beispielsweise die Anzahl der Datenpunkte, Noise Variance und Anzahl der Trainings-Epochen.\n
                    
                ## Quellen und hilfreiche Links:
                – GeeksforGeeks (2021): Scatter plot in Plotly using graph_objects class. Online: https://www.geeksforgeeks.org/scatter-plot-in-plotly-using-graph_objects-class/ \n
                – GeeksforGeeks (2022): Adam Optimizer in Tensorflow. Online: https://www.geeksforgeeks.org/adam-optimizer-in-tensorflow/ \n
                – Hugging Face (o. A.): Introduction to Gradio Blocks. Online: https://huggingface.co/learn/nlp-course/en/chapter9/7 \n
                – NumPy (o. A.): numpy.random.normal. Online: https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html \n
                – TensorFlow (o. A.-a): Basic regression: Predict fuel efficiency. Online: https://www.tensorflow.org/tutorials/keras/regression \n
                – TensorFlow (o. A.-b): tf.keras.layers.GaussianNoise. Online: https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianNoise 
            """)
    with gr.Row():
        with gr.Column():
            gr.Markdown("""### Medieninformatik-Wahlpflichtmodul: Deep Learning - ESA 2""")
        with gr.Column():
            gr.Markdown("""### Erstellt und abgegeben am 10. Juni 2024 von: Bjarne Niklas Luttermann (373960, TH Lübeck)""")

demo.launch()