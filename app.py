import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import time

# Funktion zur Bereinigung des Textes
def clean_text(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Pfade für Dateien
input_path_primary = '/content/Corpus.txt'  # Google Colab
input_path_secondary = '/kaggle/input/corpus/Corpus.txt'  # Kaggle
input_path_tertiary = 'LM_LSTM/Corpus.txt'  # Hugging Face
output_path_primary = '/content/Corpus-cleaned.txt'
output_path_secondary = '/kaggle/working/Corpus-cleaned.txt'
output_path_tertiary = 'LM_LSTM/Corpus-cleaned.txt'
model_path_primary = '/content/word_prediction_model.keras'
model_path_secondary = '/kaggle/working/word_prediction_model.keras'
model_path_tertiary = 'LM_LSTM/word_prediction_model.keras'
tokenizer_path_primary = '/content/tokenizer.pickle'
tokenizer_path_secondary = '/kaggle/working/tokenizer.pickle'
tokenizer_path_tertiary = 'LM_LSTM/tokenizer.pickle'

# Bereinigung und Speicherung des Textes
if not os.path.exists(output_path_primary):
    input_path = input_path_primary
    if os.path.exists(input_path_secondary):
        input_path = input_path_secondary
    elif os.path.exists(input_path_tertiary):
        input_path = input_path_tertiary
        
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()
    cleaned_text = clean_text(text)
    with open(output_path_primary, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# Modelltraining oder Laden des Modells
def load_model_and_tokenizer():
    if os.path.exists(model_path_primary) and os.path.exists(tokenizer_path_primary):
        model = tf.keras.models.load_model(model_path_primary)
        with open(tokenizer_path_primary, 'rb') as handle:
            tokenizer = pickle.load(handle)
    elif os.path.exists(model_path_secondary) and os.path.exists(tokenizer_path_secondary):
        model = tf.keras.models.load_model(model_path_secondary)
        with open(tokenizer_path_secondary, 'rb') as handle:
            tokenizer = pickle.load(handle)
    elif os.path.exists(model_path_tertiary) and os.path.exists(tokenizer_path_tertiary):
        model = tf.keras.models.load_model(model_path_tertiary)
        with open(tokenizer_path_tertiary, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        return None, None

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    with open(output_path_primary, 'r', encoding='utf-8') as file:
        cleaned_text = file.read()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cleaned_text])
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []

    for line in cleaned_text.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len - 1),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(total_words, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32)

    model.save(model_path_primary)
    with open(tokenizer_path_primary, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Funktion zur Vorhersage des nächsten Wortes
def predict_next_words(prompt, top_k=5):
    tokens = tokenizer.texts_to_sequences([prompt])
    padded_seq = pad_sequences(tokens, maxlen=model.input_shape[1] - 1, padding='pre')
    predictions = model.predict(padded_seq)
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_words = [(tokenizer.index_word.get(i, ''), predictions[0][i]) for i in top_indices]
    return top_words

# Berechnung der Genauigkeit
def calculate_accuracy(predictions, actual_word):
    correct_predictions = [word for word, _ in predictions if word == actual_word]
    accuracy = len(correct_predictions) / len(predictions) * 100 if predictions else 0
    return accuracy

# Gradio-Interface
def auto_generate(prompt, auto):
    generated_text = prompt
    for _ in range(10):  # Generiere 10 Wörter
        top_words = predict_next_words(generated_text)
        next_word = top_words[0][0]  # Das wahrscheinlichste Wort
        generated_text += ' ' + next_word
        time.sleep(0.2)  # Pause für 0,2 Sekunden
    return generated_text

with gr.Blocks() as demo:
    gr.Markdown("## LSTM-basierte Wortvorhersage")

    input_text = gr.Textbox(label="Eingabetext")
    prediction_button = gr.Button("Vorhersage")
    top_words = gr.Dataframe(headers=["Wort", "Wahrscheinlichkeit"], datatype=["str", "number"], interactive=False)
    
    prediction_button.click(fn=predict_next_words, inputs=input_text, outputs=top_words)

    accuracy_output = gr.Textbox(label="Vorhersage Genauigkeit (%)", interactive=False)

    def get_accuracy(prompt):
        actual_word = prompt.split()[-1] if prompt else ''
        predictions = predict_next_words(prompt)
        accuracy = calculate_accuracy(predictions, actual_word)
        return accuracy

    auto_generate_checkbox = gr.Checkbox(label="Automatische Generierung aktivieren", value=False)
    auto_generate_button = gr.Button("Generiere Text")
    generated_text_output = gr.Textbox(label="Generierter Text", interactive=False)

    auto_generate_button.click(fn=auto_generate, inputs=(input_text, auto_generate_checkbox), outputs=generated_text_output)

    gr.Markdown("""
    ## Dokumentation
    
    Diese Anwendung verwendet ein gestapeltes LSTM-Modell zur Wortvorhersage. Die Architektur des Modells besteht aus:
    - Einer Embedding-Schicht
    - Zwei LSTM-Schichten mit jeweils 100 Einheiten
    - Einer dichten Ausgabeschicht mit Softmax-Aktivierung

    Das Modell wird auf dem bereitgestellten Datensatz mit dem Verlust von kategorischer Kreuzentropie und dem Adam-Optimizer trainiert.
    
    ## Diskussion
    
    Das aktuelle Modell zeigt grundlegende Fähigkeiten zur Wortvorhersage. Bereiche zur Verbesserung umfassen:
    1. Feineinstellung der Hyperparameter (z.B. LSTM-Einheiten, Embedding-Dimension).
    2. Experimentieren mit verschiedenen Modellarchitekturen.
    3. Implementierung ausgefeilterer Textvorverarbeitung.
    4. Erforschung von Techniken zur Vermeidung von Überanpassung (z.B. Dropout, Regularisierung).
    
    ## Benutzeranleitung
    
    1. Geben Sie einen Starttext in das Feld "Eingabetext" ein.
    2. Klicken Sie auf "Vorhersage", um die 5 wahrscheinlichsten nächsten Wörter anzuzeigen.
    3. Aktivieren Sie das Kontrollkästchen "Automatische Generierung", um 10 Wörter automatisch zu generieren.
    """)

if __name__ == "__main__":
    demo.launch()
