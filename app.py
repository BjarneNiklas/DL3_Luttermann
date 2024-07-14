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
input_path_ggcolab = '/content/Corpus.txt'
input_path_kaggle = '/kaggle/input/corpus/Corpus.txt'
input_path_hugface = '/huggingface/Corpus.txt'
output_path_ggcolab = '/content/Corpus-cleaned.txt'
output_path_kaggle = '/kaggle/working/Corpus-cleaned.txt'
output_path_hugface = '/huggingface/Corpus-cleaned.txt'
model_path_ggcolab = '/content/word_prediction_model.keras'
model_path_kaggle = '/kaggle/working/word_prediction_model.keras'
tokenizer_path_ggcolab = '/content/tokenizer.pickle'
tokenizer_path_kaggle = '/kaggle/working/tokenizer.pickle'

# Bereinigung und Speicherung des Textes
def prepare_data():
    input_path = input_path_ggcolab if os.path.exists(input_path_ggcolab) else (
        input_path_kaggle if os.path.exists(input_path_kaggle) else (
            input_path_hugface if os.path.exists(input_path_hugface) else None
        )
    )
    
    if input_path is None:
        raise FileNotFoundError("Keiner der Eingabepfade konnte gefunden werden.")
    
    if not os.path.exists(os.path.dirname(output_path_ggcolab)):
        os.makedirs(os.path.dirname(output_path_ggcolab))
    
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    cleaned_text = clean_text(text)
    with open(output_path_ggcolab, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# Modelltraining oder Laden des Modells
def load_model_and_tokenizer():
    if os.path.exists(model_path_ggcolab) and os.path.exists(tokenizer_path_ggcolab):
        model = tf.keras.models.load_model(model_path_ggcolab)
        with open(tokenizer_path_ggcolab, 'rb') as handle:
            tokenizer = pickle.load(handle)
    elif os.path.exists(model_path_kaggle) and os.path.exists(tokenizer_path_kaggle):
        model = tf.keras.models.load_model(model_path_kaggle)
        with open(tokenizer_path_kaggle, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        raise FileNotFoundError("Modell- oder Tokenizer-Datei nicht gefunden.")
    
    return model, tokenizer

prepare_data()
model, tokenizer = load_model_and_tokenizer()

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

# Automatische Textgenerierung
def auto_generate(prompt, auto):
    generated_text = prompt
    for _ in range(10):  # Generiere 10 Wörter
        if auto:
            top_words = predict_next_words(generated_text)
            next_word = top_words[0][0]  # Wahrscheinlichstes Wort
            generated_text += ' ' + next_word
            time.sleep(0.2)  # Pause für 0.2 Sekunden
    return generated_text

# Gradio-Interface
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

    auto_generate_checkbox = gr.Checkbox(label="Automatische Generierung alle 0,2 Sekunden", value=False)
    auto_generate_button = gr.Button("Generiere Text")
    generated_text_output = gr.Textbox(label="Generierter Text", interactive=False)

    auto_generate_button.click(fn=auto_generate, inputs=[input_text, auto_generate_checkbox], outputs=generated_text_output)

demo.launch()
