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
input_path_hugface = 'LM_LSTM/Corpus.txt'
output_path_ggcolab = '/content/Corpus-cleaned.txt'
output_path_kaggle = '/kaggle/working/Corpus-cleaned.txt'
output_path_hugface = 'LM_LSTM/Corpus-cleaned.txt'
model_path_ggcolab = '/content/word_prediction_model.keras'
model_path_kaggle = '/kaggle/working/word_prediction_model.keras'
tokenizer_path_ggcolab = '/content/tokenizer.pickle'
tokenizer_path_kaggle = '/kaggle/working/tokenizer.pickle'

# Bereinigung und Speicherung des Textes
def prepare_data():
    if not os.path.exists(output_path_ggcolab):
        with open(input_path_ggcolab, 'r', encoding='utf-8') as file:
            text = file.read()
        cleaned_text = clean_text(text)
        with open(output_path_ggcolab, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

    # Füge hier ähnliche Logik für Kaggle und Hugging Face hinzu...

# Modelltraining oder Laden des Modells
def load_model_and_tokenizer():
    model = None
    tokenizer = None
    if os.path.exists(model_path_ggcolab) and os.path.exists(tokenizer_path_ggcolab):
        model = tf.keras.models.load_model(model_path_ggcolab)
        with open(tokenizer_path_ggcolab, 'rb') as handle:
            tokenizer = pickle.load(handle)
    # Füge hier ähnliches für Kaggle und Hugging Face hinzu...

    if model is None or tokenizer is None:
        raise FileNotFoundError("Modell- oder Tokenizer-Datei nicht gefunden.")

    return model, tokenizer

# Funktion zur Vorhersage des nächsten Wortes
def predict_next_words(prompt, top_k=5, top_p=0.95):
    tokens = tokenizer.texts_to_sequences([prompt])
    padded_seq = pad_sequences(tokens, maxlen=model.input_shape[1] - 1, padding='pre')
    predictions = model.predict(padded_seq)[0]
    
    # Top-k Sampling
    top_indices = np.argsort(predictions)[-top_k:]
    top_probs = predictions[top_indices] / np.sum(predictions[top_indices])
    
    # Nucleus Sampling (Top-p)
    sorted_indices = np.argsort(predictions)[::-1]
    cumulative_probs = np.cumsum(predictions[sorted_indices])
    sorted_indices = sorted_indices[cumulative_probs <= top_p]

    next_word_idx = np.random.choice(sorted_indices)
    return tokenizer.index_word[next_word_idx], predictions[next_word_idx]

# Gradio-Interface
def auto_generate(prompt, auto):
    generated_text = prompt
    if not auto:
        for _ in range(10):
            next_word, _ = predict_next_words(generated_text)
            generated_text += ' ' + next_word
    else:
        for _ in range(10):
            next_word, _ = predict_next_words(generated_text)
            generated_text += ' ' + next_word
            time.sleep(0.2)  # Alle 0,2 Sekunden generieren
    return generated_text

with gr.Blocks() as demo:
    gr.Markdown("## LSTM-basierte Wortvorhersage")

    input_text = gr.Textbox(label="Eingabetext")
    auto_generate_checkbox = gr.Checkbox(label="Automatische Generierung alle 0,2 Sekunden", value=False)
    auto_generate_button = gr.Button("Generiere Text")
    
    generated_text_output = gr.Textbox(label="Generierter Text", interactive=True)

    auto_generate_button.click(fn=auto_generate, inputs=(input_text, auto_generate_checkbox), outputs=generated_text_output)

demo.launch()
