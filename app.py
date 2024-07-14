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
    return re.sub(r'\s+', ' ', re.sub(r'[\(\[].*?[\)\]]', '', re.sub(r'\d+', '', text))).strip()

# Pfade für Dateien
input_paths = {
    "ggcolab": '/content/Corpus.txt',
    "kaggle": '/kaggle/input/corpus/Corpus.txt',
    "hugface": '/huggingface/Corpus.txt'
}

output_paths = {
    "ggcolab": '/content/Corpus-cleaned.txt',
    "kaggle": '/kaggle/working/Corpus-cleaned.txt',
    "hugface": '/huggingface/Corpus-cleaned.txt'
}

model_paths = {
    "ggcolab": '/content/word_prediction_model.keras',
    "kaggle": '/kaggle/working/word_prediction_model.keras',
    "hugface": '/huggingface/word_prediction_model.keras'
}

tokenizer_paths = {
    "ggcolab": '/content/tokenizer.pickle',
    "kaggle": '/kaggle/working/tokenizer.pickle',
    "hugface": '/huggingface/tokenizer.pickle'
}

# Bereinigung und Speicherung des Textes
def prepare_data():
    input_path = next((path for path in input_paths.values() if os.path.exists(path)), None)
    if input_path is None:
        raise FileNotFoundError("Keiner der Eingabepfade konnte gefunden werden.")

    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()

    cleaned_text = clean_text(text)
    output_path = output_paths["ggcolab"] if input_path == input_paths["ggcolab"] else output_paths["kaggle"] if input_path == input_paths["kaggle"] else output_paths["hugface"]
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# Modelltraining oder Laden des Modells
def load_model_and_tokenizer():
    for key in ["ggcolab", "kaggle", "hugface"]:
        if os.path.exists(model_paths[key]) and os.path.exists(tokenizer_paths[key]):
            model = tf.keras.models.load_model(model_paths[key])
            with open(tokenizer_paths[key], 'rb') as handle:
                tokenizer = pickle.load(handle)
            return model, tokenizer
    
    # Wenn die Dateien nicht existieren, erstellen und trainieren wir das Modell
    with open(output_paths["ggcolab"], 'r', encoding='utf-8') as file:
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

    # Speichern des Modells und Tokenizers
    model.save(model_paths["ggcolab"])  # Speichere als primäre Datei
    with open(tokenizer_paths["ggcolab"], 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, tokenizer

prepare_data()
model, tokenizer = load_model_and_tokenizer()

# Funktion zur Vorhersage des nächsten Wortes
def predict_next_words(prompt, top_k=5):
    tokens = tokenizer.texts_to_sequences([prompt])
    padded_seq = pad_sequences(tokens, maxlen=model.input_shape[1] - 1, padding='pre')
    predictions = model.predict(padded_seq)
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    return [(tokenizer.index_word.get(i, ''), predictions[0][i]) for i in top_indices]

# Berechnung der Genauigkeit
def calculate_accuracy(predictions, actual_word):
    correct_predictions = [word for word, _ in predictions if word == actual_word]
    return (len(correct_predictions) / len(predictions) * 100) if predictions else 0

# Automatische Textgenerierung
def auto_generate(prompt, auto):
    generated_text = prompt
    for _ in range(10):  # Generiere 10 Wörter
        if auto:
            top_words = predict_next_words(generated_text)
            next_word = top_words[0][0]  # Wahrscheinlichstes Wort
            generated_text += ' ' + next_word
            time.sleep(0.2)  # Warte 0,2 Sekunden
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

    input_text.change(fn=get_accuracy, inputs=input_text, outputs=accuracy_output)

    auto_generate_checkbox = gr.Checkbox(label="Automatische Generierung alle 0,2 Sekunden", value=False)
    auto_generate_button = gr.Button("Generiere Text")
    generated_text_output = gr.Textbox(label="Generierter Text", interactive=False)

    auto_generate_button.click(fn=auto_generate, inputs=[input_text, auto_generate_checkbox], outputs=generated_text_output)

demo.launch()
