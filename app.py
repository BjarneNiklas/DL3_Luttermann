import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Beispiel PDF-Datei
file_path = 'article.pdf'
text = read_pdf(file_path)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau
import gradio as gr
import PyPDF2

# PDF-Dateien einlesen und Text extrahieren
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Beispiel PDF-Datei
file_path = 'path/to/your/pdf_file.pdf'
text = read_pdf(file_path)

# Tokenisierung und Sequenzierung
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding der Sequenzen
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Features und Labels erstellen
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Modell erstellen
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(total_words, activation='softmax'))

# Modell kompilieren
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

# Lernrate dynamisch anpassen
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.001)

# Modell trainieren
history = model.fit(xs, ys, epochs=30, batch_size=32, callbacks=[reduce_lr], verbose=1)

# Modell speichern
model.save('lstm_model.h5')

# Beam Search Implementierung
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:k]
    return sequences

def predict_with_beam_search(text, beam_width=3):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0].reshape(1, -1)
    beam_search_results = beam_search_decoder(predicted, beam_width)
    
    text_options = [sequence_to_text(seq[0]) for seq in beam_search_results]
    probabilities = [seq[1] for seq in beam_search_results]
    return text_options, probabilities

def sequence_to_text(seq):
    return ' '.join([tokenizer.index_word[i] for i in seq if i > 0])

# Gradio-Interface
model = tf.keras.models.load_model('lstm_model.h5')

# Definition der Gradio-Komponenten
input_text = gr.inputs.Textbox(lines=2, label="Text Prompt")
predict_button = gr.Button(label="Predict")
next_button = gr.Button(label="Next")
auto_button = gr.Button(label="Auto")
stop_button = gr.Button(label="Stop")
word_choices = gr.Dropdown(label="Select Word", choices=[])

# Funktion zur Vorhersage mit Beam Search
def predict(text):
    predictions, probabilities = predict_with_beam_search(text, beam_width=3)
    choices = [(f"{word} ({prob:.2f})", word) for word, prob in zip(predictions, probabilities)]
    return gr.Dropdown.update(choices=choices, value=None)

# Funktion zur Auswahl des nächsten Wortes
def next_word(text, choice):
    return f"{text} {choice}"

# Automatische Vorhersagefunktion
def auto_predict(text):
    for _ in range(10):
        text += " " + predict_with_beam_search(text, beam_width=3)[0][0]
    return text

# Gradio-Interface erstellen
interface = gr.Interface(
    fn=predict,
    inputs=input_text,
    outputs=word_choices,
    title="Language Model mit LSTM und Beam Search",
    description="Geben Sie einen Text ein und das Modell sagt das nächste Wort voraus. Nutzen Sie Beam Search zur Verbesserung der Vorhersagequalität."
)

# Buttons und Funktionen hinzufügen
interface.add_button("Predict", predict, [input_text], word_choices)
interface.add_button("Next", next_word, [input_text, word_choices], input_text)
interface.add_button("Auto", auto_predict, input_text, input_text)
interface.add_button("Stop", None)

# Interface starten
interface.launch()


