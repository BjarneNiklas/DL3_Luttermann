import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
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
file_path = 'article.pdf'
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
model.add(Embedding(total_words, 100))
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

# Callback zur Berechnung der Perplexity
class PerplexityCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        p = np.exp(logs['loss'])
        print(f"Perplexity at epoch {epoch + 1}: {p:.4f}")

perplexity_callback = PerplexityCallback()

# Modell trainieren
history = model.fit(xs, ys, epochs=3, batch_size=32, callbacks=[reduce_lr, perplexity_callback], verbose=1)

# Modell speichern
model.save('lstm_model.keras')

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

# Funktion zur Vorhersage mit Beam Search
def predict(text):
    predictions, probabilities = predict_with_beam_search(text, beam_width=3)
    choices = [f"{word} ({prob:.2f})" for word, prob in zip(predictions, probabilities)]
    return gr.Dropdown.update(choices=choices)

# Funktion zur Auswahl des nächsten Wortes
def next_word(text, choice):
    return f"{text} {choice.split(' ')[0]}"

# Automatische Vorhersagefunktion
def auto_predict(text):
    for _ in range(10):
        text += " " + predict_with_beam_search(text, beam_width=3)[0][0]
    return text

# Funktion zum Stoppen der automatischen Vorhersage
def stop():
    global auto_predicting
    auto_predicting = False
    return "Auto prediction stopped."

# Gradio-Interface erstellen
model = tf.keras.models.load_model('lstm_model.keras')

def update_choices(text):
    choices = predict(text)
    return choices

# Gradio-Komponenten erstellen
with gr.Blocks() as interface:
    textbox = gr.Textbox(lines=2, label="Text Prompt")
    predict_button = gr.Button("Predict")
    next_button = gr.Button("Next")
    auto_button = gr.Button("Auto")
    stop_button = gr.Button("Stop")
    word_choices = gr.Dropdown(label="Select Word")
    text_output = gr.Textbox(label="Generated Text")

    predict_button.click(fn=update_choices, inputs=textbox, outputs=word_choices)
    next_button.click(fn=next_word, inputs=[textbox, word_choices], outputs=text_output)
    auto_button.click(fn=auto_predict, inputs=textbox, outputs=text_output)
    stop_button.click(fn=stop, inputs=None, outputs=text_output)

# Interface starten
interface.launch()
