import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import gradio as gr
import time

input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding der Sequenzen
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Features und Labels erstellen
X = input_sequences[:,:-1]
y = input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Modell erstellen
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modell trainieren
model.fit(X, y, epochs=3, batch_size=32, verbose=1)

# Modell speichern
model.save('word_prediction_model.h5')

# Tokenizer und Modell laden
df = pd.read_csv('Articles.csv')
texts = df['Body'].dropna().tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

max_sequence_len = max([len(tokenizer.texts_to_sequences([text])[0]) for text in texts])

model = tf.keras.models.load_model('word_prediction_model.h5')

# Funktion zur Wortvorhersage
def predict_next_words(text, num_words=1):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        predicted_word = tokenizer.index_word.get(predicted_word_index, "")
        text += " " + predicted_word
    return text

def predict_single_word(prompt):
    token_list = tokenizer.texts_to_sequences([prompt])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    top_indices = np.argsort(predicted[0])[-5:][::-1]
    top_words = [(tokenizer.index_word.get(i, ""), predicted[0][i]) for i in top_indices]
    return top_words

def append_word(prompt, word):
    return f"{prompt} {word}"

def generate_text(prompt, num_words):
    return predict_next_words(prompt, num_words)

class GradioApp:
    def __init__(self):
        self.auto_mode = False

    def predict_next(self, prompt):
        predictions = predict_single_word(prompt)
        return {word: f"{prob:.2f}" for word, prob in predictions}

    def continue_text(self, prompt):
        next_word = predict_single_word(prompt)[0][0]
        new_prompt = f"{prompt} {next_word}"
        predictions = self.predict_next(new_prompt)
        return new_prompt, predictions

    def auto_predict(self, prompt):
        self.auto_mode = True
        while self.auto_mode:
            prompt, predictions = self.continue_text(prompt)
            time.sleep(1)
        return prompt, predictions

    def stop_auto_predict(self):
        self.auto_mode = False

    def reset_text(self):
        return "", ""

app = GradioApp()

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Eingabe", placeholder="Geben Sie Ihren Text ein...")
    prediction_output = gr.Label(label="Vorhersagen")
    
    predict_button = gr.Button("Vorhersage")
    predict_button.click(app.predict_next, inputs=[prompt], outputs=[prediction_output])
    
    continue_button = gr.Button("Weiter")
    continue_button.click(app.continue_text, inputs=[prompt], outputs=[prompt, prediction_output])
    
    auto_button = gr.Button("Auto")
    auto_button.click(app.auto_predict, inputs=[prompt], outputs=[prompt, prediction_output])
    
    stop_button = gr.Button("Stopp")
    stop_button.click(app.stop_auto_predict)
    
    reset_button = gr.Button("Reset")
    reset_button.click(app.reset_text, outputs=[prompt, prediction_output])
    
    demo.launch()
