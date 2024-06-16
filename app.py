import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import gradio as gr

# Daten laden
df = pd.read_csv('Articles.csv')
texts = df['text'].tolist()

# Tokenisierung und Sequenzierung
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

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

# Funktion zur Wortvorhersage
def predict_next_words(text, num_words=1):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        predicted_word = tokenizer.index_word[predicted_word_index]
        text += " " + predicted_word
    return text

# Gradio-Interface
def predict_single_word(prompt):
    token_list = tokenizer.texts_to_sequences([prompt])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    top_indices = np.argsort(predicted[0])[-5:][::-1]
    top_words = [(tokenizer.index_word[i], predicted[0][i]) for i in top_indices]
    return top_words

def append_word(prompt, word):
    return f"{prompt} {word}"

def generate_text(prompt, num_words):
    return predict_next_words(prompt, num_words)

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Eingabe", placeholder="Geben Sie Ihren Text ein...")
    prediction_output = gr.Label(label="Vorhersagen")
    
    def predict_next(prompt):
        predictions = predict_single_word(prompt)
        prediction_output.update(value={word: f"{prob:.2f}" for word, prob in predictions})
    
    predict_button = gr.Button("Vorhersage")
    predict_button.click(predict_next, inputs=[prompt], outputs=[prediction_output])
    
    def continue_text(prompt):
        next_word = predict_single_word(prompt)[0][0]
        prompt.update(value=f"{prompt.value} {next_word}")
        predict_next(prompt.value)
    
    continue_button = gr.Button("Weiter")
    continue_button.click(continue_text, inputs=[prompt], outputs=[prompt, prediction_output])
    
    auto_mode = False
    
    def auto_predict(prompt):
        nonlocal auto_mode
        auto_mode = True
        while auto_mode:
            continue_text(prompt)
            time.sleep(1)  # Kleine Verzögerung zwischen den Vorhersagen
    
    def stop_auto_predict():
        nonlocal auto_mode
        auto_mode = False
    
    auto_button = gr.Button("Auto")
    auto_button.click(auto_predict, inputs=[prompt], outputs=[prompt, prediction_output])
    
    stop_button = gr.Button("Stopp")
    stop_button.click(stop_auto_predict)
    
    def reset_text():
        prompt.update(value="")
        prediction_output.update(value="")
    
    reset_button = gr.Button("Reset")
    reset_button.click(reset_text)
    
    demo.launch()
