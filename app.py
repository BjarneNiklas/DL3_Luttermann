import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Constants
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 100
LSTM_UNITS = 100
AUTO_GENERATE_INTERVAL = 0.2  # Interval for auto-generation in seconds

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load the data
text = load_data('Corpus.txt')

# Tokenize the text
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create predictors and label
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the model
model = Sequential([
    Embedding(total_words, EMBEDDING_DIM, input_length=max_sequence_length-1),
    LSTM(LSTM_UNITS, return_sequences=True),
    LSTM(LSTM_UNITS),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (consider removing this if already trained)
# history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# Functions for prediction and text generation
def predict_next_word(text, num_words=5):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(sequence, verbose=0)[0]
    top_n = np.argsort(predicted)[-num_words:][::-1]
    return [(tokenizer.index_word[idx], predicted[idx]) for idx in top_n]

def generate_text(seed_text, num_words=10):
    generated_text = seed_text
    for _ in range(num_words):
        predictions = predict_next_word(generated_text)
        next_word = predictions[0][0]
        generated_text += " " + next_word
        time.sleep(AUTO_GENERATE_INTERVAL)  # Pause for auto generation
    return generated_text

# Gradio interface
def predict(text):
    predictions = predict_next_word(text)
    return {word: float(prob) for word, prob in predictions}

def append_word(text, word):
    return text + " " + word

def auto_generate(text, num_words=10):
    return generate_text(text, num_words)

def reset():
    return "", "", None

def plot_loss():
    fig = px.line(
        x=range(1, len(history.history['loss']) + 1),
        y=[history.history['loss'], history.history['val_loss']],
        labels={'x': 'Epoch', 'y': 'Loss'},
        title='Training and Validation Loss'
    )
    return fig

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Language Model Word Prediction")
    
    with gr.Row():
        input_text = gr.Textbox(label="Input Text")
        output_text = gr.Textbox(label="Generated Text")
    
    with gr.Row():
        predict_btn = gr.Button("Predict")
        continue_btn = gr.Button("Continue")
        auto_btn = gr.Button("Auto")
        reset_btn = gr.Button("Reset")
    
    word_choices = gr.Label(label="Next Word Predictions")
    
    predict_btn.click(predict, inputs=input_text, outputs=word_choices)
    continue_btn.click(append_word, inputs=[input_text, word_choices], outputs=input_text)
    auto_btn.click(auto_generate, inputs=input_text, outputs=output_text)
    reset_btn.click(reset, outputs=[input_text, output_text, word_choices])

    gr.Markdown("""
    ## Documentation
    
    This application uses a stacked LSTM model for next word prediction. The model architecture consists of:
    - An embedding layer
    - Two LSTM layers with 100 units each
    - A dense output layer with softmax activation
    
    The model is trained on the provided dataset using categorical cross-entropy loss and the Adam optimizer.
    
    ## Discussion
    
    The current model demonstrates basic next word prediction capabilities. Areas for improvement include:
    1. Fine-tuning hyperparameters (e.g., LSTM units, embedding dimension)
    2. Experimenting with different model architectures
    3. Implementing more sophisticated text preprocessing
    4. Exploring techniques to prevent overfitting (e.g., dropout, regularization)
    
    ## User Guide
    
    1. Enter a seed text in the "Input Text" box.
    2. Click "Predict" to see the top 5 predicted next words.
    3. Click "Continue" to append the top predicted word.
    4. Click "Auto" to generate 10 words automatically every 0.2 seconds.
    5. Use "Reset" to clear all inputs and outputs.
    """)

if __name__ == "__main__":
    demo.launch()
