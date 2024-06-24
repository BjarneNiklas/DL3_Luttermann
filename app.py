import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 100
LSTM_UNITS = 100

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    
    # Print column names to help identify the correct column
    print("Columns in the CSV file:", data.columns)
    
    # Assume the text is in the first column if 'text' is not present
    text_column = 'text' if 'text' in data.columns else data.columns[0]
    
    return ' '.join(data[text_column].astype(str))

# Load the data
try:
    text = load_data('Articles.csv')
except Exception as e:
    print(f"Error loading data: {str(e)}")
    print("Please check your CSV file and ensure it contains the necessary text data.")
    exit(1)

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
model.summary()

# Train the model
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# ... (rest of the code remains the same)

if __name__ == "__main__":
    demo.launch()