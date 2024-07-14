import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to clean text
def clean_text(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Paths for files
input_paths = {
    'colab': '/content/Corpus.txt',
    'kaggle': '/kaggle/input/corpus/Corpus.txt',
    'hugface': 'LM_LSTM/Corpus.txt'
}
output_paths = {
    'colab': '/content/Corpus-cleaned.txt',
    'kaggle': '/kaggle/working/Corpus-cleaned.txt',
    'hugface': 'LM_LSTM/Corpus-cleaned.txt'
}
model_paths = {
    'colab': '/content/word_prediction_model.keras',
    'kaggle': '/kaggle/working/word_prediction_model.keras',
    'hugface': 'LM_LSTM/word_prediction_model.keras'
}
tokenizer_paths = {
    'colab': '/content/tokenizer.pickle',
    'kaggle': '/kaggle/working/tokenizer.pickle',
    'hugface': 'LM_LSTM/tokenizer.pickle'
}

# Determine the current environment
def get_current_environment():
    if os.path.exists('/content'):
        return 'colab'
    elif os.path.exists('/kaggle'):
        return 'kaggle'
    else:
        return 'hugface'

current_env = get_current_environment()
input_path = input_paths[current_env]
output_path = output_paths[current_env]
model_path = model_paths[current_env]
tokenizer_path = tokenizer_paths[current_env]

# Preparing data
def prepare_data():
    print(f"Preparing data from: {input_path}")
    if not os.path.exists(output_path):
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()
        cleaned_text = clean_text(text)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
    print(f"Data prepared and saved to: {output_path}")

# Load model and tokenizer
def load_model_and_tokenizer():
    print(f"Loading model from: {model_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    model = None
    tokenizer = None
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Model and tokenizer loaded successfully.")
    else:
        raise FileNotFoundError("Model or tokenizer file not found.")
    return model, tokenizer

# Calculate perplexity
def calculate_perplexity(predictions):
    log_predictions = np.log(predictions + 1e-8)
    perplexity = np.exp(-np.mean(log_predictions))
    return perplexity

# Predict next words with temperature sampling
def predict_next_words(prompt, top_k=5, temperature=1.0):
    print(f"Predicting next words for prompt: {prompt}")
    tokens = tokenizer.texts_to_sequences([prompt])
    padded_seq = pad_sequences(tokens, maxlen=model.input_shape[1] - 1, padding='pre')
    print(f"Padded sequence: {padded_seq}")
    predictions = model.predict(padded_seq)[0]
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    top_indices = np.argsort(predictions)[-top_k:]
    next_word_idx = np.random.choice(top_indices, p=predictions[top_indices])
    next_word = tokenizer.index_word[next_word_idx]
    perplexity = calculate_perplexity(predictions)
    print(f"Next word: {next_word}, Perplexity: {perplexity}")
    return next_word, perplexity

# Gradio interface
def auto_generate(prompt, auto):
    generated_text = prompt
    perplexities = []
    if not auto:
        for _ in range(10):
            next_word, perplexity = predict_next_words(generated_text)
            generated_text += ' ' + next_word
            perplexities.append(perplexity)
    else:
        for _ in range(10):
            next_word, perplexity = predict_next_words(generated_text)
            generated_text += ' ' + next_word
            perplexities.append(perplexity)
            time.sleep(0.2)  # Generate every 0.2 seconds
    print(f"Generated text: {generated_text}")
    return generated_text

# Gradio Blocks Interface
with gr.Blocks() as demo:
    gr.Markdown("## LSTM-based Word Prediction")

    input_text = gr.Textbox(label="Input Text", interactive=True)
    auto_generate_checkbox = gr.Checkbox(label="Automatic Generation every 0.2 seconds", value=False)
    auto_generate_button = gr.Button("Generate Text")
    
    # Add other required buttons
    predict_button = gr.Button("Predict")
    next_button = gr.Button("Next")
    auto_button = gr.Button("Auto")
    stop_button = gr.Button("Stop")
    reset_button = gr.Button("Reset")

    # Define the functions for new buttons
    def predict_text(prompt):
        next_word, perplexity = predict_next_words(prompt)
        return prompt + ' ' + next_word

    def append_next_word(prompt):
        next_word, perplexity = predict_next_words(prompt)
        return prompt + ' ' + next_word

    def reset_text():
        return ""

    auto_generate_button.click(fn=auto_generate, inputs=[input_text, auto_generate_checkbox], outputs=input_text)
    predict_button.click(fn=predict_text, inputs=input_text, outputs=input_text)
    next_button.click(fn=append_next_word, inputs=input_text, outputs=input_text)
    reset_button.click(fn=reset_text, outputs=input_text)

demo.launch()

# Prepare data and load model/tokenizer
prepare_data()
model, tokenizer = load_model_and_tokenizer()
