import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr
import time
from tensorflow.keras.models import load_model

# Function to clean text and convert old German spelling to new spelling
def clean_text(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('daß', 'dass').replace('muß', 'muss').replace('müßte', 'müsste').replace('dürfte', 'dürfte').replace('müßte', 'müsste')
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
    if not os.path.exists(output_path):
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()
        cleaned_text = clean_text(text)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

# Train model if it doesn't exist
def train_model():
    with open(output_path, 'r', encoding='utf-8') as file:
        cleaned_text = file.read()
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cleaned_text])
    
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    
    for line in cleaned_text.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(total_words, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32)
    
    model.save(model_path)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return model, tokenizer

# Load model and tokenizer
def load_model_and_tokenizer():
    model = None
    tokenizer = None
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        model, tokenizer = train_model()
    return model, tokenizer

# Calculate perplexity
def calculate_perplexity(predictions):
    log_predictions = np.log(predictions + 1e-8)
    perplexity = np.exp(-np.mean(log_predictions))
    return perplexity

# Predict next words with temperature sampling
def predict_next_words(prompt, top_k=5, temperature=1.0, recent_words=set()):
    tokens = tokenizer.texts_to_sequences([prompt])
    padded_seq = pad_sequences(tokens, maxlen=model.input_shape[1] - 1, padding='pre')
    predictions = model.predict(padded_seq)[0]
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    top_indices = np.argsort(predictions)[-top_k:][::-1]  # Reverse order to get descending probabilities

    next_words = [(tokenizer.index_word[idx], predictions[idx]) for idx in top_indices if tokenizer.index_word[idx] not in recent_words]
    perplexity = calculate_perplexity(predictions)

    return next_words, perplexity

# Gradio interface functions
def predict_text(prompt):
    next_words, perplexity = predict_next_words(prompt)
    top_words = [f"{word} ({prob*100:.2f} %)".replace('.', ',') for word, prob in next_words]
    top_words = [top_words[i:i + 1] for i in range(0, len(top_words), 1)]  # Create rows of 1 word each
    return prompt, f"Perplexity: {perplexity:.4f}".replace('.', ','), top_words

def append_next_word(prompt, recent_words=set()):
    next_words, perplexity = predict_next_words(prompt, recent_words=recent_words)
    next_word = next_words[0][0]
    recent_words.add(next_word)
    if len(recent_words) > 5:  # Limit history length to 5
        recent_words.pop()
    new_prompt = prompt + ' ' + next_word
    top_words = [f"{word} ({prob*100:.2f} %)".replace('.', ',') for word, prob in next_words]
    top_words = [top_words[i:i + 1] for i in range(0, len(top_words), 1)]  # Create rows of 1 word each
    return new_prompt, f"Perplexity: {perplexity:.4f}".replace('.', ','), top_words

def reset_text():
    global stop_signal
    stop_signal = True
    return "", "", []

# Global flag for stopping auto-generation
stop_signal = False

# Auto generation function
def auto_generate_text(prompt):
    global stop_signal
    stop_signal = False
    generated_text = prompt
    recent_words = set()
    for _ in range(10):
        if stop_signal:
            break
        next_words, perplexity = predict_next_words(generated_text, recent_words=recent_words)
        next_word = next_words[0][0]
        recent_words.add(next_word)
        if len(recent_words) > 5:  # Limit history length to 5
            recent_words.pop()
        generated_text += ' ' + next_word
        top_words = [f"{word} ({prob*100:.2f} %)".replace('.', ',') for word, prob in next_words]
        top_words = [top_words[i:i + 1] for i in range(0, len(top_words), 1)]  # Create rows of 1 word each
        yield generated_text, f"Perplexity: {perplexity:.4f}".replace('.', ','), top_words
        time.sleep(0.2)  # Reduce sleep time to 0.2 seconds

# Stop auto-generation
def stop_auto_generation():
    global stop_signal
    stop_signal = True

# Prepare data and load model/tokenizer
prepare_data()
model, tokenizer = load_model_and_tokenizer()

# Gradio Blocks Interface
with gr.Blocks() as demo:
    gr.Markdown("## LSTM-based Word Prediction")

    probabilities = gr.Dataframe(
        label="Gib einen beliebigen Text ein",
        headers=["Wähle nächstes Wort aus"],
        datatype=["str"],
        col_count=1
    )

    input_text = gr.Textbox(label="Input Text", interactive=True)

    predict_button = gr.Button("Vorhersage")
    next_button = gr.Button("Weiter")
    auto_button = gr.Button("Auto")
    stop_button = gr.Button("Stopp")
    reset_button = gr.Button("Reset")

    perplexity_text = gr.Textbox(label="Perplexity", interactive=False)

    predict_button.click(fn=predict_text, inputs=input_text, outputs=[input_text, perplexity_text, probabilities])
    next_button.click(fn=append_next_word, inputs=input_text, outputs=[input_text, perplexity_text, probabilities])
    reset_button.click(fn=reset_text, outputs=[input_text, perplexity_text, probabilities])

    auto_button.click(fn=auto_generate_text, inputs=input_text, outputs=[input_text, perplexity_text, probabilities])
    stop_button.click(fn=stop_auto_generation)

    def select_predicted_word(evt: gr.SelectData, prompt):
        word = evt.value.split(' ')[0]  # Extract word from format "word (probability%)"
        new_prompt = prompt + ' ' + word
        next_words, perplexity = predict_next_words(new_prompt)
        top_words = [f"{word} ({prob*100:.2f} %)".replace('.', ',') for word, prob in next_words]
        top_words = [top_words[i:i + 1] for i in range(0, len(top_words), 1)]  # Create rows of 1 word each
        return new_prompt, f"Perplexity: {perplexity:.4f}".replace('.', ','), top_words

    probabilities.select(fn=select_predicted_word, inputs=[input_text], outputs=[input_text, perplexity_text, probabilities])

demo.launch()
