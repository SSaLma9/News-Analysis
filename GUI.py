import re
import string
from bs4 import BeautifulSoup
from contractions import fix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import json
import pandas as pd
import numpy as np

import tensorflow as tf
from transformers import TFBertModel
from transformers import BertTokenizer

import tkinter as tk
from tkinter import filedialog, messagebox, Text, Scrollbar, VERTICAL, RIGHT, Y, END

import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from transformers import BertTokenizer

# Initialize the main Tkinter window
window = tk.Tk()
window.title("Sentiment Analysis System")
window.geometry("800x600")

# Global variables for data
history = []

# Initialize NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()


# Load the trained model
custom_objects = {'TFBertModel': TFBertModel}
model = tf.keras.models.load_model('my_model02.h5', custom_objects=custom_objects)

def preprocess_text(text):
    """ Preprocess text by removing HTML tags, stopwords, punctuation, stemming, etc. """
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [porter_stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

def load_batch_file():
    """ Load CSV or text file for batch processing """
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    if file_path:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            with open(file_path, 'r') as file:
                data = file.readlines()
            df = pd.DataFrame(data, columns=['text'])

        df['processed_text'] = df['text'].apply(preprocess_text)
        batch_analysis(df['text'].tolist(), df['processed_text'].tolist())
    else:
        messagebox.showerror("Error", "Failed to load file.")

def batch_analysis(original_texts, preprocessed_texts):
    """ Perform batch analysis on uploaded data """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_mask = tokenize_and_pad(preprocessed_texts, tokenizer)

    # Create a dictionary for inputs
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])  # Ensure your model's predict method accepts this format

    labels = ['negative', 'positive', 'neutral']
    predicted_labels = [labels[np.argmax(pred)] for pred in predictions]
    confidence_scores = [max(tf.nn.softmax(pred)) for pred in predictions]

    for i in range(len(original_texts)):
        add_to_history(original_texts[i], predicted_labels[i], confidence_scores[i].numpy())

def single_text_analysis():
    """ Perform analysis on a single user input text """
    input_text = user_input.get()

    if input_text.strip():
        preprocessed_text = preprocess_text(input_text)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_ids, attention_mask = tokenize_and_pad([preprocessed_text], tokenizer)

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        predictions = model.predict([inputs['input_ids'], inputs['attention_mask']])

        labels = ['negative', 'positive', 'neutral']
        predicted_label = labels[np.argmax(predictions)]
        confidence_score = max(tf.nn.softmax(predictions[0]))

        add_to_history(input_text, predicted_label, confidence_score.numpy())
        messagebox.showinfo("Result", f"Sentiment: {predicted_label} (Confidence: {confidence_score:.2f})")
    else:
        messagebox.showerror("Error", "Please enter some text.")


def load_batch_file():
    """ Load CSV or text file for batch processing """
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                if 'text' not in df.columns:
                    messagebox.showerror("Error", "CSV file must contain a 'text' column.")
                    return
            else:
                with open(file_path, 'r') as file:
                    data = file.readlines()
                df = pd.DataFrame(data, columns=['text'])

            df['processed_text'] = df['text'].apply(preprocess_text)
            batch_analysis(df['text'].tolist(), df['processed_text'].tolist())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    else:
        messagebox.showerror("Error", "Failed to load file.")


def tokenize_and_pad(texts, tokenizer, max_length=40):
    inputs = tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='tf')
    return inputs['input_ids'], inputs['attention_mask']

def add_to_history(text, sentiment, confidence):
    """ Add analysis results to history """
    history.append({'text': text, 'sentiment': sentiment, 'confidence': confidence})
    display_history()

def display_history():
    """ Display the history of analyzed texts and results """
    text_area.delete(1.0, END)
    for entry in history:
        text_area.insert(END, f"Text: {entry['text']}\n")
        text_area.insert(END, f"Sentiment: {entry['sentiment']}, Confidence: {entry['confidence']:.2f}\n")
        text_area.insert(END, "-" * 60 + "\n")

# GUI Components

# Single Text Input
tk.Label(window, text="Enter text for analysis:").pack(pady=10)
user_input = tk.Entry(window, width=100)
user_input.pack(pady=10)

analyze_button = tk.Button(window, text="Analyze Single Text", command=single_text_analysis)
analyze_button.pack(pady=10)

# Batch File Input
batch_button = tk.Button(window, text="Upload Batch File (CSV or Text)", command=load_batch_file)
batch_button.pack(pady=10)

# History Display
tk.Label(window, text="Analysis History:").pack(pady=10)
text_area = Text(window, wrap="word")
text_area.pack(expand=True, fill="both", padx=10, pady=10)

# Scrollbar for Text Area
scrollbar = Scrollbar(text_area, orient=VERTICAL)
scrollbar.pack(side=RIGHT, fill=Y)
text_area.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=text_area.yview)

# Start the Tkinter event loop
window.mainloop()
