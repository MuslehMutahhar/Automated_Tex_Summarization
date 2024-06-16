
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import spacy
from heapq import nlargest
from collections import Counter
from transformers import pipeline
import logging

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load transformers pipeline for abstractive summarization
summarizer = pipeline("summarization", model='t5-base', tokenizer='t5-base', framework='pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extractive_summary', methods=['POST'])
def extractive_summary():
    try:
        text = request.form['text']
        num_sentences = int(request.form['num_sentences'])
        
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text != '\n']
        
        word_freq = Counter(tokens)
        max_freq = max(word_freq.values())
        for word in word_freq.keys():
            word_freq[word] = word_freq[word] / max_freq

        sent_token = [sent.text for sent in doc.sents]
        sent_score = {}
        for sent in sent_token:
            for word in sent.split():
                if word.lower() in word_freq.keys():
                    if sent not in sent_score.keys():
                        sent_score[sent] = word_freq[word.lower()]
                    else:
                        sent_score[sent] += word_freq[word.lower()]

        summary_sentences = nlargest(num_sentences, sent_score, key=sent_score.get)
        extractive_summary = " ".join(summary_sentences)

        return jsonify({'summary': extractive_summary})
    except Exception as e:
        logging.error(f"Error in extractive summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/abstractive_summary', methods=['POST'])
def abstractive_summary():
    try:
        text = request.form['text']
        max_length = int(request.form['max_length'])
        min_length = int(request.form['min_length'])
        
        logging.info(f"Text: {text[:50]}...")  
        logging.info(f"Max Length: {max_length}, Min Length: {min_length}")
        
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        abstractive_summary = summary[0]['summary_text']

        return jsonify({'summary': abstractive_summary})
    except Exception as e:
        logging.error(f"Error in abstractive summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
