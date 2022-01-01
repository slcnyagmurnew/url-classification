from flask import Flask, render_template, request
import pickle
import argparse
from utils import scrape_url
import nltk
import json
nltk.download('wordnet')
import random


app = Flask(__name__)

with open('config.json', 'r') as f:
    json_config = json.load(f)
    frequency_path = json_config['frequency_path']


pickle_in = open(frequency_path, "rb")
words_frequency = pickle.load(pickle_in)


@app.route('/', methods=['POST', 'GET'])
def index():
    ad = random.randint(1, 5)
    filename = None
    response = None
    if request.method == 'POST':
        url = request.form['url']
        print(url)
        results = scrape_url(url, words_frequency)
        if results:
            response = results[0]
            filename = f'static/images/{response}/{ad}.jpg'
            return render_template('index.html', filename=filename, response=response)
    return render_template('index.html')
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)