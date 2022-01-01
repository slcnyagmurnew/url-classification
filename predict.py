import pickle
import argparse
from utils import scrape_url, create_test_data
import nltk
import json
# nltk.download('wordnet')


with open('config.json', 'r') as f:
    json_config = json.load(f)
    frequency_path = json_config['frequency_path']
    one_hot = json_config['category_one_hot']


# create_test_data(one_hot)
pickle_in = open(frequency_path, "rb")
words_frequency = pickle.load(pickle_in)

parser = argparse.ArgumentParser(description='URLs for category predictions')
parser.add_argument('--url', help='Predict custom website')

args = parser.parse_args()

if args.url:
    url = args.url
    print(url)
    results = scrape_url(url, words_frequency)
    if results:
        print('Predicted main category:', results[0])
else:
    parser.error("Please specify websites input type")
