import re
from pandas.core.arrays.sparse import dtype
import requests
import numpy as np
import json
from seaborn import heatmap
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from nltk.tokenize import word_tokenize  # split sentence to words
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer  # go to word's root (was -> to be)
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

wnl = WordNetLemmatizer()

with open('config.json', 'r') as f:
    json_config = json.load(f)
    headers = json_config['headers']
    frequency = json_config['frequency']
    frequency_path = json_config['frequency_path']
    domains = json_config['domains']
    

stop_words = set(stopwords.words('english'))
with open("dataset/stopwords.txt") as f:
    """
    add desired stop words into global stopwords list.
    """
    for word in f:
        stop_words.add(word.replace('\n', ''))

for tld in domains:
    stop_words.add(tld)


def scrape_url(url, words_frequency):
    """
    Scrapping the url with request's response type.
    :param url: given url for scrapping.
    :param words_frequency: dictionary that has categories and its tokens with frequency order.
    :return:
    """
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            [tag.decompose() for tag in soup("script")]
            [tag.decompose() for tag in soup("style")]
            text = soup.get_text()
            cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
            tokens = word_tokenize(cleaned_text)
            tokens_lemmatize = remove_stopwords(tokens)
            return predict_category(words_frequency, tokens_lemmatize)
        else:
            print(
                f'Error occurred : ({res.status_code}).')
    except Exception as e:
        print(f'Error :\n {e}')
        return False


def predict_category(words_frequency, tokens):
    """
    Get intersection of given tokens and words frequency.
    Find weights of words for each category and add this result into category weights.
    Pull the category index which has maximum category weight.
    :param words_frequency:
    :param tokens:
    :return:
    """
    category_weights = []
    for category in words_frequency:
        weight = 0
        intersect_words = set(words_frequency[category]).intersection(set(tokens))
        for word in intersect_words:
            if word in tokens:
                index = words_frequency[category].index(word)
                weight += frequency - index
        category_weights.append(weight)

    category_index = category_weights.index(max(category_weights))
    main_category = list(words_frequency.keys())[category_index]
    category_weight = max(category_weights)
    # print('weight:', category_weight)
    category_weights[category_index] = 0
    category_index = category_weights.index(max(category_weights))
    main_category_2 = list(words_frequency.keys())[category_index]
    category_weight_2 = max(category_weights)
    return main_category, category_weight, main_category_2, category_weight_2


def remove_stopwords(tokens):
    tokens_list = []
    for word in tokens:
        word = wnl.lemmatize(word.lower())
        if word not in stop_words:
            tokens_list.append(word)
    return list(filter(lambda x: len(x) > 1, tokens_list))


def scrape(props):
    """
    Scrapping the url. Usage in preprocess.
    :param props:
    :return: response of get request.
    """
    i = props[0]
    url = props[1]
    print(i, url)
    try:
        return requests.get(url, headers=headers, timeout=15)
    except:
        return ''


def parse_request(props):
    """
    Parse request result gathered from process executor.
    This function runs when response of request works properly.
    :param props:
    :return: tokens passed from lemmatization, cleaning text and removing stop words operations.
    """
    i = props[0]
    response = props[1]
    if response != '' and response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        [tag.decompose() for tag in soup("script")]
        [tag.decompose() for tag in soup("style")]
        text = soup.get_text()
        # regex: Clean non letter characters from the HTML response (syntax).
        cleaned_text = re.sub('[^a-zA-Z]+', ' ', text).strip()
        # token: split text into tokens.
        tokens = word_tokenize(cleaned_text)
        # lemma: Clean stop words like common words(and, you etc.) from the tokens list.
        tokens_lemmatize = remove_stopwords(tokens)
        return i, tokens_lemmatize
    else:
        return i, ['']


def create_category_features():
    df = pd.read_csv('dataset/url_features.csv')
    url_types = tuple(df['main_category'].unique())
    url_df = pd.DataFrame(url_types, columns=['url_type']) # creating instance of labelencoder
    labelencoder = LabelEncoder() # Assigning numerical values and storing in another column
    url_df['url_category'] = labelencoder.fit_transform(url_df['url_type'])
    url_df['tokens'] = ""

    df = df[df['tokens'].str.len() > 4].reset_index(drop=True)
    for category in df.main_category.unique():
        df_temp = df.loc[df['main_category'] == category]
        print('cat:', category)
        token_list = []
        for row in df_temp['tokens']:
            word_list = ast.literal_eval(row)
            token_list.extend(word_list)
        category_index = url_df[url_df['url_type'] == category].index.values.astype(int)[0]
        url_df.at[category_index, 'tokens'] = list(dict.fromkeys(token_list))
        print(url_df)

    url_df.to_csv('dataset/category_features.csv', index=False)


# def create_model(df):
#     for category in df.url_type:
#         most_common = [word[0] for word in nltk.FreqDist(list(df.tokens)).most_common(frequency)]
#         vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
#         a = vectorizer.fit_transform(most_common)
#         print(a)
#         # words_frequency[category] = most_common


def create_test_data(map_dict):
    """
    returns: dataframe for testing
    """
    df = pd.read_csv('dataset/URL_Classification.csv', header=None, usecols=[1,2])
    df = df[df[2].isin(['Health', 'Sports', 'Shopping', 'News', 'Computers'])]
    df[2] = df[2].replace(map_dict)
    # df.to_csv('dataset/test_category.csv', index=False)
    return df


def one_url_predict(url):
    pickle_in = open(frequency_path, "rb")
    words_frequency = pickle.load(pickle_in)
    results = scrape_url(url, words_frequency)
    return results[0]


def get_statistics(real_data, predicted_data):
    """
    returns statistics of model
    :param real_data: real prediction class data exp: Y_test
    :param predicted_data: predicted  prediction class data
    :return:
    """
    report = classification_report(real_data, predicted_data)
    print("Classification Report:", )
    print(report)

    score = accuracy_score(real_data, predicted_data)
    print("Accuracy:", score)

    cm = confusion_matrix(real_data, predicted_data, normalize='true')
    print(cm)
    cm_df = pd.DataFrame(cm, columns=[0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    heatmap(cm_df, cmap='Blues', annot=True)
    plt.show()
