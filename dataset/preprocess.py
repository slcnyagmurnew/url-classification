import pandas as pd
import os
import sys
import datetime
import pickle
import nltk
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
file_name = 'URL-categorization-DFE.csv'
file_path = os.path.join(sys.path[0], file_name)

"""
Unique main category types:
['Internet_and_Telecom' 'Career_and_Education' 'News_and_Media' 'Science'
 'Gambling' 'Books_and_Literature' 'Pets_and_Animals' 'Health' 'Sports'
 'Computer_and_Electronics' 'Reference' 'Arts_and_Entertainment'
 'Beauty_and_Fitness' 'Adult' 'Shopping' 'Games' 'Autos_and_Vehicles'
 'Law_and_Government' 'Finance' 'Business_and_Industry'
 'Recreation_and_Hobbies' 'Food_and_Drink' 'People_and_Society'
 'Home_and_Garden' 'Travel']
"""

"""
['Arts_and_Entertainment', 'People_and_Society', 'Business_and_Industry',
'Computer_and_Electronics', 'Science', 'Recreation_and_Hobbies', 'Sports',
'Shopping', 'Health', 'Reference', 'Games', 'Adult', 'Home_and_Garden',
'News_and_Media']
"""

# Last categories
"""
['Health', 'Sports', 'Shopping', 'News_and_Media', 'Books_and_Literature',
'Finance', 'Career_and_Education', 'Computer_and_Electronics']
"""


with open('config.json', 'r') as f:
    json_config = json.load(f)
    domains = json_config['domains']
    thread = json_config['thread']
    multiprocessing = json_config['multiprocessing']
    token = json_config['token']
    frequency = json_config['frequency']
    frequency_path = json_config['frequency_path']

df = pd.read_csv(file_path)
df = df[df['main_category'].isin(['Health', 'Sports', 'Shopping', 'News_and_Media', 'Computer_and_Electronics'])]
df = df[['url', 'main_category', 'main_category:confidence']]

print('Selected columns: ', df['main_category'].value_counts())

df['url'] = df['url'].apply(lambda x: 'http://' + x)
df['tld'] = df['url'].apply(lambda x: x.split('.')[-1])
# drops the current index of the DataFrame and replaces it with an index of increasing integers.
df = df[df.tld.isin(domains)].reset_index(drop=True)
df['tokens'] = ''

print("Scraping begins: ", datetime.now())
with ThreadPoolExecutor(thread) as thread_executor:
    """
    Urls in dataframe are sent to scrape function with enumerated values.
    This operation split into 16 threads to increase retrieving url speed.
    It takes ~50 minute to finish scrapping.
    Example: 0 - http://url.com
    """
    start = datetime.now()
    results = thread_executor.map(scrape, [(i, elem) for i, elem in enumerate(df['url'])])
finish_1 = datetime.now()
print(f'Scraping completed in {finish_1-start}.')

print("Analyzing begins: ", datetime.now())
with ProcessPoolExecutor(multiprocessing) as process_executor:
    """
    Single process runs very slowly with this operation.
    To increase analysis of obtained urls' speed, this operation split into 2 processes.
    Urls are mapped to parse request function to get data from urls.
    """
    start = datetime.now()
    response = process_executor.map(parse_request, [(i, elem) for i, elem in enumerate(results)])

for props in response:
    """
    Add result of scrapped url content with tokens into related dataframe location.
    """
    i = props[0]
    tokens = props[1]
    df.at[i, 'tokens'] = tokens
finish_2 = datetime.now()
print(f'Analyzing completed in {finish_2-start}.')

df.to_csv(token, index=False)

words_frequency = {}
for category in df.main_category.unique():
    """
    nltk.FreqDist(all_words): create frequency distribution.
    nltk.FreqDist(all_words).most_common(frequency): returns a list with given frequency number.
    In this list, elements are in ascending order by frequency of each element.
    For each category, tokens are copied into a list named all words.
    All words are put in order by frequency of each word(in nltk.FreqDist).
    Finally, category and its list of words are put into dictionary.
    """
    # print(category)
    all_words = []
    df_temp = df[df.main_category == category]
    for word in df_temp.tokens:
        all_words.extend(word)
    most_common = [word[0] for word in nltk.FreqDist(all_words).most_common(frequency)]
    words_frequency[category] = most_common

# Save word frequency model with pickle
pickle_out = open(frequency_path, "wb")
pickle.dump(words_frequency, pickle_out)
print('Model saved.')
pickle_out.close()