import pandas as pd
from sklearn.utils import shuffle
from utils import one_url_predict, get_statistics

df = pd.read_csv('dataset/test_category.csv', header=None, usecols=[0,1])

df_test = pd.read_csv('dataset/test_predict.csv', header=None, usecols=[0,1])

category_map = {
        "Health": 0,
        "Sports": 1,
        "Shopping": 2,
        "News_and_Media": 3,
        "Computer_and_Electronics": 4
    }

def split_data(df):
    counts = [0, 0, 0, 0, 0]
    size = 100
    df = shuffle(df)
    new_df = pd.DataFrame()

    for i in range(1, len(df) - 1):
        id = category_map.get(str(df.iloc[i][1]))
        if counts[id] < size:
            new_df = new_df.append(df.iloc[i], ignore_index=True)
            counts[id] += 1
        sum_counts = counts[0] + counts[1] + counts[2] + counts[3]
        if sum_counts >= size * len(counts):
            break
    new_df.to_csv(path_or_buf='dataset/test_predict.csv', index=False)
    print('Out csv written !')
    pass

def get_results(df):
    results = []
    real_classes = []

    for i in range(1, len(df) - 1):
        try:
            predicted_class = one_url_predict(str(df.iloc[i][0]))
            predicted_class_id = category_map.get(predicted_class)
            results.append(predicted_class_id)

            id = category_map.get(str(df.iloc[i][1]))
            real_classes.append(id)
        except Exception as err:
            print(err)
    get_statistics(real_classes, results)

# split_data(df)
# get_results(df_test)
