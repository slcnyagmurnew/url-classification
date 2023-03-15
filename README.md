# Url Classification

In this project, website URLs are classified into 5 different categories (Health, Sports, Shopping, News_and_Media, Computer_and_Electronics). GET requests are sent via Flask to URLs in dataset. Multithreading is used for sending thousand of requests and multiprocessing is used for parsing responses simultaneously. The data obtained from the site is passed through NLP preprocesses such as removing stop-words and tokenization. Most common words and their frequencies are used to create different categories. In prediction,  preprocessing operations carried out and the category result is derived from frequency model.