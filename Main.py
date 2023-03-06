import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from cleantext import clean
from collections import Counter
import re


def clean_data(input_text):
    cleaned_text = clean(input_text,  # does not remove special characters such as < , ^ etc.
        normalize_whitespace=True,
        fix_unicode=True,  # fix various unicode errors
        to_ascii=True,  # transliterate to closest ASCII representation
        lower=True,  # lowercase text
        no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
        no_urls=True,  # replace all URLs with a special token
        no_emails=True,  # replace all email addresses with a special token
        no_phone_numbers=True,  # replace all phone numbers with a special token
        no_numbers=True,  # replace all numbers with a special token
        no_digits=True,  # replace all digits with a special token
        no_currency_symbols=True,  # replace all currency symbols with a special token
        no_punct=True,  # remove punctuations
        replace_with_punct="",  # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="<DIGIT>",
        replace_with_currency_symbol="<CUR>",
        lang="en")

    return cleaned_text


df = pd.read_csv('news_sample.csv', encoding='utf8')
result_cleaned_text = clean_data(df.iloc[1]['content'])

def count_replaced_characterization(input_text, regex_filter):
    word_filter_list = []
    print(len(input_text))
    for i in regex_filter:
        words = re.findall(i, input_text)
        word_filter_list.append((i, len(words)))

    for i in word_filter_list:
        input_text = re.sub(i[0], '', input_text)
        input_text = re.sub(' +', ' ', input_text)
    print(len(input_text))
    return input_text


result_replaced_characterization = count_replaced_characterization(result_cleaned_text, ['<url>', '<email>', '<phone>', '<number>', '<digit>', '<cur>'])


def remove_stopwords(input_text):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    input_text = word_tokenize(input_text)

    for w in input_text:
        if w not in stop_words:
            filtered_sentence.append(w)
    print(len(filtered_sentence))
    return filtered_sentence


result_removed_stopwords = remove_stopwords(result_replaced_characterization)


def stem_words(input_text):
    ps = PorterStemmer()
    stemmed_words = []
    input_text = word_tokenize(input_text)

    for w in input_text:
        root_words = ps.stem(w)
        stemmed_words.append(root_words)
    print(len(stemmed_words))


stem_words(result_replaced_characterization)


def count_words(input_text):
    tokenized_text = word_tokenize(input_text)

    counter = Counter(tokenized_text)

