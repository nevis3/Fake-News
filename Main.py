import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from cleantext import clean
from collections import Counter
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('news_cleaned_2018_02_13.csv', encoding='utf8', nrows=100000)


def clean_data(input_text, regex_filter):
    cleaned_text = re.sub(r'(\S+\.com*\S+)', '<url>', input_text)
    cleaned_text = re.sub(r'(\S+\.net*\S+)', '<url>', cleaned_text)
    #cleaned_text = re.sub(r'\|', '', cleaned_text)
    cleaned_text = clean(cleaned_text,  # does not remove special characters such as < , ^ etc.
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
        no_emoji=True,
        replace_with_punct="",  # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="<DIGIT>",
        replace_with_currency_symbol="<CUR>",
        lang="en")

    word_filter_list = []
    #print(len(cleaned_text))

    for i in regex_filter:
        words = re.findall(i, cleaned_text)
        word_filter_list.append((i, len(words)))
    #print(word_filter_list)

    for i in word_filter_list:
        cleaned_text = re.sub(i[0], '', cleaned_text)
        cleaned_text = re.sub(' +', ' ', cleaned_text)
    #print(len(cleaned_text))

    #pre_stop_words = word_tokenize(cleaned_text)

    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    cleaned_text = word_tokenize(cleaned_text)

    for w in cleaned_text:
        if w not in stop_words:
            filtered_sentence.append(w)
    #print(len(filtered_sentence))

    #pre_stemmed_words = filtered_sentence

    ps = PorterStemmer()
    stemmed_words = []
    # input_text = word_tokenize(input_text)

    for w in filtered_sentence:
        root_words = ps.stem(w)
        stemmed_words.append(root_words)
        # print(root_words)
    #print(len(stemmed_words))

    return stemmed_words #, word_filter_list, pre_stop_words, pre_stemmed_words, stemmed_words

word_counter = Counter()
url_counter = [['<url>', 0], ['<email>', 0], ['<phone>', 0], ['<number>', 0], ['<digit>', 0], ['<cur>', 0]]
pre_stopwords_counter = Counter()
pre_stemmed_words = Counter()
end_result = []

for i in range(0, len(df)-1):
    #print(str(df.iloc[i]['type']))
    if str(df.iloc[i]['type']) == "nan" or str(df.iloc[i]['type']) == "unknown":
        #print(df.iloc[i]['type'])
        continue
    data_new = clean_data(df.iloc[i]['content'], ['<url>', '<email>', '<phone>', '<number>', '<digit>', '<cur>'])

    #word_counter += Counter(data_new[0])
    #pre_stopwords_counter += Counter(data_new[2])
    #pre_stemmed_words += Counter(data_new[3])
    type_name = df.iloc[i]['type']
    if type_name in ['unreliable', 'bias', 'clickbait', 'junksci', 'political', 'conspiracy', 'hate', 'rumor', 'satire']:
        type_name = 'fake'
    end_result.append([' '.join(data_new), type_name])


    #for j in range(0, len(data_new[1])):
        #url_counter[j][1] += data_new[1][j][1]

#print(end_result)
item_list = list(word_counter.items())
sorted_list = sorted(item_list, key=(lambda tpl: tpl[1]), reverse=True)

#print(end_result)
#print("cleaned", len(sorted_list))
#print(sorted_list)
#print("<words> lenght", len(url_counter))
#print("<words>", url_counter)
#print("pre stopwords", len(pre_stopwords_counter))
#print("pre stemmed", len(pre_stemmed_words))

#df_processed = pd.DataFrame (sorted_list, columns = ['words', 'amount'])
df_processed_end_results = pd.DataFrame(end_result, columns=['artikler', 'type'])
df_processed_end_results.to_csv('processed.csv')
#print(df_processed_end_results)
#print(df_processed['words'])

"""sorted_final = sorted_list[0:1000]
name_list, count_list = zip(*sorted_final)

fig = plt.figure()
ax = fig.add_axes([0,0,3.5,1])
ax.bar(name_list, count_list)
plt.show()"""

#enc = OneHotEncoder(handle_unknown='ignore')

#X = OneHotEncoder().fit_transform(sorted_list)


X = df_processed_end_results['artikler']
vectorizer = CountVectorizer() #Counts and vectorizes
X = vectorizer.fit_transform(X)

y = df_processed_end_results['type']
encoder = LabelEncoder() #Good for binary use, and sets fake as 0 and reliable as 1
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

#Baseline models
DecisionTree = DecisionTreeClassifier()
LogisticRegression = LogisticRegression(max_iter=1000)
LinearRegression = LinearRegression()

DecisionTree.fit(X_train, y_train)
LogisticRegression.fit(X_train, y_train)

y_pred_decision = DecisionTree.predict(X_test)
y_pred_logistic = LogisticRegression.predict(X_test)

acc_decision = accuracy_score(y_test, y_pred_decision)
acc_logistic = accuracy_score(y_test, y_pred_logistic)

print(acc_decision)
print(acc_logistic)

# print("X:", X_train, X_test, X_val)
# print("y:", y_train, y_test, y_val)
#typecounter = Counter(df['type'])

#print(set(df['type']))
#print(set(df_processed_end_results['type']))
#print(typecounter)

