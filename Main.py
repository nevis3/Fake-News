import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

df = pd.read_csv('news_sample.csv')
#print(df.iloc[0])
#print(df['content'].iloc[0])
content_tokens = word_tokenize(df['content'].values[0])
#print(content)

stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in content_tokens if not w.lower() in stop_words]

for w in content_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)

ps = PorterStemmer()

for w in filtered_sentence:
    print(w, " : ", ps.stem(w))