import io
import os
import re
import nltk
import json
import numpy as np

from os.path import abspath, join, dirname
from joblib import load
from nltk.corpus import stopwords

# Additional sklearn imports for SVM/Bayes + Feature Selection
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif

from sentiment_analyzer import SentimentIntensityAnalyzer
import sentiment_analyzer

if __name__ == "__main__":
    data_dir = '../data'  # Setting your own file path here.

    x_filename = 'tweets.txt'
    tweets = data_preprocessing(data_dir, x_filename)

    print("Loading data...")
    x = np.array(tweets)

    senti_classifier = SentimentIntensityAnalyzer()
    def get_senti_features(x):
        scores = [list(senti_classifier.polarity_scores(instance).values()) for instance in x]
        for score in scores:
            score[-1] += 1  # normalize to 0, 2 scale
        return np.array(scores)

    # Put features together
    feats_union = FeatureUnion([ 
        ('tfidf', TfidfVectorizer()),
        ('senti', FunctionTransformer(get_senti_features, validate=False)),
    ])

    x_feats = feats_union.fit_transform(x)
    f_selector = SelectPercentile(f_classif, percentile=60)
    f_selector.fit(x_feats, y)
    x_feats = f_selector.transform(x_feats).toarray()
    print(x_feats.shape)

    classifier = SVC(C=1, gamma=1)
    model = load('text_senti.joblib')
    predicts = model.predict(x_feats)

    with open(os.path.join(data_dir, 'text_sent_scores.txt'), 'w') as file:
        for predict in predicts:
            file.write('%s\n' %predict)

    print('Done predicting!\nOutput in text_sent_scores.txt')


def data_preprocessing(data_dir, x_filename):
    # addition
    stops = set(stopwords.words('english'))
    stops.add('rt') # can remove

    words_stat = {}  # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    tweets = []
    with open(os.path.join(data_dir, x_filename), encoding='utf-8') as f:
        for i, line in enumerate(f):
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            content = tweet_obj['text'].replace("\n", " ")

            postprocess_tweet = []
            words = preprocess(content)

            for word in words:
                if word not in stops:
                    postprocess_tweet.append(word)
                    if word in words_stat.keys():
                        words_stat[word][0] += 1
                        if i != words_stat[word][2]:
                            words_stat[word][1] += 1
                            words_stat[word][2] = i
                    else:
                        words_stat[word] = [1,1,i]
            tweets.append(' '.join(postprocess_tweet))

    print("Preprocessing is completed")
    return tweets


def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>', re.S)
    return html_prog.sub('', str)


def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)


def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)


def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)


def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)


def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)


def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

def rm_punctuation(current_tweet):
    return re.sub(r'[^\w\s]','',current_tweet)

porter = nltk.PorterStemmer()

def preprocess(str):
    str = str.lower()
    str = rm_url(str)
    str = rm_at_user(str)
    str = rm_repeat_chars(str)
    str = rm_hashtag_symbol(str)
    str = rm_time(str)

    try:
        str = nltk.tokenize.word_tokenize(str)
        try:
            str = [porter.stem(t) for t in str]
        except:
            print(str)
            pass
    except:
        print(str)
        pass
        
    return str