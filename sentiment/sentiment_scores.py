# encoding=utf-8

import io
import os
import re
import nltk
import math
import string
import requests
import json
import pickle
import numpy as np
from nltk.corpus import stopwords
from inspect import getsourcefile
from os.path import abspath, join, dirname

from sentiment_analyzer import SentimentIntensityAnalyzer


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


def pre_process(str):
    # do not change the preprocessing order only if you know what you're doing
    str = str.lower()
    str = rm_url(str)
    str = rm_at_user(str)
    str = rm_repeat_chars(str)
    str = rm_hashtag_symbol(str)
    str = rm_time(str)

    return str


def data_preprocessing(data_dir, x_filename):
    # load and process samples
    print('start loading and process samples...')
    texts = []
    with open(os.path.join(data_dir, x_filename), encoding="utf-8") as f:
        for i, line in enumerate(f):
            texts.append(pre_process(line))

    print("Preprocessing is completed")
    return texts


if __name__ == "__main__":
    data_dir = '../data'  # Setting your own file path here.

    x_filename = 'video_text.txt'

    classifier = SentimentIntensityAnalyzer()

    print("Predicting...")
    x = data_preprocessing(data_dir, x_filename)

    with open(os.path.join(data_dir, 'sentiment_scores.txt'), 'w') as f:
        for i in range(len(x)):
            f.write('%s\n' % classifier.polarity_scores(x[i])["compound"])

    print("File created.")