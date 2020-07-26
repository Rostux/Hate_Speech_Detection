import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import re
import string
import unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

from loaddata import train2, test2
from preprocessing import denoise_text, strip_html, remove_between_square_brackets, remove_stopwords, lowering


print("Before cleaning")
train2.head()
test2.head()

print("after cleaning")


train2['text'] = train2['text'].apply(denoise_text)
train2.head()

test2['text'] = test2['text'].apply(denoise_text)
test2.head()

x_train = train2['text']
x_test = test2['text']
y_train = train2['toxic']
y_test = test2['toxic']
