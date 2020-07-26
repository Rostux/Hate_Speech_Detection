from embedding import embedding_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import text, sequence
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
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
rom wordcloud import WordCloud, STOPWORDS


max_features = 10000
maxlen = 300
epochs = 10
embed_size = 100
filters = 250
kernel_size = 3
hidden_dims = 250


model2 = Sequential()
model2.add(Embedding(max_features, output_dim=embed_size, weights=[
           embedding_matrix], input_length=maxlen, trainable=False))

model2.add(Dropout(0.2))
model2.add(Conv1D(filters,
                  kernel_size,
                  padding='valid',
                  activation='relu'))
model2.add(MaxPooling1D())
model2.add(Conv1D(filters,
                  5,
                  padding='valid',
                  activation='relu'))
model2.add(GlobalMaxPooling1D())
model2.add(Dense(hidden_dims, activation='relu'))
model2.add(Dropout(0.2))

model2.add(Dense(1, activation='sigmoid'))
model2.summary()
model2.compile(loss='binary_crossentropy',
               optimizer='adam', metrics=['accuracy'])
