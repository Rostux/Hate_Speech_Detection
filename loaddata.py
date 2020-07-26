import numpy as np
import pandas as pd

test2 = pd.read_csv('../input/hate-speech-detection/toxic_test.csv')
train2 = pd.read_csv('../input/hate-speech-detection/toxic_train.csv')


del test2['Unnamed: 0']
del train2['Unnamed: 0']
train2['text'] = train2['comment_text']
test2['text'] = test2['comment_text']
del train2['comment_text']
del test2['comment_text']
