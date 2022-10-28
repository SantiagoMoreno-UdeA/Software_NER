# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:17:50 2022

@author: gita
"""
import numpy as np
import pandas as pd 
from Preprocesamiento import preprocess_tweet
from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('espejelomar/sentece-embeddings-BETO')


data_test = pd.read_csv('trainingandtestdata/testdata.manual.2009.06.14.csv', header = None)

matrix = np.zeros((1,769))
for text, label in zip(data_test[5],data_test[0]):
    if label != 2:
        embeddings = model.encode([preprocess_tweet(text, stop_tweets_flag=False)])
        label = 1 if label == 4 else 0
        array = np.concatenate((embeddings[0], np.asarray([label])))
        matrix = np.vstack((matrix,array))
        
matrix=matrix[1:]

pd.DataFrame(matrix).to_csv('data_test.csv', header=None, index=None) 

k = pd.read_csv('data_test.csv', header=None )
p = k.to_numpy()
X = p.T[:768].T
y = p.T[-1].T

data_train = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', header = None, encoding = 'latin-1')

matrix = np.zeros((1,769))
for text, label in zip(data_train[5],data_test[0]):
    if label != 2:
        embeddings = model.encode([preprocess_tweet(text, stop_tweets_flag=False)])
        label = 1 if label == 4 else 0
        array = np.concatenate((embeddings[0], np.asarray([label])))
        matrix = np.vstack((matrix,array))
        
matrix=matrix[1:]

pd.DataFrame(matrix).to_csv('data_train.csv', header=None, index=None) 