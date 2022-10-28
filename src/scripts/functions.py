# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:46:45 2022

@author: gita
"""
from upsampling import upsampling_ner
#from flair.datasets import ColumnCorpus
#from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from torch.optim.lr_scheduler import OneCycleLR
from flair.data import Sentence
import torch
import pandas as pd
import json

def check_create(path):
    import os
    
    if not (os.path.isdir(path)):
        os.makedirs(path)

def upsampling_data(path_data, probability, entities_to_upsample, entities, methods):
    
    upsampler = upsampling_ner('path_data/train.txt', entities+['O'])
    data, data_labels = upsampler.get_dataset()
    new_samples, new_labels = upsampler.upsampling(entities_to_upsample ,probability,methods)
    data += new_samples
    data_labels += new_labels
    
    AQUI
    # AQUI
    return json


# def training_model(name):
#     data_folder = '../../data/train'
    
#     path_model = '../../models/{}'.format(name)
    
#     columns = {0: 'text', 1:'pos', 2:'ner'}
#     # init a corpus using column format, data folder and the names of the train, dev and test files
#     corpus: Corpus = ColumnCorpus(data_folder, columns,
#                                   train_file='train.txt',
#                                   test_file='test.txt' )
#                                   #dev_file='dev.txt')



            
#     # 2. what tag do we want to predict?
#     tag_type = 'ner'

#     # 3. make the tag dictionary from the corpus
#     #tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
#     tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
            
#     embeddings = TransformerWordEmbeddings(
#                 model='xlm-roberta-large',
#                 layers="-1",
#                 subtoken_pooling="first", 
#                 fine_tune=True,
#                 use_context=True,
#             )


#     # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    
#     tagger = SequenceTagger(
#         hidden_size=256,
#         embeddings=embeddings,
#         tag_dictionary=tag_dictionary,
#         tag_type='ner',
#         use_crf=False,
#         use_rnn=False,
#         reproject_embeddings=False,
#     )

#     # 6. initialize trainer with AdamW optimizer
    

#     trainer = ModelTrainer(tagger, corpus, )

#     # 7. run training with XLM parameters (20 epochs, small LR)

#     trainer.train(path_model,
#                   learning_rate=5.0e-6,
#                   mini_batch_size=4,
#                   mini_batch_chunk_size=1,
#                   max_epochs=50,
#                   scheduler=OneCycleLR,
#                   embeddings_storage_mode='cpu',
#                   optimizer=torch.optim.AdamW,
#                   )

#     print("Model {} trained and saved in {}".format(name,'models/{}'.format(name)))
    
def use_model(name, path_data, output_dir):
    
    path_model = '../../models/{}'.format(name)
    try:
        tagger = SequenceTagger.load(path_model+'/best-model_F.pt')
    except:
        tagger = SequenceTagger.load(path_model+'/final-model_F.pt')
    
    
    data = pd.read_json(path_data, orient ='index', encoding='utf-8')[0]
    sentences=data['sentences']
    results = {'text':"", 'text_labeled':"",'sentences':[]}
    for s in sentences:
        sentence = Sentence(s['text'])
        tagger.predict(sentence)
        if '→' in sentence.to_tagged_string():
            sen_tagged = ' ' .join(sentence.to_tagged_string().split('→')[1][2:-1].split(','))
        else: 
            sen_tagged = sentence.to_tagged_string()[11:-1]
        sen_dict_temp = {'text':sentence.to_plain_string(), 'text_labeled':sen_tagged, 'tokens':[]}
        #return sentence
        for t in sentence.tokens:
            token = {'text':t.text, 'label':t.get_label('ner').value}
            sen_dict_temp['tokens'].append(token)
        results['sentences'].append(sen_dict_temp)
        results['text'] += sentence.to_plain_string() 
        #return sentence
        results['text_labeled'] += sen_tagged
    #return results
    with open(output_dir, "w", encoding='utf-8') as write_file:
        json.dump(results, write_file)

    
       
    
def save_results(model):
    print(s)
    
def see_available_models():
    print(s)
    
def load_data_to_train(data_json):
    print(s)
    
def load_data_to_test(data_json):
    print(s)
    
