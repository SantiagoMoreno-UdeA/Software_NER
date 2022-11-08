# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:46:45 2022

@author: gita
"""
from upsampling import upsampling_ner
from flair.datasets import ColumnCorpus
from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from torch.optim.lr_scheduler import OneCycleLR
from flair.data import Sentence
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import torch
import pandas as pd
import json
import os
import operator
import flair

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


def training_model(name):
    flair.device = torch.device('cpu') 
    data_folder = data_folder = '../../data/train'
    path_model = '../../models/{}'.format(name)
    
    columns = {0: 'text', 1:'ner'}
    # init a corpus using column format, data folder and the names of the train, dev and test files
    try:
        corpus: Corpus = ColumnCorpus(data_folder, columns,
                                      train_file='train.txt',
                                      test_file='test.txt' )
                                      #dev_file='dev.txt')
    except: 
        print('Invalid input document in training')
        return 8



            
    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    #tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
            
    try:
        embeddings = TransformerWordEmbeddings(
                    model='xlm-roberta-large',
                    layers="-1",
                    subtoken_pooling="first", 
                    fine_tune=True,
                    use_context=True,
                )
    except: 
        print('Error while loading embeddings form RoBERTa')
        return 5

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    
    try:
        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type='ner',
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
        )
    except: 
        print('Error making tagger')
        return 6

    # 6. initialize trainer with AdamW optimizer
    

    trainer = ModelTrainer(tagger, corpus)

    # 7. run training with XLM parameters (20 epochs, small LR)
    try:
        trainer.train(path_model,
                      learning_rate=5.0e-6,
                      mini_batch_size=1,
                      mini_batch_chunk_size=1,
                      max_epochs=50,
                      scheduler=OneCycleLR,
                      embeddings_storage_mode='cpu',
                      optimizer=torch.optim.AdamW,
                      )
    except: 
        print('Error training the model')
        return 7
    
    print("Model {} trained and saved in {}".format(name,'models/{}'.format(name)))
    
def use_model(name, path_data, output_dir):
    
    #--------------Load the trained model-------------------------
    path_model = '../../models/{}'.format(name)
    
    if not (os.path.isdir(path_model)): 
        print('Model does not exists')
        return 10
        
    if not os.path.isfile(path_data): 
        print('Input file does not exists')
        return 9 
    
    try:
        tagger = SequenceTagger.load(path_model+'/best-model.pt')
    except:
        try:
            tagger = SequenceTagger.load(path_model+'/final-model.pt')
        except: 
            print('Invalid model')
            return 0
    
    #-----------------Load the document-------------------------
    try:
        data = pd.read_json(path_data, orient ='index', encoding='utf-8')[0]
    except: 
        print('Can\'t open the input file')
        return 2
    
    if len(data) <= 0:
        print(f"length of document greater than 0 expected, got: {len(data)}")
        return 2
    
    try:
        sentences=data['sentences']
        t = sentences[0]['text']
    except: 
        print('Invalid JSON format in document {}'.format(path_data))
        return 3
    
    
    #-----------------Tagged the document-------------------------
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
        
    #-----------------Save the results-------------------------
    with open(output_dir, "w", encoding='utf-8') as write_file:
        json.dump(results, write_file)

    
       
def json_to_txt(path_data_documents):
    #-------------List the documents in the path------------
    documents=os.listdir(path_data_documents)
    if len(documents) <= 0:
        print('There are not documents in the folder')
        return 4
    
    data_from_documents={'id':[],'document':[],'sentence':[],'word':[],'tag':[]}
    
    #--------------Verify each documment-------------
    for num,doc in enumerate(documents):
        data=path_data_documents+'/'+doc
        df = pd.read_json(data, orient ='index')[0]
        try:
            sentences = df['sentences']
            t = sentences[0]['text']
            t = sentences[0]['id']
            t = sentences[0]['tokens']
            j = t[0]['text']
            j = t[0]['begin']
            j = t[0]['end']
            tags = df['mentions']
            if tags:
                
                tg = tags[0]['id']
                tg = tags[0]['begin']
                tg = tags[0]['end']
                tg = tags[0]['type']
                print('Final')
        except: 
            print('Invalid JSON input format in document {}'.format(doc))
            return 3
            
       
        #-----------------Organize the data----------------
        for s in sentences:
            id_senten=s['id']
            for tk in s['tokens']:
                if len(tk['text'])==1:
                    #if ord(tk['text'])>=48 and ord(tk['text'])<=57 and ord(tk['text'])>=65 and ord(tk['text'])<=90 and ord(tk['text'])>=97 and ord(tk['text'])<=122:
                    tk_beg=tk['begin']
                    tk_end=tk['end']
                    data_from_documents['id'].append('d'+str(num)+'_'+id_senten)
                    data_from_documents['document'].append(doc)
                    data_from_documents['word'].append(tk['text'])
                    data_from_documents['sentence'].append(s['text'])
                    data_from_documents['tag'].append('O')
                    for tg in tags:
                        if id_senten == tg['id'].split('-')[0] and tk['begin']>=tg['begin'] and tk['begin']<tg['end']:
                            data_from_documents['tag'][-1]=tg['type']
                            break
                        
                else:
                    tk_beg=tk['begin']
                    tk_end=tk['end']
                    data_from_documents['id'].append('d'+str(num)+'_'+id_senten)
                    data_from_documents['document'].append(doc)
                    data_from_documents['word'].append(tk['text'])
                    data_from_documents['sentence'].append(s['text'])
                    data_from_documents['tag'].append('O')
                    for tg in tags:
                        if id_senten == tg['id'].split('-')[0] and tk['begin']>=tg['begin'] and tk['begin']<tg['end']:
                            data_from_documents['tag'][-1]=tg['type']
                            break

    X=np.array(data_from_documents['word'])
    y=np.array(data_from_documents['tag'])     
    groups=np.array(data_from_documents['id'])  
    
    
    #-------------------Save the data in CONLL format--------------
    group_kfold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    group_kfold.get_n_splits(X, y, groups)
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train, groups_test = groups[train_index], groups[test_index]
        break


                    

    X_write=[X_train,X_test]
    y_write=[y_train,y_test]
    groups_write=[groups_train, groups_test]
    archivos=['train','test']
    
    
    for k in range(2):
        X_temp = X_write[k]
        y_temp = y_write[k]
        groups_temp = groups_write[k]
        arch=archivos[k]
        id_in=groups_temp[0]
        
            
        data_folder  = '../../data/train'
        check_create(data_folder)
        count = 0
        with open('../../data/train/{}.txt'.format(arch), mode='w', encoding='utf-8') as f:
            for i in range(len(X_temp)):
                if groups_temp[i] != id_in:
                    id_in=groups_temp[i]
                    f.write('\n')
                    count = 0

                count += 1
                f.write(X_temp[i]+' '+ y_temp[i])
                f.write('\n')
                
                if count >= 50: 
                    count = 0
                    f.write('\n')

            

                        

