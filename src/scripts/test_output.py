# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:28:13 2022

@author: gita
"""

import pandas as pd
import os
from flair.data import Sentence

default_path = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/src/scripts"
os.chdir(default_path)

#0a61f100-85f6-11e8-86f2-47a4191dbc0d-0
#3cb4fa20-89cb-11e8-a485-d149999fe64b-0
#2e2a0a20-89f5-11e8-a22e-dfff0e4dee48-0
#4fa73510-8a13-11e8-8585-f31ad517583b-0

path_data = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Datasets/camara_comercio_NER/gt/4fa73510-8a13-11e8-8585-f31ad517583b-0.json "
path_predicted = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/document_tagged.json"


entities = ['ACTIVIDAD','FECHA','PERSONA','EMPRESA','LIMITACION','TIPO_DOCUMENTO_IDENTIDAD','TITULO_FECHA_MATRICULA','DOCUMENTO_IDENTIDAD','TITULO_ACTIVIDAD_PRINCIPAL','TITULO_NOMBRE_O_RAZON_SOCIAL','CARGO_REPRESENTANTE_LEGAL','TITULO_ACTIVIDAD_SECUNDARIA','TITULO_ACTIVIDAD_OTRA','TIPO_ORGANIZACION','CARGO_SUPLENTE','TITULO_ESTADO_MATRICULA','TIPO_ORGANIZACION_ANTIGUO','TITULO_FECHA_VIGENCIA','TITULO_LIMITACIONES','TITULO_TIPO_ORGANIZACION','ESTADO_MATRICULA']
ent_short=['ACT','FCH','PER','ORG','LIM','TDID','TFMT','DID','TACTP', 'TNRS', 'CRL','TACTS','TACTO','TORG', 'CAS', 'TEMT', 'TORGA','TFV','TLIM','TTORG','EMT']
#
dic_original_temp={'tokens':[],'tags':[]}


df = pd.read_json(path_data, orient ='index', encoding='utf-8')[0]
sentences=df['sentences']
tags=df['mentions']
num=0
for s in sentences:
    id_senten=s['id']
    sentence = Sentence(s['text'])
    
    
    for tk in s['tokens']:
        tags_temp=[]
        
        if len(tk['text'])==1:
            #if ord(tk['text'])>=48 and ord(tk['text'])<=57 and ord(tk['text'])>=65 and ord(tk['text'])<=90 and ord(tk['text'])>=97 and ord(tk['text'])<=122:
            dic_original_temp['tokens'].append(tk['text'])
            tk_beg=tk['begin']
            tk_end=tk['end']
            tags_temp.append('O')
            for tg in tags:
                if id_senten == tg['id'].split('-')[0] and tk['begin']>=tg['begin'] and tk['begin']<tg['end']:
                    if tg['type']=='PERSONA' or tg['type']=='PERSONA_NATURAL': tags_temp[-1]='PER'
                    tags_temp[-1]=ent_short[entities.index(tg['type'])]
                    break
            dic_original_temp['tags'].append(tags_temp[-1])

                
        else:
            dic_original_temp['tokens'].append(tk['text'])
            tk_beg=tk['begin']
            tk_end=tk['end']
            tags_temp.append('O')
            for tg in tags:
                if id_senten == tg['id'].split('-')[0] and tk['begin']>=tg['begin'] and tk['begin']<tg['end']:
                    if tg['type']=='PERSONA' or tg['type']=='PERSONA_NATURAL': tags_temp[-1]='PER'
                    tags_temp[-1]=ent_short[entities.index(tg['type'])]
                    break
            dic_original_temp['tags'].append(tags_temp[-1])
        
    
dic_original = {'word':[],'tag':[]}    
for idx in range(len(dic_original_temp['tokens'])):
    token = Sentence(dic_original_temp['tokens'][idx])
    for t in token:
        dic_original['word'].append(t.text)
        dic_original['tag'].append(dic_original_temp['tags'][idx])

    

    
                
df = pd.read_json(path_predicted, orient ='index', encoding='utf-8')[0]      
dic_output = {'word':[],'tag':[]}
sentences=df['sentences']
for s in sentences:
    for tk in s['tokens']:
        dic_output['word'].append(tk['text'])
        dic_output['tag'].append(tk['label'])
   
dic_results={}
for ent in ent_short+['O']: 
    dic_results[ent] = {'recognized':0, 'total':0}
    
for idx in range(len(dic_output['tag'])):
    dic_results[dic_original['tag'][idx]]['total'] += 1
    if dic_original['tag'][idx] == dic_output['tag'][idx]:
        dic_results[dic_original['tag'][idx]]['recognized'] += 1
#%% 
# JSON FORMAT OUTPUT

# Document:{ text:"Texto"
          
#           text_labeled: "Texto \ENTITY"
          
#           sentences:[{ text:"Texto"
          
#           text_labeled: "Texto \ENTITY"
          
#           tokens: [ {text:"Texto", label : "ENTITY"},
#                    {text:"Texto", label : "ENTITY"},
#                    {text:"Texto", label : "ENTITY"}
              
#               ] 
              
#               },
  
#            { text:"Texto"
          
#           text_labeled: "Texto <ENTITY>"
          
#           tokens: [ {text:"Texto", label : "ENTITY"},
#                    {text:"Texto", label : "ENTITY"},
#                    {text:"Texto", label : "ENTITY"}
              
#               ]
 
#               }          
#            ]

#     }

#%% 

# JSON FORMAT INPUT

# json{...
#      sentences:{
#          s:{
#              text:
#              }
#                 }
     
#      ...}