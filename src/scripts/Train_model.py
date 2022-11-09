# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:56:09 2022

@author: gita
"""
import os 
import argparse
from functions import json_to_txt, training_model, characterize_data, upsampling_data
default_path = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/src/scripts"
os.chdir(default_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='New model name')
    parser.add_argument('input_dir', type=str, help='Absolute path input directory')
    parser.add_argument('up_sample_flag', type=bool, help='True if up sample')
    args = parser.parse_args()
    
    #print(args.input_dir)


#path_data_documents="C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Datasets/camara_comercio_NER/gt"

    
    Error = json_to_txt(args.input_dir)
    if type(Error)==int:
        print('Error processing the input docuements, code error {}'.format(Error))
        
    else:
        if args.up_sample_flag:
            entities_dict=characterize_data()
            entities = list(entities_dict.keys())
            entities_to_upsample = [entities[i] for i,value in enumerate(entities_dict.values()) if value < 200]
            upsampling_data(entities_to_upsample, 0.8,  entities)
        Error = training_model(args.model)
        if type(Error)==int:
            print('Error training the model, code error {}'.format(Error))
        else: 
            print('Training complete')
