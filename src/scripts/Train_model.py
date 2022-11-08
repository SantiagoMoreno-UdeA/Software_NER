# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:56:09 2022

@author: gita
"""
import os 
import argparse
from functions import json_to_txt, training_model
default_path = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/src/scripts"
os.chdir(default_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Absolute path input directory')
    args = parser.parse_args()
    
    #print(args.input_dir)


#path_data_documents="C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Datasets/camara_comercio_NER/gt"

    Error = json_to_txt(args.input_dir)
    
    if type(Error)==int:
        print('Error processing the input docuements, code error {}'.format(Error))
        
    else:
        Error = training_model('CCC_trained')
        if type(Error)==int:
            print('Error training the model, code error {}'.format(Error))
        else: 
            print('Training complete')