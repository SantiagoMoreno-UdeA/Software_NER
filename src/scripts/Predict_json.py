# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 00:32:02 2022

@author: sanmo
"""
import os 
default_path = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/src/scripts"
os.chdir(default_path)

import argparse
from functions import use_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('input_data', type=str, help='Absolute path input file')
    parser.add_argument('output_data', type=str, help='Absolute path output file')
    args = parser.parse_args()
    
    #print(args.model, args.input_data, args.output_data)
    Error = use_model(args.model, args.input_data, args.output_data)
    if type(Error)==int:
        print('Tagged not complete, error code {}'.format(Error))
    else:
        print('Tagged complete')

# path_data = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Datasets/camara_comercio_NER/gt/3cb4fa20-89cb-11e8-a485-d149999fe64b-0.json "
# output_dir = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/document_tagged.json"
# sentence = use_model('CCC', path_data, output_dir)
