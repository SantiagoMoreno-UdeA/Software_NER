# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:24:40 2022

@author: gita
"""
import os 
import gradio as gr
import sys


default_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(default_path)
sys.path.insert(0, '../scripts')

from functions import use_model, tag_sentence

path_data = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Datasets/camara_comercio_NER/gt/3cb4fa20-89cb-11e8-a485-d149999fe64b-0.json "
output_dir = "C:/Users/gita/OneDrive - Universidad de Antioquia/GITA/Maestría/Programas/Software NER/document_tagged.json"


r=tag_sentence('Camara de comercio de Medellin el día 01/11/2021.', 'CCC', True)
Error = use_model('CCC', path_data ,output_dir, True)