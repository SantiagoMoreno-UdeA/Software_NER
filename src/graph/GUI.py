# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:01:08 2022

@author: Santiago Moreno
"""
import os 
import gradio as gr
import sys


default_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(default_path)
sys.path.insert(0, '../scripts')

from functions import use_model, tag_sentence


models = os.listdir('../../models')
def Tagging(Mode, Model, Sentence, Input_file, Output_file, Cuda):
    if Mode == '.JSON':
        use_model(Model, Input_file, Output_file, Cuda)
    elif Mode == 'Sentence':
        return tag_sentence(Sentence, Model, Cuda)
    return 0
demo = gr.Interface(
    Tagging,
    [
         gr.Radio(['Sentence', '.JSON']),
         gr.Radio(list(models)),
        "text", 
        "text", 
        "text",
        gr.Radio([True,False]),
    ],
    "text",
    # examples=[
    #     [5, "add", 3],
    #     [4, "divide", 2],
    #     [-4, "multiply", 2.5],
    #     [0, "subtract", 1.2],
    # ],
    title="Named Entity Recognizer",
    description="Named Entity Recognition system from a pretrained model",
)
demo.launch()