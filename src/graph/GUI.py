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

def Trainer():
    return 0

def Tagger_sentence(Model, Sentence, Cuda):
    results = tag_sentence(Sentence, Model, Cuda)
    if type(results)==int:
        return "Error {}, see documentation".format(results)
    else:
        return results['Highligth']

def Tagger_json(Model, Input_file, Output_file, Cuda):
    results = use_model(Model, Input_file, Output_file, Cuda)
    if type(results)==int:
        return "Error {}, see documentation".format(results)
    else:
        return { "text" : results['text'], 'entities': results['entities']}



with gr.Blocks() as demo:
    gr.Markdown("Named Entity Recognition(NER) System. Use Tagger to do NER from a pretrained model, or use Trainer to train a new NER model")
    with gr.Tab("Tagger"):
        with gr.Tab("Sentence"):
            inputs =[
                 gr.Radio(list(models), label='Model'),
                 gr.Textbox(placeholder="Enter sentence here...", label='Sentence'), 
                 gr.Radio([True,False], label='CUDA'),
            ]
            output = gr.HighlightedText()
            tagger_sen = gr.Button("Tag")
            tagger_sen.click(Tagger_sentence, inputs=inputs, outputs=output)
        with gr.Tab(".JSON"):
            inputs =[
                 gr.Radio(list(models), label='Model'),
                 gr.Textbox(placeholder="Enter sentence here...", label='Input file path'), 
                 gr.Textbox(placeholder="Enter sentence here...", label='Output file path'),
                 gr.Radio([True,False], label='CUDA'),
            ]
            output = gr.HighlightedText()
            tagger_json = gr.Button("Tag")
            tagger_json.click(Tagger_json, inputs=inputs, outputs=output)
    with gr.Tab("Trainer"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        trainer = gr.Button("Train")


    
    trainer.click(Trainer, inputs=image_input, outputs=image_output)
    
    
# demo = gr.Interface(
#     Tagging,
#     [
     
#          gr.Radio(['Sentence', '.JSON']),
#          gr.Radio(list(models)),
#          gr.Textbox(placeholder="Enter sentence here..."), 
#          gr.Textbox(placeholder="Enter sentence here..."), 
#          gr.Textbox(placeholder="Enter sentence here..."),
#          gr.Radio([True,False]),
#     ],
#     gr.HighlightedText(),
#     # examples=[
#     #     [5, "add", 3],
#     #     [4, "divide", 2],
#     #     [-4, "multiply", 2.5],
#     #     [0, "subtract", 1.2],
#     # ],
#     title="Named Entity Recognizer",
#     description="Named Entity Recognition system from a pretrained model",
# )
demo.launch(inline=True, inbrowser=True)