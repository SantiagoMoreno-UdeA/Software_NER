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

from functions import use_model, tag_sentence, json_to_txt, training_model, characterize_data, upsampling_data, usage_cuda


models = os.listdir('../../models')

#-------------------------------------------Functions-----------------------------------------------
def get_models():
    return os.listdir('../../models')

def Trainer(fast, model_name, input_dir, Upsampling, Cuda):
    if fast: epochs = 1
    else: epochs = 20
    
    if Cuda: 
        cuda_info = usage_cuda(True)
    else: 
        cuda_info = usage_cuda(False)
    
    yield cuda_info
    
    Error = json_to_txt(input_dir)
    if type(Error)==int:
        yield 'Error processing the input documents, code error {}'.format(cuda_info,Error)
    if Upsampling:
        entities_dict=characterize_data()
        entities = list(entities_dict.keys())
        entities_to_upsample = [entities[i] for i,value in enumerate(entities_dict.values()) if value < 200]
        upsampling_data(entities_to_upsample, 0.8,  entities)
    Error = training_model(model_name, epochs)
    if type(Error)==int:
        yield 'Error training the model, code error {}'.format(cuda_info, Error)
    else: 
        yield 'Training complete, model {} could be found at models/{}'.format(cuda_info, model_name,model_name)


def Tagger_sentence(Model, Sentence, Cuda):
    if Cuda: cuda_info = usage_cuda(True)
    else: cuda_info = usage_cuda(False)

    results = tag_sentence(Sentence, Model)
    if type(results)==int:
        return "Error {}, see documentation".format(results)
    else:
        return results['Highligth']

def Tagger_json(Model, Input_file, Output_file, Cuda):
    results = use_model(Model, Input_file, Output_file, Cuda)
    if type(results)==int:
        error_dict = {}
        return "Error {}, see documentation".format(results), error_dict
    else:
        return { "text" : results['text'], 'entities': results['entities']}, results


#---------------------------------GUI-------------------------------------
if __name__ == '__main__':
    with gr.Blocks(title='Named Entity Recognition by GITA') as demo:
        gr.Markdown("Named Entity Recognition(NER) System. Use Tagger to do NER from a pretrained model, or use Trainer to train a new NER model")
       
        with gr.Tab("Tagger"):
            with gr.Tab("Sentence"):
                with gr.Row():
                    with gr.Column():
                        inputs =[
                             gr.Radio(list(models), label='Model'),
                             gr.Textbox(placeholder="Enter sentence here...", label='Sentence'), 
                             gr.Radio([True,False], label='CUDA', value=False),
                        ]
                    output = gr.HighlightedText()
                tagger_sen = gr.Button("Tag")
                tagger_sen.click(Tagger_sentence, inputs=inputs, outputs=output)
           
            with gr.Tab(".JSON"):
                with gr.Row():
                    with gr.Column():
                        b= gr.Radio(list(models), label='Model')
                        b.change(get_models, outputs=models)
                        inputs =[
                             b,
                             gr.File(label='Input data file'),
                             #gr.Textbox(placeholder="Enter path here...", label='Input data file path'), 
                             gr.Textbox(placeholder="Enter path here...", label='Output data file path'), #value='../../data/Tagged/document_tagged.json'),
                             gr.Radio([True,False], label='CUDA', value=False),
                        ]
                    output = [
                        gr.HighlightedText(),
                        gr.JSON(),
                        ]
                tagger_json = gr.Button("Tag")
                tagger_json.click(Tagger_json, inputs=inputs, outputs=output)
       
        
        with gr.Tab("Trainer"):
            with gr.Row():
                with gr.Column():
                    train_input = inputs =[
                         gr.Radio([True,False], label='Fast training', value=True),
                         gr.Textbox(placeholder="Enter model name here...", label='New model name'),
                         gr.Textbox(placeholder="Enter path here...", label='Input data directory path'), 
                         gr.Radio([True,False], label='Upsampling', value=False),
                         gr.Radio([True,False], label='CUDA', value=False),
                    ]
                train_output = gr.TextArea(placeholder="Output information", label='Output')
            trainer = gr.Button("Train")
    
    
        
        trainer.click(Trainer, inputs=train_input, outputs=train_output)
        
    
    demo.launch(inbrowser=True)