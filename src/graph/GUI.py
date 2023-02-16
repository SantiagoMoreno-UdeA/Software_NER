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
sys.path.insert(0, default_path+'/../scripts')

from src.scripts.functions import use_model, tag_sentence, json_to_txt, training_model, characterize_data, upsampling_data, usage_cuda, copy_data


models = os.listdir(default_path+'/../../models')

#-------------------------------------------Functions-----------------------------------------------


def Trainer(fast, model_name, standard, input_dir, Upsampling, Cuda):
    if fast: epochs = 1
    else: epochs = 20
    
    if Cuda: 
        cuda_info = usage_cuda(True)
    else: 
        cuda_info = usage_cuda(False)
    
    
    if standard:
        copy_data(input_dir)
    else:
        Error = json_to_txt(input_dir)
        if type(Error)==int:
            yield 'Error processing the input documents, code error {}'.format(Error)
    if Upsampling:
        yield cuda_info+'\n'+'-'*20+'Upsampling'+'-'*20
        entities_dict=characterize_data()
        entities = list(entities_dict.keys())
        entities_to_upsample = [entities[i] for i,value in enumerate(entities_dict.values()) if value < 200]
        upsampling_data(entities_to_upsample, 0.8,  entities)
        yield '-'*20+'Training'+'-'*20
    else:
        yield cuda_info+'\n'+'-'*20+'Training'+'-'*20
    Error = training_model(model_name, epochs)
    if type(Error)==int:
        yield 'Error training the model, code error {}'.format(Error)
    else: 
        yield 'Training complete, model {} could be found at models/{}'.format(model_name,model_name)


def Tagger_sentence(Model, Sentence, Cuda):
    if Cuda: cuda_info = usage_cuda(True)
    else: cuda_info = usage_cuda(False)
    yield cuda_info+'\n'+'-'*20+'Tagging'+'-'*20
    results = tag_sentence(Sentence, Model)
    if type(results)==int:
        yield "Error {}, see documentation".format(results)
    else:
        yield results['Highligth']

def Tagger_json(Model, Input_file, Output_file, Cuda):
    if Cuda: cuda_info = usage_cuda(True)
    else: cuda_info = usage_cuda(False)
    
    yield cuda_info+'\n'+'-'*20+'Tagging'+'-'*20, {}
    
    results = use_model(Model, Input_file.name, Output_file)
    if type(results)==int:
        error_dict = {}
        yield "Error {}, see documentation".format(results), error_dict
    else:
        yield { "text" : results['text'], 'entities': results['entities']}, results


#---------------------------------GUI-------------------------------------
def execute_GUI():
    with gr.Blocks(title='NER', css="#title {font-size: 150% } #sub {font-size: 120% } ") as demo:
        
        gr.Markdown("Named Entity Recognition(NER) by GITA and PRATECH.",elem_id="title")
        gr.Markdown("Software developed by Santiago Moreno, Daniel Escobar, Rafael Orozco",elem_id="sub")
        gr.Markdown("Named Entity Recognition(NER) System.")
        gr.Markdown("Use Tagger to apply NER from a pretrained model in a sentence or a given document in JSON format.")
        gr.Markdown("Use Trainer to train a new NER model from a directory of documents in JSON format.")
        with gr.Tab("Tagger"):
            with gr.Tab("Sentence"):
                with gr.Row():
                    with gr.Column():
                        b = gr.Radio(list(models), label='Model')
                        inputs =[
                             b,
                             gr.Textbox(placeholder="Enter sentence here...", label='Sentence'), 
                             gr.Radio([True,False], label='CUDA', value=False),
                        ]
                        tagger_sen = gr.Button("Tag")
                    output = gr.HighlightedText()
                
           
                
                tagger_sen.click(Tagger_sentence, inputs=inputs, outputs=output)
                b.change(fn=lambda value: gr.update(choices=list(os.listdir('../../models'))), inputs=b, outputs=b)
                gr.Examples(
                
                    examples=[
                        ['CCC',"Camara de comercio de medellín. El ciudadano JAIME JARAMILLO VELEZ identificado con C.C. 12546987 ingresó al plantel el día 1/01/2022"],
                        ['CCC',"Razón Social GASEOSAS GLACIAR S.A.S, ACTIVIDAD PRINCIPAL fabricación y distribución de bebidas endulzadas"]
                     ],
                    inputs=inputs
                    )
      
               
            with gr.Tab("Document"):
                with gr.Row():
                    with gr.Column(): 
                        c = gr.Radio(list(models), label='Model')
                        inputs =[
                             c,
                             gr.File(label='Input data file'),
                             gr.Textbox(placeholder="Enter path here...", label='Output data file path'), #value='../../data/Tagged/document_tagged.json'),
                             gr.Radio([True,False], label='CUDA', value=False),
                        ]
                        tagger_json = gr.Button("Tag")
                    output = [
                        gr.HighlightedText(),
                        gr.JSON(),
                        ]
                
                tagger_json.click(Tagger_json, inputs=inputs, outputs=output)
                c.change(fn=lambda value: gr.update(choices=list(os.listdir('../../models'))), inputs=c, outputs=c)
                
         
        with gr.Tab("Trainer"):
            with gr.Row():
                with gr.Column():
                    train_input = inputs =[
                         gr.Radio([True,False], label='Fast training', value=True),
                         gr.Textbox(placeholder="Enter model name here...", label='New model name'),
                         gr.Radio([True,False], label='Standard input', value=False),
                         gr.Textbox(placeholder="Enter path here...", label='Input data directory path'), 
                         gr.Radio([True,False], label='Upsampling', value=False),
                         gr.Radio([True,False], label='CUDA', value=False),
                    ]
                    trainer = gr.Button("Train")
                train_output = gr.TextArea(placeholder="Output information", label='Output')
            
    
    
        
        trainer.click(Trainer, inputs=train_input, outputs=train_output)
        

        
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8080,inbrowser=True)


