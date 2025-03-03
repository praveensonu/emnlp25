import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from transformers import pipeline
import torch
from config_eval import Config
import pandas as pd
from utils import update_json_dict, process_quiz_questions


mcq_data = pd.read_csv('/home/praveen/theoden/emnlp_25/dataset/mcq_data.csv')

cfg = Config()

pipe = pipeline('text-generation', model=cfg.model_id, device_map = "auto",
                    model_kwargs={"torch_dtype": torch.bfloat16})


if cfg.exp_type == 'dob':
    print('DOB Evaluation on MCQs')
    dob_questions = mcq_data.loc[mcq_data['standardized_section'] == 'Basic Info']
    retain_questions = mcq_data.loc[mcq_data['standardized_section'] != 'Basic Info']
    print('Processing DOB questions')
    dob_questions = process_quiz_questions(dob_questions, pipe, max_new_tokens=1)
    print('Processing Retain questions')
    retain_questions = process_quiz_questions(retain_questions, pipe, max_new_tokens=1)
    print('Calculating accuracy for DOB and retain')
    dob_accuracy = round((dob_questions['ul_answers'] == 'D').mean()*100, 2)
    retain_accuracy = round((retain_questions['ul_answers'] == 'D').mean()*100, 2)
    print(f'Updating results at {cfg.results_path}')
    results = {
        cfg.loss_type : {
            'exp_type' : cfg.exp_type,
            'accuracy on entity' : dob_accuracy,
            'accuracy on retain' : retain_accuracy
        }
    }

elif cfg.exp_type == 'entity':
    print('Entity Evaluation on MCQs')
    entity_actors =['Jessica Lange', 'Kevin Spacey', 'Renée Zellweger', 'Laurence Olivier', 'Cameron Diaz']
    entity_questions = mcq_data.loc[mcq_data['celebrity'].isin(entity_actors)]
    retain_questions = mcq_data.loc[~mcq_data['celebrity'].isin(entity_actors)]
    print('Processing Entity questions')
    entity_questions = process_quiz_questions(entity_questions, pipe, max_new_tokens=1)
    print('Processing Retain questions')
    retain_questions = process_quiz_questions(retain_questions, pipe, max_new_tokens=1)
    print('Calculating accuracy for Entity and retain')
    entity_accuracy = round((entity_questions['ul_answers'] == 'D').mean()*100, 2)
    retain_accuracy = round((retain_questions['ul_answers'] == 'D').mean()*100, 2)
    print(f'Updating results at {cfg.results_path}')
    results = {
        cfg.loss_type : {
            'exp_type' : cfg.exp_type,
            'accuracy on entity' : entity_accuracy,
            'accuracy on retain' : retain_accuracy
        }
    }


update_json_dict(cfg.results_path, results)