import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from transformers import pipeline
import torch
from config_eval import Config_eval
import pandas as pd
from cloze_util import process_cloze_questions, compute_mean_scores
from utils import update_json_dict

cfg = Config_eval()

cloze_data = pd.read_csv(cfg.data_path)

pipe = pipeline('text-generation', model=cfg.model_id, device_map = "auto",
        model_kwargs={"torch_dtype": torch.bfloat16})


if cfg.exp_type == 'dob':
    print('DOB Evaluation on Cloze')
    dob_questions = cloze_data.loc[cloze_data['section'] == 'Basic Info']
    retain_questions = cloze_data.loc[cloze_data['section'] != 'Basic Info']
    print('Processing DOB questions')
    dob_questions = process_cloze_questions(dob_questions, pipe)
    print('Processing Retain questions')
    retain_questions = process_cloze_questions(retain_questions, pipe)
    print('Calculating exact match, similarity and bleu score for forget ')
    results_forget = compute_mean_scores(dob_questions)
    print('Calculating exact match, similarity and bleu score for retain')
    results_retain = compute_mean_scores(retain_questions)
    print(f'Updating results at {cfg.results_path}')
    results = {
        cfg.loss_type : {
            'exp_type' : cfg.exp_type,
            'results_forget' : results_forget,
            'results_retain' : results_retain
        }
    }

elif cfg.exp_type == 'entity':
    print('Entity Evaluation on Cloze')
    entity_actors =['Jessica Lange', 'Kevin Spacey', 'Renée Zellweger', 'Laurence Olivier', 'Cameron Diaz']
    entity_questions = cloze_data.loc[cloze_data['celebrity'].isin(entity_actors)]
    retain_questions = cloze_data.loc[~cloze_data['celebrity'].isin(entity_actors)]
    print('Processing Entity questions')
    entity_questions = process_cloze_questions(entity_questions, pipe)
    print('Processing Retain questions')
    retain_questions = process_cloze_questions(retain_questions, pipe)
    print('Calculating exact match, similarity and bleu score for forget ')
    results_forget = compute_mean_scores(entity_questions)
    print('Calculating exact match, similarity and bleu score for retain')
    results_retain = compute_mean_scores(retain_questions)
    print(f'Updating results at {cfg.results_path}')
    results = {
        cfg.loss_type : {
            'exp_type' : cfg.exp_type,
            'results_forget' : results_forget,
            'results_retain' : results_retain
        }
    }

update_json_dict(cfg.results_path, results)