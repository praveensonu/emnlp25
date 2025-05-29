import pandas as pd
from data_module import  SingleDataset
import json
import os
import torch

def write_json(data_path, logs):
    with open(data_path, 'w') as f:
        json.dump(logs, f, indent=4)

def read_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def update_json_dict(data_path, new_results):
    # Check if the file exists; if not, start with an empty dictionary
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    
    # Expecting new_results to be a dictionary
    if not isinstance(new_results, dict):
        raise ValueError("new_results must be a dictionary when updating a dict-based JSON")
    
    # Merge new_results into the existing data
    data.update(new_results)
    
    # Write the updated dictionary back to the file
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def process_quiz_questions(df, pipe, max_new_tokens=1):
    """
    Process a DataFrame of quiz questions, format each question with its options,
    and use a language model to generate answers.
    
    Args:
        dob_questions (pandas.DataFrame): DataFrame containing quiz questions with columns
                                         'mcq_raw_options', 'mcq_question', and 'ul_answers'
        pipe (function): The language model pipeline function to generate answers
        
    Returns:
        pandas.DataFrame: The input DataFrame with the 'ul_answers' column updated
    """
    df['ul_answers'] = ''
    for i, row in df.iterrows():
        options = row['mcq_raw_options']
        choices = ''.join(options).replace('[','').replace(']','').replace('\'','').replace(',','')
        questions = row['mcq_question']
        instruct = f"Choose the correct answer from the options below. Answer with a single letter of the correct choice either A, B, C or D. Question:{questions} Options:{choices}"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": instruct},
        ]
        
        outputs = pipe(messages, max_new_tokens=max_new_tokens)
        df.at[i, 'ul_answers'] = outputs[0]['generated_text'][-1]['content']
    
    return df