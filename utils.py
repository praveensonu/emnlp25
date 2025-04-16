import pandas as pd
from data_module import DualDataset, SingleDataset, VanillaInterleavedDataset, InterleavedDualDataset
import json
from packed_data import SingleDatasetPacked, DualDatasetPacked
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


def load_dataset_from_path(data_path):
    """
    Load and validate a dataset from a CSV file path
    
    Args:
        data_path (str): Path to CSV file with 'question' and 'answer' columns
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe
    """
    df = pd.read_csv(data_path)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("Dataset must contain 'question' and 'answer' columns")
    return df


def create_dual_dataset(forget_path, retain_path, tokenizer, max_length, template_format=None):
    """
    Helper function to create a DualDataset from file paths
    
    Args:
        forget_path (str): Path to forget dataset CSV
        retain_path (str): Path to retain dataset CSV
        tokenizer: Tokenizer instance
        max_length (int): Maximum sequence length
        template_format (str, optional): Format template
        
    Returns:
        DualDataset: Initialized dual dataset
    """
    forget_data = load_dataset_from_path(forget_path)
    retain_data = load_dataset_from_path(retain_path)
    return DualDataset(forget_data, retain_data, tokenizer, max_length, template_format)

def create_vanilla_interleaved_dataset(forget_path, retain_path, tokenizer, max_length, bs, template_format=None):
    """
    Helper function to create a DualDataset with interleaved data from file paths
    
    Args:
        forget_path (str): Path to forget dataset CSV

        retain_path (str): Path to retain dataset CSV
        tokenizer: Tokenizer instance
        max_length (int): Maximum sequence length
        template_format (str, optional): Format template
        
    Returns:
        DualDataset: Initialized dual dataset with interleaved data
    """
    forget_data = load_dataset_from_path(forget_path)
    retain_data = load_dataset_from_path(retain_path)
    return VanillaInterleavedDataset(forget_data, retain_data, tokenizer, max_length, bs, template_format)

def create_interleaved_dual_dataset(forget_path, retain_path, tokenizer, max_length, n, bs, template_format=None):
    """
    Helper function to create a DualDataset with interleaved data from file paths
    
    Args:
        forget_path (str): Path to forget dataset CSV
        retain_path (str): Path to retain dataset CSV
        tokenizer: Tokenizer instance
        max_length (int): Maximum sequence length
        template_format (str, optional): Format template
        
    Returns:
        DualDataset: Initialized dual dataset with interleaved data
    """
    forget_data = load_dataset_from_path(forget_path)
    retain_data = load_dataset_from_path(retain_path)
    return InterleavedDualDataset(forget_data, retain_data, tokenizer, max_length, n, bs,template_format)


def create_single_dataset(data_path, tokenizer, max_length, template_format=None):
    """
    Helper function to create a SingleDataset from file path
    
    Args:
        data_path (str): Path to CSV file
        tokenizer: Tokenizer instance
        max_length (int): Maximum sequence length
        template_format (str, optional): Format template
        
    Returns:
        SingleDataset: Initialized single dataset
    """
    return SingleDataset(data_path, tokenizer, max_length, template_format)



def create_packed_dataset(data_path, tokenizer, template_format=None):
    """
    Helper function to create a QAForgetDataset with constrained packing from file path
    
    Args:
        data_path (str): Path to dataset CSV
        tokenizer: Tokenizer instance
        template_format (str, optional): Format template
        
    Returns:
        QAForgetDataset: Initialized dataset with constrained packing
    """
    return SingleDatasetPacked(data_path, tokenizer, template_format)

def create_packed_dual_dataset(forget_path, retain_path, tokenizer, template_format=None):
    """
    Helper function to create a DualDataset with constrained packing from file paths
    
    Args:
        forget_path (str): Path to forget dataset CSV
        retain_path (str): Path to retain dataset CSV
        tokenizer: Tokenizer instance
        template_format (str, optional): Format template
        
    Returns:
        DualDataset: Initialized dual dataset with constrained packing
    """
    forget_data = load_dataset_from_path(forget_path)
    retain_data = load_dataset_from_path(retain_path)
    return DualDatasetPacked(forget_data, retain_data, tokenizer, template_format)



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