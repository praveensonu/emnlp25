from data_module import convert_raw_data_to_model_qa
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler
from typing import Any
import pandas as pd
import math
import random
import itertools
from transformers import default_data_collator


class VanillaDPODataset(Dataset):
    """
    Dataset class for creating data for forgetting.
    Processes 'question'/'answer' pairs and 'question'/'idk' pairs separately.

    Args:
        forget_data (pd.DataFrame): DataFrame containing 'question', 'answer', and 'idk' columns.
        tokenizer: tokenizer instance to process text
        max_length (int): maximum sequence length
        template_format (str, optional): format template for structuring input
        question_key (str): Column name for the question. Defaults to 'question'.
        answer_key (str): Column name for the answer to forget. Defaults to 'answer'.
        idk_key (str): Column name for the 'I don't know' or alternative response. Defaults to 'idk'.

    Returns:
        A dictionary containing processed data for both the original answer and the idk response:
        {
            'answer_input_ids', 'answer_labels', 'answer_attention_mask',
            'idk_input_ids', 'idk_labels', 'idk_attention_mask'
        }
    """
    def __init__(self, forget_data: pd.DataFrame, tokenizer: Any, 
                 max_length: int, 
                 template_format: str = None,
                 question_key: str = 'question',
                 answer_key: str = 'answer',
                 idk_key: str = 'idk'):
        if not all(k in forget_data.columns for k in [question_key, answer_key, idk_key]):
             raise ValueError(f"forget_data must contain columns: {question_key}, {answer_key}, {idk_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.qk = question_key
        self.ak = answer_key
        self.ik = idk_key

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        row = self.forget_data.iloc[idx]
        q = row[self.qk]
        ans = row[self.ak]
        idk = row[self.ik]

        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, ans,
                                                self.template_format)
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, idk,
                                                self.template_format)

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
        }
    

class ForgetIdkRetainDataset(Dataset):
    """
    For each row in forget_data (must have 'question','answer','idk') and the
    parallel retain_data (must have 'question','answer'), returns a dict:
      {
        'answer_input_ids': …,
        'answer_labels': …,
        'answer_attention_mask': …,
        'idk_input_ids': …,
        'idk_labels': …,
        'idk_attention_mask': …,
        'retain_input_ids': …,
        'retain_labels': …,
        'retain_attention_mask': …,
      }

    Basically, for each sample, it return a dictionary of forget + idk and retain inputs.
    """
    def __init__(
        self,
        forget_data: pd.DataFrame,
        retain_data: pd.DataFrame,
        tokenizer,
        max_length: int,
        template_format: str = None,
        question_key: str = 'question',
        answer_key: str = 'answer',
        idk_key: str = 'idk',
    ):
        # validate
        if not all(col in forget_data.columns for col in [question_key, answer_key, idk_key]):
            raise ValueError(f"forget_data must contain: {question_key}, {answer_key}, {idk_key}")
        if not all(col in retain_data.columns for col in [question_key, answer_key]):
            raise ValueError(f"retain_data must contain: {question_key}, {answer_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.qk, self.ak, self.ik = question_key, answer_key, idk_key

    def __len__(self):
        # assumes forget_data and retain_data are same length
        return len(self.forget_data)

    def __getitem__(self, idx):
        f_row = self.forget_data.iloc[idx]
        r_row = self.retain_data.iloc[idx]

        # forget answer
        q = f_row[self.qk]
        ans = f_row[self.ak]
        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, ans, self.template_format)

        # forget "idk"
        idk = f_row[self.ik]
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, idk, self.template_format)

        # retain answer
        retain_q = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, retain_q, retain_ans, self.template_format)

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
            'retain_input_ids':      ri,
            'retain_labels':         rl,
            'retain_attention_mask': rm,
        }
    


class RetainOnlyDataset(Dataset):
    def __init__(self, retain_df, tokenizer,
                 max_length, template_format = None,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.df = retain_df.reset_index(drop=True)
        self.tk = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q = row[self.qk]
        ans = row[self.ak]

        ri, rl, rm = convert_raw_data_to_model_qa(self.tk, self.max_length, q, ans, self.template_format)

        return {
            'retain_input_ids':      ri,
            'retain_labels':         rl,
            'retain_attention_mask': rm,
        }
    
class TwoStreamBatchSampler(Sampler):
    """
    Yields batches of size B = primary_batch + secondary_bathc
     - primary batch drawn from forget_data
     - secondary batch drawn from retain_data (cyclically)
    cycles the smaller 
    
    """
    def __init__(self, forget_idx, retain_idx, batch_size, primary_batch_size):

        if len(forget_idx) <= len(retain_idx):
            self.primary, self.secondary = list(forget_idx), list(retain_idx)
        else:
            self.primary, self.secondary = list(retain_idx), list(forget_idx)
        
        self.batch_size = batch_size
        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = batch_size - primary_batch_size

    def __iter__(self):
        # shuffle primary every epoch
        prim_order = random.sample(self.primary, len(self.primary))
        # forever cycle secondary
        sec_cycle = itertools.cycle(random.sample(self.secondary, len(self.secondary)))

        for i in range(0, len(prim_order), self.primary_batch_size):
            p_batch = prim_order[i:i+self.primary_batch_size]

            if len(p_batch) < self.primary_batch_size:
                break

            s_batch = [next(sec_cycle) for _ in range(self.secondary_batch_size)]
            batch = p_batch + s_batch
            random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return math.floor(len(self.primary) / self.primary_batch_size)

          

def mixed_collate_fn_dpo(features):
    """
    collates a list of features containing potentially mixed dictionaries
    from forget and retain dataset
    """

    forget_features = []
    retain_features = []

    for feature in features:
        if 'answer_input_ids' in feature:
            forget_features.append({k:v for k,v in feature.items() if isinstance(v, torch.Tensor)})
        elif 'retain_input_ids' in feature:
            retain_features.append({k:v for k,v in feature.items() if isinstance(v, torch.Tensor)})

    collated_batch = {}
    num_forget = len(forget_features)
    num_retain = len(retain_features)

    if num_forget > 0:
        forget_collated = default_data_collator(forget_features)
        collated_batch.update(forget_collated)

    if num_retain > 0:
        retain_collated = default_data_collator(retain_features)
        collated_batch.update(retain_collated)

    collated_batch['num_forget'] = num_forget
    collated_batch['num_retain'] = num_retain

    return collated_batch

        