from data_module import convert_raw_data_to_model_qa
from torch.utils.data import Dataset
from typing import Any
import pandas as pd



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