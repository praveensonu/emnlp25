from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import PreTrainedTokenizer, default_data_collator
from typing import Tuple
import math
import pandas as pd
from typing import Dict, List, Set, Tuple, Any



def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    full_text = question + answer
    num_question_tokens = len(tokenizer.tokenize(question, add_special_tokens=False)) #this is important, we 
    encoded = tokenizer(
        full_text,
        add_special_tokens=False, #this is important, we keep false cause we already added the special tokens from template
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    #change label to -100 for question tokens, including assistant header and end of header.
    for i in range(num_question_tokens): label[i] = -100
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class SingleDataset(Dataset):
    def __init__(self, forget_data,
                 tokenizer,
                 max_length=512,
                 question_key = 'question',
                 answer_key = 'answer'):
        """
        Initializes the dataset for gradient ascent finetuning

        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            max_length (int, optional): maximum sequence length for tokenization. Defaults to 512.
            template_format (str, optional): format template for structuring input
        """
        self.data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx][self.qk]
        answer = self.data.iloc[idx][self.ak]
        return convert_raw_data_to_model_qa(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            question=question,
            answer=answer
        )


class DualDatasetRandom(Dataset):
    """
    TOFU way of implementation.

    Args:
        forget_data (pd.DataFrame): DataFrame for forgetting.
        retain_data (pd.DataFrame): DataFrame for retaining.
        tokenizer: tokenizer instance to process text.
        max_length (int): maximum sequence length.
        question_key (str): column name for questions.
        answer_key (str): column name for answers.
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # The forget sample is chosen sequentially by the DataLoader's index.
        forget_idx = idx
        # A new random sample is chosen every time __getitem__ is called.
        retain_idx = torch.randint(0, len(self.retain), (1,)).item()

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)
    

class DualDataset(Dataset):
    """
    Dataset class for creating data for forget and retain (used by gradient difference)

    Args:
        forget_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for forgetting
        retain_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for retaining
        tokenizer: tokenizer instance to process text
        max_length (int): maximum sequence length
        template_format (str, optional): format template for structuring input

    Returns:
        Tuple of forget and retain samples:
        (
            (forget_input_ids, forget_labels, forget_attention_mask),
            (retain_input_ids, retain_labels, retain_attention_mask)
        )
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
    def __len__(self):
        return max(len(self.forget), len(self.retain))

    def __getitem__(self, idx):
        # Cyclic rotation of data
        forget_idx = idx % len(self.forget)
        retain_idx = idx % len(self.retain)

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)



class DualTitleDataset(Dataset):
    """
    Dataset that returns pre-paired forget/retain rows.

    Expects a DataFrame with columns:
      question_forget, answer_forget, question_retain, answer_retain
    (plus any other columns you don’t care about).
    """
    def __init__(
        self,
        paired_df,
        tokenizer,
        max_length,
        question_key: str = "question",
        answer_key: str = "answer"
    ):
        # e.g. paired_df.columns = [
        #   'title', 'question_forget', 'answer_forget',
        #   'question_retain', 'answer_retain', … ]
        self.df = paired_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # pull out the two sides
        q_forget = row[f"{self.qk}_forget"]
        a_forget = row[f"{self.ak}_forget"]
        q_retain = row[f"{self.qk}_retain"]
        a_retain = row[f"{self.ak}_retain"]

        # convert as before
        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q_forget, a_forget
        )
        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q_retain, a_retain
        )

        return (forget_data, retain_data)




# ==== from here the batching type dataset class starts, we dont use these in paper for now ====

def cycle_df_to_length(df, target_len):
    """
    Cyclically repeats rows of a DataFrame to reach a target length.
    If target_len is 0, returns an empty DataFrame with original columns.
    If df is empty and target_len > 0, raises ValueError.
    """
    if target_len == 0:
        return pd.DataFrame(columns=df.columns)

    original_len = len(df)
    if original_len == 0:
        raise ValueError(f"Cannot create {target_len} rows from an empty DataFrame.")
    num_repeats = math.ceil(target_len / original_len)

    padded_df = pd.concat([df] * int(num_repeats), ignore_index=True).iloc[:target_len]
    return padded_df


def _arrange_and_combine_dataframes_internal(
    forget_df: pd.DataFrame,
    retain_df: pd.DataFrame,
    block_size: int,
    n_forget: int,
    n_retain: int
) -> pd.DataFrame:
    """
    Internal helper to combine forget_df and retain_df.
    """
    len_forget_orig = len(forget_df)
    len_retain_orig = len(retain_df)


    if n_forget > 0:
        if len_forget_orig == 0:

            num_blocks_for_forget = 0
        else:
            num_blocks_for_forget = math.ceil(len_forget_orig / n_forget)
    else:
        num_blocks_for_forget = 0


    if n_retain > 0:
        if len_retain_orig == 0:
            num_blocks_for_retain = 0
        else:
            num_blocks_for_retain = math.ceil(len_retain_orig / n_retain)
    else:
        num_blocks_for_retain = 0

    provisional_total_blocks = max(num_blocks_for_forget, num_blocks_for_retain)


    if provisional_total_blocks == 0 :
        total_num_blocks = 0
    elif provisional_total_blocks % block_size == 0:
        total_num_blocks = provisional_total_blocks
    else:
        total_num_units = provisional_total_blocks

        if total_num_units > 0 and total_num_units % (n_forget + n_retain) != 0:
             total_num_units = (total_num_units // (n_forget+n_retain) + 1) * (n_forget+n_retain)


    total_forget_needed = total_num_units * n_forget
    total_retain_needed = total_num_units * n_retain

    # Pad dataframes
    padded_forget_df = cycle_df_to_length(forget_df, total_forget_needed)
    padded_retain_df = cycle_df_to_length(retain_df, total_retain_needed)

    if total_num_units == 0:
        cols = forget_df.columns if not forget_df.empty else (retain_df.columns if not retain_df.empty else [])
        return pd.DataFrame(columns=cols)

    combined_blocks_list = []
    forget_ptr = 0
    retain_ptr = 0

    for _ in range(total_num_units):
        if n_forget > 0:
            combined_blocks_list.append(padded_forget_df.iloc[forget_ptr : forget_ptr + n_forget])
            forget_ptr += n_forget
        if n_retain > 0:
            combined_blocks_list.append(padded_retain_df.iloc[retain_ptr : retain_ptr + n_retain])
            retain_ptr += n_retain

    final_df = pd.concat(combined_blocks_list, ignore_index=True)
    return final_df


class DualBatchDataset(Dataset):
    """
    Dataset class that combines 'forget' and 'retain' data in a specified ratio,
    processes them into a DPO-like format (question/answer and question/idk pairs),
    and includes a 'factor' to distinguish sample types.
    """
    def __init__(self,
                 forget_df: pd.DataFrame,
                 retain_df: pd.DataFrame,
                 tokenizer: Any,
                 max_length: int,
                 block_size: int,
                 n_forget: int,
                 n_retain: int,
                 question_key: str = 'question',
                 answer_key: str = 'answer',
                 idk_key: str = 'idk',
                 factor_key: str = 'factor',
                 title_key: str = 'title'
                 ):

        if n_forget + n_retain != block_size:
            raise ValueError(f"n_forget ({n_forget}) + n_retain ({n_retain}) must equal block_size ({block_size})")
        if n_forget < 0 or n_retain < 0:
             raise ValueError("n_forget and n_retain must be non-negative.")
        required_cols = [question_key, answer_key, idk_key, factor_key, title_key]
        if n_forget > 0:
            if forget_df is None or forget_df.empty:
                raise ValueError("forget_df cannot be empty if n_forget > 0")
            if not all(k in forget_df.columns for k in required_cols):
                 raise ValueError(f"forget_df must contain columns: {required_cols}")
        if n_retain > 0:
            if retain_df is None or retain_df.empty:
                raise ValueError("retain_df cannot be empty if n_retain > 0")
            if not all(k in retain_df.columns for k in required_cols):
                 raise ValueError(f"retain_df must contain columns: {required_cols}")


        _forget_df = forget_df if forget_df is not None else pd.DataFrame(columns=required_cols)
        _retain_df = retain_df if retain_df is not None else pd.DataFrame(columns=required_cols)


        self.combined_data = _arrange_and_combine_dataframes_internal(
            _forget_df, _retain_df, block_size, n_forget, n_retain
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
        self.ik = idk_key
        self.fk = factor_key

        print(f"Combined dataset initialized with {len(self.combined_data)} samples.")
        if not self.combined_data.empty:
            print("Verifying sample structure (first few blocks):")
            num_verify_blocks = min(3, len(self.combined_data) // block_size)
            for i in range(num_verify_blocks):
                start_idx = i * block_size
                actual_n_forget = sum(self.combined_data.iloc[start_idx : start_idx + n_forget][self.fk] < 0)
                actual_n_retain = sum(self.combined_data.iloc[start_idx + n_forget : start_idx + block_size][self.fk] > 0)

                print(f"  Block {i}: {actual_n_forget} forget, {actual_n_retain} retain samples. Expected: {n_forget}, {n_retain}")
                if actual_n_forget != n_forget or actual_n_retain != n_retain :
                    print(f"    WARN: Mismatch in block {i} structure. Got {actual_n_forget} forget, {actual_n_retain} retain.")
                    print(f"    Data in block: \n{self.combined_data.iloc[start_idx : start_idx + block_size][[self.qk, self.fk]]}")


    def __len__(self):
        return len(self.combined_data)
    def __getitem__(self, idx) -> Dict[str, Any]:
        if idx >= len(self.combined_data):
            raise IndexError("Index out of bounds")

        row = self.combined_data.iloc[idx]

        q = str(row[self.qk])
        ans = str(row[self.ak])
        factor = float(row[self.fk])

        input_ids, labels, attention_mask = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, ans,
                                                )

        return (
         input_ids, labels, attention_mask, factor) # because custom_collator_interleaved_ga expects a tuple