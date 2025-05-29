import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import pandas as pd
import math
from transformers import default_data_collator


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
                 question_key: str = 'question',
                 answer_key: str = 'answer',
                 idk_key: str = 'idk'):
        if not all(k in forget_data.columns for k in [question_key, answer_key, idk_key]):
             raise ValueError(f"forget_data must contain columns: {question_key}, {answer_key}, {idk_key}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
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
                                                )
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, idk,
                                                )

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
        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, ans)

        # forget "idk"
        idk = f_row[self.ik]
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, q, idk)

        # retain answer
        retain_q = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(self.tokenizer, self.max_length, retain_q, retain_ans)

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


class TitleForgetIdkRetainDataset(Dataset):
    """
    Expects a single DataFrame with columns:
      question_forget, answer_forget, idk_forget,
      question_retain, answer_retain

    Returns, for each row, a dict with tokenized inputs/labels/masks for:
      - forget-answer
      - forget-idk
      - retain-answer
    """
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int,
        # Override if column names differ:
        q_forget_key: str = 'question_forget',
        a_forget_key: str = 'answer_forget',
        idk_forget_key: str = 'idk_forget',
        q_retain_key: str = 'question_retain',
        a_retain_key: str = 'answer_retain',
    ):
        required = [
            q_forget_key, a_forget_key, idk_forget_key,
            q_retain_key, a_retain_key
        ]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # store the keys
        self.qf, self.af, self.ifk = q_forget_key, a_forget_key, idk_forget_key
        self.qr, self.ar = q_retain_key, a_retain_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---- forget-answer ----
        qf = row[self.qf]
        af = row[self.af]
        fa_input_ids, fa_labels, fa_attention_mask = \
            convert_raw_data_to_model_qa(self.tokenizer, self.max_length, qf, af)

        # ---- forget-idk ----
        idkf = row[self.ifk]
        fi_input_ids, fi_labels, fi_attention_mask = \
            convert_raw_data_to_model_qa(self.tokenizer, self.max_length, qf, idkf)

        # ---- retain-answer ----
        qr = row[self.qr]
        ar = row[self.ar]
        ra_input_ids, ra_labels, ra_attention_mask = \
            convert_raw_data_to_model_qa(self.tokenizer, self.max_length, qr, ar)

        return {
            # forget-answer
            'answer_input_ids':      fa_input_ids,
            'answer_labels':         fa_labels,
            'answer_attention_mask': fa_attention_mask,
            # forget-idk
            'idk_input_ids':         fi_input_ids,
            'idk_labels':            fi_labels,
            'idk_attention_mask':    fi_attention_mask,
            # retain-answer
            'retain_input_ids':      ra_input_ids,
            'retain_labels':         ra_labels,
            'retain_attention_mask': ra_attention_mask,
        }



class CyclicForgetIdkRetainDataset(Dataset):
    """
    Cycles through the *shorter* split so that every row of the *longer*
    split is visited exactly once per epoch.  In the common case where
    retain_data is larger, you iterate over retain_data sequentially and
    wrap around forget_data via idx % len(forget_data).
    """
    def __init__(
        self,
        forget_data: pd.DataFrame,
        retain_data: pd.DataFrame,
        tokenizer,
        max_length: int,
        question_key: str = 'question',
        answer_key: str = 'answer',
        idk_key: str = 'idk',
    ):
        # validation
        req_f = {question_key, answer_key, idk_key}
        req_r = {question_key, answer_key}
        if not req_f.issubset(forget_data.columns):
            raise ValueError(f"forget_data must contain: {', '.join(req_f)}")
        if not req_r.issubset(retain_data.columns):
            raise ValueError(f"retain_data must contain: {', '.join(req_r)}")

        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk, self.ak, self.ik = question_key, answer_key, idk_key
        self.f_len = len(self.forget_data)
        self.r_len = len(self.retain_data)

    def __len__(self):
        """Length is the *longer* split so that we see every row once."""
        return max(self.f_len, self.r_len)

    def _row(self, df, idx, modulo_len):
        """Helper to get a row with modulo wrap-around."""
        return df.iloc[idx % modulo_len]

    def __getitem__(self, idx):
        f_row = self._row(self.forget_data, idx, self.f_len)
        r_row = self._row(self.retain_data, idx, self.r_len)

        q = f_row[self.qk]
        ans = f_row[self.ak]
        ai, al, am = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q, ans
        )

        idk = f_row[self.ik]
        ii, il, im = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q, idk
        )

        retain_q   = r_row[self.qk]
        retain_ans = r_row[self.ak]
        ri, rl, rm = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, retain_q, retain_ans
        )

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



# ================= Update 5/11/2025 code =================
# =========== doing interleaving outside the dataset class through interleaving the dataframe itself =======

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


class CombinedForgetRetainDataset(Dataset):
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
                f_count = self.combined_data.iloc[start_idx : start_idx + n_forget][self.fk].nunique()
                r_count = self.combined_data.iloc[start_idx + n_forget : start_idx + block_size][self.fk].nunique()
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
        raw_idk = row[self.ik]
        if isinstance(raw_idk, bool):
            idk = "I don't know." if raw_idk else "This is known."
        else:
            idk = str(raw_idk)

        factor = float(row[self.fk])

        ai, al, am = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, ans,
                                                )
        ii, il, im = convert_raw_data_to_model_qa(self.tokenizer,
                                                self.max_length,
                                                q, idk,
                                                )

        return {
            'answer_input_ids':      ai,
            'answer_labels':         al,
            'answer_attention_mask': am,
            'idk_input_ids':         ii,
            'idk_labels':            il,
            'idk_attention_mask':    im,
            'factor':                factor,
            'original_index':        torch.tensor(idx, dtype=torch.long)
        }