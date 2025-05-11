from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import PreTrainedTokenizer, default_data_collator
from typing import Tuple
import math
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
import itertools
import random


def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    
    messages = [{"role": "user", "content": question}]
    new_question = tokenizer.apply_chat_template(
        messages,
        tokenizer = False,
        add_generataion_prompt=True
    )
    
    full_text = str(new_question) + answer
    num_question_tokens = len(tokenizer.tokenize(str(new_question), add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
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

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class SingleDataset(Dataset):
    def __init__(self, data_path, 
                 tokenizer, 
                 max_length=512, 
                 template_format=None, 
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
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
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
            answer=answer,
            template_format=self.template_format
        )


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
    def __init__(self, forget_data, retain_data, tokenizer, max_length, template_format=None,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
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
            self.template_format
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
            self.template_format
        )

        return (forget_data, retain_data)


class BasicGradDiffDataset(Dataset):
    """
    Combines a forget-df and a retain-df (each with 'question' & 'answer'),
    tokenizes both, and returns:
      {
        'input_ids': Tensor,
        'labels': Tensor,
        'attention_mask': Tensor,
        'factor': Tensor(-1 or +1)
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
    ):
        # validate
        for df_name, df in [('forget_data', forget_data), ('retain_data', retain_data)]:
            if not all(col in df.columns for col in [question_key, answer_key]):
                raise ValueError(f"{df_name} must contain columns: {question_key}, {answer_key}")

        # tag factors
        forget_df = forget_data[[question_key, answer_key]].copy()
        forget_df['factor'] = -1.0
        retain_df = retain_data[[question_key, answer_key]].copy()
        retain_df['factor'] = 1.0

        # merge
        self.data = pd.concat([forget_df, retain_df], ignore_index=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        q, ans, factor = row[self.qk], row[self.ak], float(row['factor'])

        input_ids, labels, attention_mask = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, q, ans, self.template_format
        )
        return (
            input_ids,
            labels,
            attention_mask,
            torch.tensor(factor, dtype=torch.float),
        )



class VanillaInterleavedDataset(Dataset):
    """
    Baseline Dataset class creating interleaved data with roughly equal numbers
    of forget and retain samples per block/batch.

    Each sample is a tuple:
        (input_ids, labels, attention_mask, fraction)
    where fraction is -1 for forget examples and 1 for retain examples.

    The samples are interleaved in blocks of size `bs`. In each block, roughly
    the first half (`bs // 2`) samples are taken from the forget dataset and
    the second half (`bs - bs // 2`) samples are taken from the retain dataset.

    For example, if bs=8, the output sequence of samples within a block will be:
        [forget, forget, forget, forget, retain, retain, retain, retain]

    If bs=7, it will be:
        [forget, forget, forget, retain, retain, retain, retain]

    Both the forget and retain selections are performed cyclically.

    Args:
        forget_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for forgetting.
        retain_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for retaining.
        tokenizer: Tokenizer instance to process text.
        max_length (int): Maximum sequence length.
        bs (int): Block (or "batch") size. Determines the interleaving structure (bs//2 forget, bs - bs//2 retain).
        template_format (str, optional): Template for structuring the input.
        question_key (str, optional): Column name for question. Defaults to 'question'.
        answer_key (str, optional): Column name for answer. Defaults to 'answer'.
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length, bs,
                 template_format=None, question_key='question', answer_key='answer'):

        if not isinstance(bs, int) or bs < 2:
            raise ValueError("bs (block/batch size) must be an integer >= 2 for interleaving.")

        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.qk = question_key
        self.ak = answer_key
        self.bs = bs # Block size

        self.num_forget = len(self.forget)
        self.num_retain = len(self.retain)

        # Determine samples per block
        self.n_forget_per_block = self.bs // 2
        self.n_retain_per_block = self.bs - self.n_forget_per_block # Handles odd bs correctly

        if self.num_forget == 0 and self.n_forget_per_block > 0:
             print(f"Warning: Forget dataset is empty, but block size {bs} expects {self.n_forget_per_block} forget samples per block.")
             # Adjust effective forget per block if empty
             self.n_forget_per_block = 0
        if self.num_retain == 0 and self.n_retain_per_block > 0:
             print(f"Warning: Retain dataset is empty, but block size {bs} expects {self.n_retain_per_block} retain samples per block.")
             # Adjust effective retain per block if empty
             self.n_retain_per_block = 0

        # Recalculate effective block size if datasets were empty
        self.effective_bs = self.n_forget_per_block + self.n_retain_per_block
        if self.effective_bs == 0:
             print("Warning: Both forget and retain datasets appear empty or block size is invalid. Dataset length will be 0.")
             self.num_blocks = 0
        else:
             # Calculate number of blocks needed
             num_forget_blocks = 0
             if self.num_forget > 0 and self.n_forget_per_block > 0:
                 num_forget_blocks = math.ceil(self.num_forget / self.n_forget_per_block)

             num_retain_blocks = 0
             if self.num_retain > 0 and self.n_retain_per_block > 0:
                 num_retain_blocks = math.ceil(self.num_retain / self.n_retain_per_block)

             self.num_blocks = max(num_forget_blocks, num_retain_blocks)

        print(f"Vanilla Dataset Info: bs={bs}, n_forget={self.n_forget_per_block}, n_retain={self.n_retain_per_block}")
        print(f"Num Forget Samples: {self.num_forget}, Num Retain Samples: {self.num_retain}")
        print(f"Calculated num_blocks: {self.num_blocks}")


    def __len__(self):
        # Total number of samples is the number of blocks multiplied by the *effective* block size.
        return self.num_blocks * self.effective_bs # Use effective_bs in case one dataset was empty

    def __getitem__(self, idx):
        if self.effective_bs == 0:
            raise IndexError("Dataset is effectively empty.")

        # Determine which block we are in and the position within that *effective* block.
        block_index = idx // self.effective_bs
        pos_in_effective_block = idx % self.effective_bs

        if pos_in_effective_block < self.n_forget_per_block:
            # Forget sample
            if self.num_forget == 0:
                 raise RuntimeError("Attempting to fetch forget sample but forget dataset is empty.") # Should not happen if init logic is correct
            # Calculate the overall forget index based on blocks and position within forget part
            effective_idx = (block_index * self.n_forget_per_block + pos_in_effective_block) % self.num_forget
            sample_row = self.forget.iloc[effective_idx]
            factor = -1.0
        else:
            # Retain sample
            if self.num_retain == 0:
                 raise RuntimeError("Attempting to fetch retain sample but retain dataset is empty.") # Should not happen
            # Calculate the overall retain index based on blocks and position within retain part
            pos_in_retain_part = pos_in_effective_block - self.n_forget_per_block
            effective_idx = (block_index * self.n_retain_per_block + pos_in_retain_part) % self.num_retain
            sample_row = self.retain.iloc[effective_idx]
            factor = 1.0

        # Process the text using the conversion function
        processed_sample = convert_raw_data_to_model_qa( # Assuming this exists and works
            self.tokenizer,
            self.max_length,
            sample_row[self.qk],
            sample_row[self.ak],
            self.template_format
        )

        input_ids, labels, attention_mask = processed_sample
        return (input_ids, labels, attention_mask, factor)
    



class InterleavedDualDataset(Dataset):
    def __init__(self, forget_data, retain_data, tokenizer, max_length, n, bs,
                 template_format=None, question_key='question', answer_key='answer'):

        if not isinstance(n, int) or not isinstance(bs, int) or n <= 0 or bs <= 0:
             raise ValueError("n and bs must be positive integers.")
        if n >= bs:
            raise ValueError(f"n (forget samples={n}) must be strictly less than bs (block/batch size={bs})")

        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.qk = question_key
        self.ak = answer_key
        self.n = n
        self.bs = bs # please calculate the bs before passing it to the class, it is per_device_train_batch_size * ngpus * grad accumulation steps
        self.num_forget = len(self.forget)
        self.num_retain = len(self.retain)
        self.retain_per_block = self.bs - self.n

        if self.num_forget == 0:
            print("Warning: Forget dataset is empty.")
        if self.num_retain == 0:
             print("Warning: Retain dataset is empty. Cannot interleave.")
             if self.retain_per_block > 0:
                 print("Warning: Cannot create retain samples as retain dataset is empty.")

        # Calculate number of blocks needed to cover the longer effective sequence
        num_forget_blocks = 0
        if self.num_forget > 0 and self.n > 0:
             num_forget_blocks = math.ceil(self.num_forget / self.n)

        num_retain_blocks = 0
        if self.num_retain > 0 and self.retain_per_block > 0:
             num_retain_blocks = math.ceil(self.num_retain / self.retain_per_block)

        self.num_blocks = max(num_forget_blocks, num_retain_blocks)
        print(f"Dataset Info: n={n}, bs={bs}, num_forget={self.num_forget}, num_retain={self.num_retain}")
        print(f"Calculated num_blocks: {self.num_blocks} (based on {num_forget_blocks} forget vs {num_retain_blocks} retain blocks)")

    def __len__(self):
        # Total number of samples is the number of blocks multiplied by the block size.
        return self.num_blocks * self.bs

    def __getitem__(self, idx):
        if self.num_blocks == 0: # Handle case where datasets were empty
             raise IndexError("Dataset is empty.")

        block_index = idx // self.bs
        pos_in_block = idx % self.bs

        if pos_in_block < self.n:
            # Forget sample
            if self.num_forget == 0:
                 # This should ideally not be reached if len is calculated correctly
                 # but indicates an issue if forget is empty but len > 0
                 raise RuntimeError("Trying to get forget sample, but forget dataset is empty.")
            effective_idx = (block_index * self.n + pos_in_block) % self.num_forget # Correct index for n > 1
            sample_row = self.forget.iloc[effective_idx]
            factor = -1.0 # Use float
        else:
            # Retain sample
            if self.num_retain == 0 or self.retain_per_block <= 0:
                 # Should not happen if len is correct and n < bs
                 raise RuntimeError("Trying to get retain sample, but retain dataset is empty or n>=bs.")
            # Correct index for retain part
            effective_idx = (block_index * self.retain_per_block + (pos_in_block - self.n)) % self.num_retain
            sample_row = self.retain.iloc[effective_idx]
            factor = 1.0 # Use float

        # Process the text using the conversion function.
        processed_sample = convert_raw_data_to_model_qa( # Make sure this returns tensors or collator handles lists
            self.tokenizer,
            self.max_length,
            sample_row[self.qk],
            sample_row[self.ak],
            self.template_format
        )

        # Assuming processed_sample = (input_ids_tensor, labels_tensor, attention_mask_tensor)
        input_ids, labels, attention_mask = processed_sample
        # Return tuple that collator expects
        return (input_ids, labels, attention_mask, factor)


## This is the Dataset class for batching similar forget and retain samples together.
class PairedTitleDataset(Dataset):
    def __init__(self, forget_data: pd.DataFrame, retain_data: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer, max_length: int, n: int, bs: int,
                 template_format: str = None, title_key: str = 'title',
                 question_key: str = 'question', answer_key: str = 'answer'):
        """
        Dataset that prepares the forget and retain samples for training.
        Basically, we have some forget samples and retain samples. They have question, answer and title columns.
        We want to create a dataset that has the forget and retain samples in a way that they are interleaved, and each batch
        consists of n forget samples and bs - n retain samples. The retain samples are selected based on the title of the forget samples.
        If there are different titles against forget titles, we randomly select forget title samples and batch them (n samples) with retain samples.

        Args:
            # ... (Args description same as before) ...
        """
        if not isinstance(n, int) or not isinstance(bs, int) or n <= 0 or bs <= 0:
             raise ValueError("n and bs must be positive integers.")
        if n >= bs:
            raise ValueError(f"n (forget samples={n}) must be strictly less than bs (block/batch size={bs})")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.title_k = title_key
        self.qk = question_key
        self.ak = answer_key
        self.n = n
        self.bs = bs
        self.retain_per_batch = self.bs - self.n

        # --- Preprocessing: Assign Title IDs ---
        print("Preprocessing data and assigning title IDs...")
        self.forget_data = forget_data.reset_index(drop=True)
        self.retain_data = retain_data.reset_index(drop=True)

        # 1. Assign IDs to forget titles (1 to N)
        unique_forget_titles = self.forget_data[self.title_k].unique()
        self.title_to_id: Dict[str, int] = {title: i + 1 for i, title in enumerate(unique_forget_titles)}
        self.forget_data['title_id'] = self.forget_data[self.title_k].map(self.title_to_id)
        self.forget_title_ids: Set[int] = set(self.title_to_id.values())
        print(f"Found {len(self.forget_title_ids)} unique forget titles.")

        if not self.forget_title_ids:
            raise ValueError("Forget data is empty or contains no unique titles. Cannot assign random IDs.")
        # Create a list of forget title IDs for random selection
        self.forget_title_id_list = list(self.forget_title_ids)

        # 2. Assign IDs to retain titles (use existing if match forget, else N+1 onwards)
        self.extra_retain_title_ids: Set[int] = set()
        next_extra_id = len(self.forget_title_ids) + 1
        retain_title_ids = []
        retain_titles_map = {} # Keep track of titles already assigned an extra ID
        for title in self.retain_data[self.title_k]:
            if title in self.title_to_id:
                # Title is in forget set or already assigned an extra ID
                retain_title_ids.append(self.title_to_id[title])
                if title not in unique_forget_titles: # Check if it's an extra title we already processed
                    self.extra_retain_title_ids.add(self.title_to_id[title])
            else:
                 # Title is new (extra retain), assign a new ID
                 new_id = next_extra_id
                 self.title_to_id[title] = new_id
                 self.extra_retain_title_ids.add(new_id)
                 retain_title_ids.append(new_id)
                 next_extra_id += 1


        self.retain_data['title_id'] = retain_title_ids
        print(f"Found {len(self.extra_retain_title_ids)} unique extra retain titles (not in forget set).")
        print(f"Total unique titles across both datasets: {len(self.title_to_id)}")

        # --- Pre-compute Sample Definitions for Batches ---
        print("Pre-computing batch structures and sample definitions...")
        self.sample_definitions: List[Dict[str, Any]] = [] # Final flat list

        # Group indices by title_id
        forget_indices_by_title = self.forget_data.groupby('title_id').groups
        retain_indices_by_title = self.retain_data.groupby('title_id').groups

        # Temporary list to hold the *batches* from paired titles
        paired_batches: List[List[Dict[str, Any]]] = []
        # Temporary list to hold *individual* sample definitions for extra retain samples
        extra_retain_sample_defs: List[Dict[str, Any]] = []


        # 1. Create paired batches for forget titles
        print("Processing paired batches for forget titles...")
        for tid in self.forget_title_ids:
            f_indices = forget_indices_by_title.get(tid)
            # Important: Get retain indices matching THIS forget title ID
            r_indices = retain_indices_by_title.get(tid)

            if f_indices is None or len(f_indices) == 0:
                print(f"Warning: Forget title ID {tid} has no forget samples. Skipping pairing.")
                continue
            if r_indices is None or len(r_indices) == 0:
                print(f"Warning: Forget title ID {tid} has no matching retain samples. Skipping pairing.")
                # Decide if you want to skip entirely or create forget-only batches?
                # Original code skips. Let's stick to that.
                continue

            num_f = len(f_indices)
            num_r = len(r_indices)

            # Determine number of cycles needed to cover all samples of the larger group
            num_f_cycles = math.ceil(num_f / self.n)
            num_r_cycles = math.ceil(num_r / self.retain_per_batch)
            num_total_cycles = max(num_f_cycles, num_r_cycles) # Number of times we need to fill a batch structure

            if num_total_cycles == 0: continue

            print(f"Title ID {tid} (Paired): Num Forget={num_f}, Num Retain={num_r}. Cycling {num_total_cycles} times.")

            f_iter = itertools.cycle(f_indices)
            r_iter = itertools.cycle(r_indices)

            for _ in range(num_total_cycles):
                current_batch_samples = []
                # Add forget samples
                for _ in range(self.n):
                    current_batch_samples.append({
                        'type': 'forget',
                        'index': next(f_iter),
                        'title_id': tid,
                        'factor': -1.0
                    })
                # Add retain samples
                for _ in range(self.retain_per_batch):
                    current_batch_samples.append({
                        'type': 'retain',
                        'index': next(r_iter),
                        'title_id': tid,
                        'factor': 1.0
                    })
                # Ensure the created batch has the expected size
                # This assumes n + retain_per_batch == bs, enforced by checks above
                if len(current_batch_samples) != self.bs:
                     print(f"Warning: Batch for title {tid} has size {len(current_batch_samples)}, expected {self.bs}")
                paired_batches.append(current_batch_samples)

        # ****** START OF MODIFIED SECTION ******
        # 2. Create sample definitions for extra retain titles (assign random forget title ID)
        print(f"Processing {len(self.extra_retain_title_ids)} extra retain titles...")
        processed_count = 0
        for tid in self.extra_retain_title_ids: # Iterate through the IDs assigned to extra retain titles
            # Get all retain samples that were assigned this specific *extra* retain title ID
            r_indices = retain_indices_by_title.get(tid)

            if r_indices is None or len(r_indices) == 0:
                 # This might happen if an extra title ID was generated but somehow no samples map to it
                 print(f"Warning: Extra Retain title ID {tid} has no samples in grouped data? Skipping.")
                 continue

            # Iterate through each individual sample index associated with this extra retain title ID
            for r_idx in r_indices:
                # Choose a random title ID from the *forget* set
                random_forget_tid = random.choice(self.forget_title_id_list)

                # Create the sample definition with the *random forget tid*
                extra_retain_sample_defs.append({
                    'type': 'retain',
                    'index': r_idx,                  # Original index in retain_data
                    'title_id': random_forget_tid, # Assign random forget title ID
                    'factor': 1.0,
                    'original_extra_tid': tid      # Optional: store original extra tid for debugging/analysis
                })
                processed_count += 1
        print(f"Created {processed_count} definitions for individual extra retain samples.")
        # ****** END OF MODIFIED SECTION ******


        # 3. Flatten paired batches and combine with extra retain samples
        print("Flattening paired batches and combining all sample definitions...")
        # Flatten the paired batches first
        self.sample_definitions = [item for batch in paired_batches for item in batch]
        # Add the definitions for the extra retain samples (which are already individual items)
        self.sample_definitions.extend(extra_retain_sample_defs)

        # 4. Shuffle the final combined list
        print("Shuffling all sample definitions...")
        random.shuffle(self.sample_definitions)

        if not self.sample_definitions:
            print("Warning: No samples were generated. Check input data and parameters (n, bs).")

        print(f"Dataset initialized. Total samples to be yielded: {len(self.sample_definitions)}")


    def __len__(self):
        return len(self.sample_definitions)

    def __getitem__(self, idx):
        if idx >= len(self.sample_definitions):
             raise IndexError("Index out of bounds")

        sample_def = self.sample_definitions[idx]
        sample_type = sample_def['type']
        original_index = sample_def['index']
        title_id = sample_def['title_id'] # This is the possibly random ID for extra retain samples
        factor = sample_def['factor']

        source_df = self.forget_data if sample_type == 'forget' else self.retain_data

        try:
            sample_row = source_df.loc[original_index]
        except KeyError:
             # Should be less likely with reset_index, but keep for safety
             raise KeyError(f"Original index {original_index} not found in {sample_type} data.")

        # Process the text using the conversion function.
        input_ids, labels, attention_mask = convert_raw_data_to_model_qa(
            self.tokenizer,
            self.max_length,
            sample_row[self.qk],
            sample_row[self.ak],
            self.template_format
        )

        # Return tuple: input_ids, labels, attention_mask, factor, title_id
        return (input_ids, labels, attention_mask, torch.tensor(factor, dtype=torch.float), torch.tensor(title_id, dtype=torch.long))



class InterleavedRatioDataset(Dataset):
    def __init__(self,
                 forget_data: pd.DataFrame,
                 retain_data: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int,
                 num_forget_in_logical_unit: int,
                 num_retain_in_logical_unit: int,
                 template_format: str = None,
                 question_key: str = 'question',
                 answer_key: str = 'answer'):

        # --- Input Validations ---
        if not isinstance(forget_data, pd.DataFrame):
            raise TypeError("forget_data must be a pandas DataFrame.")
        if not isinstance(retain_data, pd.DataFrame):
            raise TypeError("retain_data must be a pandas DataFrame.")
        if not isinstance(tokenizer, PreTrainedTokenizer):
            raise TypeError("tokenizer must be a PreTrainedTokenizerBase instance.")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if not isinstance(num_forget_in_logical_unit, int) or num_forget_in_logical_unit < 0:
            raise ValueError("num_forget_in_logical_unit must be a non-negative integer.")
        if not isinstance(num_retain_in_logical_unit, int) or num_retain_in_logical_unit < 0:
            raise ValueError("num_retain_in_logical_unit must be a non-negative integer.")
        if num_forget_in_logical_unit == 0 and num_retain_in_logical_unit == 0:
            raise ValueError("Both num_forget_in_logical_unit and num_retain_in_logical_unit cannot be zero.")

        for df_name, df, req_keys in [
            ('forget_data', forget_data, [question_key, answer_key] if num_forget_in_logical_unit > 0 else []),
            ('retain_data', retain_data, [question_key, answer_key] if num_retain_in_logical_unit > 0 else [])
        ]:
            if not df.empty or req_keys: # Only check columns if df is not empty or keys are expected
                 if not all(col in df.columns for col in req_keys):
                    raise ValueError(f"{df_name} must contain columns: {req_keys} if it's to be used.")


        self.forget_df = forget_data.reset_index(drop=True)
        self.retain_df = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.qk = question_key
        self.ak = answer_key

        self.num_total_forget = len(self.forget_df)
        self.num_total_retain = len(self.retain_df)

        self.n_forget_per_unit = 0
        if self.num_total_forget > 0 and num_forget_in_logical_unit > 0:
            self.n_forget_per_unit = num_forget_in_logical_unit
        elif num_forget_in_logical_unit > 0 and self.num_total_forget == 0:
            print(f"Warning: Requested {num_forget_in_logical_unit} forget samples per unit, but forget_data is empty. No forget samples will be yielded.")

        self.n_retain_per_unit = 0
        if self.num_total_retain > 0 and num_retain_in_logical_unit > 0:
            self.n_retain_per_unit = num_retain_in_logical_unit
        elif num_retain_in_logical_unit > 0 and self.num_total_retain == 0:
            print(f"Warning: Requested {num_retain_in_logical_unit} retain samples per unit, but retain_data is empty. No retain samples will be yielded.")

        self.logical_unit_length = self.n_forget_per_unit + self.n_retain_per_unit

        if self.logical_unit_length == 0:
            print("Warning: Effective logical unit length is 0. Dataset will be empty.")
            self._len = 0
            self.num_logical_units_to_iterate = 0
        else:
            num_f_units_needed = 0
            if self.n_forget_per_unit > 0:
                num_f_units_needed = math.ceil(self.num_total_forget / self.n_forget_per_unit)

            num_r_units_needed = 0
            if self.n_retain_per_unit > 0:
                num_r_units_needed = math.ceil(self.num_total_retain / self.n_retain_per_unit)

            self.num_logical_units_to_iterate = max(num_f_units_needed, num_r_units_needed)
            if self.num_logical_units_to_iterate == 0 and (self.num_total_forget > 0 or self.num_total_retain > 0) :
                 # This can happen if one unit count is >0 but the corresponding dataset is empty,
                 # and the other unit count is 0. Max becomes 0.
                 # If only one type of sample is active and has data, iterate through all of them.
                 if self.n_forget_per_unit > 0 and self.num_total_forget > 0 and self.n_retain_per_unit == 0 :
                     self._len = self.num_total_forget
                     self.num_logical_units_to_iterate = num_f_units_needed # re-calc for clarity
                 elif self.n_retain_per_unit > 0 and self.num_total_retain > 0 and self.n_forget_per_unit == 0:
                     self._len = self.num_total_retain
                     self.num_logical_units_to_iterate = num_r_units_needed # re-calc for clarity
                 else: # Both requested but one or both data sources empty, making effective unit non-constructible.
                     self._len = 0
            else:
                 self._len = self.num_logical_units_to_iterate * self.logical_unit_length


        print(f"--- InterleavedRatioDataset Initialized ---")
        print(f"Requested: {num_forget_in_logical_unit} Forget, {num_retain_in_logical_unit} Retain per logical unit.")
        print(f"Available: {self.num_total_forget} Forget, {self.num_total_retain} Retain samples.")
        print(f"Effective: {self.n_forget_per_unit} Forget, {self.n_retain_per_unit} Retain per logical unit.")
        print(f"Logical Unit Length: {self.logical_unit_length}")
        print(f"Num Logical Units to Iterate (to cover all data): {self.num_logical_units_to_iterate}")
        print(f"Total Dataset Length (__len__): {self._len}")
        print(f"-------------------------------------------")


    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self._len == 0:
            raise IndexError("Dataset is empty or improperly configured.")
        if idx >= self._len:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._len}")

        # Determine which logical unit this index falls into (for cycling through source data)
        logical_unit_cycle_idx = idx // self.logical_unit_length
        # Position within the current F,F,...,R,R,... pattern of the logical unit
        pos_in_logical_unit = idx % self.logical_unit_length

        sample_row = None
        factor_val = 0.0

        if pos_in_logical_unit < self.n_forget_per_unit:
            # This slot is for a forget sample
            if self.num_total_forget == 0: # Should be caught by _len == 0 if n_forget_per_unit > 0
                raise RuntimeError("Attempting to fetch a forget sample, but no forget data is available (this should not happen if configured correctly).")
            # Index into the original forget_df, cycling if necessary
            effective_forget_idx = (logical_unit_cycle_idx * self.n_forget_per_unit + pos_in_logical_unit) % self.num_total_forget
            sample_row = self.forget_df.iloc[effective_forget_idx]
            factor_val = -1.0
        else:
            # This slot is for a retain sample
            if self.num_total_retain == 0: # Should be caught
                raise RuntimeError("Attempting to fetch a retain sample, but no retain data is available (this should not happen if configured correctly).")
            # Position within the retain part of the logical unit
            pos_in_retain_part = pos_in_logical_unit - self.n_forget_per_unit
            effective_retain_idx = (logical_unit_cycle_idx * self.n_retain_per_unit + pos_in_retain_part) % self.num_total_retain
            sample_row = self.retain_df.iloc[effective_retain_idx]
            factor_val = 1.0

        input_ids, labels, attention_mask = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length, sample_row[self.qk], sample_row[self.ak], self.template_format
        )
        return (input_ids, labels, attention_mask, torch.tensor(factor_val, dtype=torch.float))




