from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import PreTrainedTokenizer
from typing import Tuple
import math
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
import itertools
import random


def convert_raw_data_to_model_qa(tokenizer: PreTrainedTokenizer, 
                                    max_length: int, 
                                    question: str, 
                                    answer: str,
                                    template_format=None) -> torch.Tensor:
    """
    Tokenizes question answer pair and returns input_ids, labels, and attention_mask into SFT format.
    
    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize the input.
        max_length (int): Maximum sequence length. This includes max_new_tokens + token length of question.
        question (str): Question to be tokenized.
        answer (str): Answer to be tokenized.
        template_format (str, optional): Custom template format. If None, will use the tokenizer's chat template.
    
    Returns:
        torch.Tensor: Each input_ids, labels, and attention_mask in their own tensor.
    """
    # Format the question using either custom template or chat template
    if template_format:
        new_question = template_format.format(instruction=question)
    else:
        # Use the tokenizer's built-in chat template
        messages = [{"role": "user", "content": question}]
        new_question = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    full_text = new_question + answer
    
    # Get the number of tokens in the question part
    prompt_inputs = tokenizer(new_question, return_tensors="pt")
    num_question_tokens = prompt_inputs["input_ids"].size(1)
    
    # Tokenize the full text
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    
    # Padding logic
    pad_length = max_length - len(encoded["input_ids"])
    
    # Use the tokenizer's pad token instead of hardcoded values if available
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Create padded input_ids
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] + [pad_token_id] * (pad_length - 1) if pad_length > 0 else encoded['input_ids']
    
    # Create padded attention mask
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length if pad_length > 0 else encoded['attention_mask']
    
    # Create labels, masking the prompt tokens
    if len(encoded['input_ids']) == max_length:
        label = encoded['input_ids'].copy()
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
    
    # Mask prompt tokens in labels
    for i in range(num_question_tokens):
        label[i] = -100
    
    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


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
        Dataset that batches forget and retain samples based on a shared 'title'.

        Args:
            forget_data (pd.DataFrame): DataFrame with forget samples. Must contain title_key, question_key, answer_key.
            retain_data (pd.DataFrame): DataFrame with retain samples. Must contain title_key, question_key, answer_key.
            tokenizer (PreTrainedTokenizer): Tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            n (int): Target number of forget samples per paired batch.
            bs (int): Total batch size (block size).
            template_format (str, optional): Custom template string. Defaults to None (uses tokenizer chat template).
            title_key (str): Column name for the title/topic. Defaults to 'title'.
            question_key (str): Column name for the question. Defaults to 'question'.
            answer_key (str): Column name for the answer. Defaults to 'answer'.
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

        # 2. Assign IDs to retain titles (use existing if match forget, else N+1 onwards)
        self.extra_retain_title_ids: Set[int] = set()
        next_extra_id = len(self.forget_title_ids) + 1
        retain_title_ids = []
        for title in self.retain_data[self.title_k]:
            if title in self.title_to_id:
                retain_title_ids.append(self.title_to_id[title])
            else:
                # Assign a new ID if not seen before in *this* loop
                if title not in self.title_to_id:
                    self.title_to_id[title] = next_extra_id
                    self.extra_retain_title_ids.add(next_extra_id)
                    next_extra_id += 1
                retain_title_ids.append(self.title_to_id[title]) # Use the newly assigned ID

        self.retain_data['title_id'] = retain_title_ids
        print(f"Found {len(self.extra_retain_title_ids)} unique extra retain titles (not in forget set).")
        print(f"Total unique titles across both datasets: {len(self.title_to_id)}")

        # --- Pre-compute Sample Definitions for Batches ---
        print("Pre-computing batch structures...")
        self.sample_definitions: List[Dict[str, Any]] = []

        # Group indices by title_id
        forget_indices_by_title = self.forget_data.groupby('title_id').groups
        retain_indices_by_title = self.retain_data.groupby('title_id').groups

        all_batches: List[List[Dict[str, Any]]] = []

        # 1. Create paired batches for forget titles
        for tid in self.forget_title_ids:
            f_indices = forget_indices_by_title.get(tid)
            r_indices = retain_indices_by_title.get(tid)

            if f_indices is None or len(f_indices) == 0:
                print(f"Warning: Forget title ID {tid} has no forget samples. Skipping pairing.")
                continue
            if r_indices is None or len(r_indices) == 0:
                print(f"Warning: Forget title ID {tid} has no matching retain samples. Skipping pairing.")
                continue

            num_f = len(f_indices)
            num_r = len(r_indices)

            # Determine number of batches needed to cover all samples of the larger group for this title
            num_forget_yields = math.ceil(num_f / self.n) * self.n
            num_retain_yields = math.ceil(num_r / self.retain_per_batch) * self.retain_per_batch
            total_yields = max(num_forget_yields, num_retain_yields)
            num_batches_for_title = math.ceil(total_yields / self.bs)

            if num_batches_for_title == 0: continue

            print(f"Title ID {tid} (Forget): Num Forget={num_f}, Num Retain={num_r}. Creating {num_batches_for_title} paired batches.")

            f_iter = itertools.cycle(f_indices)
            r_iter = itertools.cycle(r_indices)

            for _ in range(num_batches_for_title):
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
                all_batches.append(current_batch_samples)

        # 2. Create retain-only batches for extra retain titles
        for tid in self.extra_retain_title_ids:
            r_indices = retain_indices_by_title.get(tid)

            if r_indices is None or len(r_indices) == 0:
                 print(f"Warning: Extra Retain title ID {tid} has no samples?") # Should not happen based on logic above
                 continue

            num_r = len(r_indices)
            num_batches_for_title = math.ceil(num_r / self.bs)

            if num_batches_for_title == 0: continue

            print(f"Title ID {tid} (Extra Retain): Num Retain={num_r}. Creating {num_batches_for_title} retain-only batches.")

            r_iter = itertools.cycle(r_indices)

            for _ in range(num_batches_for_title):
                current_batch_samples = []
                # Add retain samples
                samples_to_add = min(self.bs, num_r) # Handle cases where num_r < bs
                num_r -= samples_to_add # Track remaining samples for accurate count if needed, although cycle handles it
                for _ in range(samples_to_add): # Add up to bs samples
                     current_batch_samples.append({
                        'type': 'retain',
                        'index': next(r_iter),
                        'title_id': tid,
                        'factor': 1.0
                    })
                if current_batch_samples: # Only add if we actually got samples
                     all_batches.append(current_batch_samples)


        # 3. Shuffle the batches and flatten
        random.shuffle(all_batches)
        self.sample_definitions = [item for batch in all_batches for item in batch]

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
        title_id = sample_def['title_id']
        factor = sample_def['factor']

        source_df = self.forget_data if sample_type == 'forget' else self.retain_data

        try:
            sample_row = source_df.loc[original_index]
        except KeyError:
             print(f"Error retrieving original_index {original_index} from {sample_type} data. Available indices: {source_df.index}")
             # Handle error appropriately, maybe raise or return a dummy item?
             # For now, re-raising might be best to catch issues early.
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

def custom_data_collator_forget(samples):
    """
    Collate function for the forget dataset only

    Args:
        samples (list of tuples): Each tuple contains (input_ids, labels, attention_mask)

    Returns:
        dict: batched_inputs, labels, attention_masks.

    """
    input_ids = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    attention_mask = torch.stack([sample[2] for sample in samples])
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}




def custom_data_collator_interleaved_ga(samples):
    """
    Collate function for the forget dataset samples.

    Each sample is expected to be a tuple:
        (input_ids, labels, attention_mask, fraction)
    
    Returns:
        dict: A dictionary with the following keys:
            - 'input_ids': Batched input_ids tensor.
            - 'labels': Batched labels tensor.
            - 'attention_mask': Batched attention_mask tensor.
            - 'factor': Batched tensor of factors (-1 or +1) for each sample.
    """
    input_ids = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    attention_mask = torch.stack([sample[2] for sample in samples])
    factors = torch.tensor([sample[3] for sample in samples], dtype=torch.float)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'factor': factors
    }


def custom_data_collator_paired_title(samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function specifically for PairedTitleDataset samples.

    Each sample is expected to be a tuple:
        (input_ids_tensor, labels_tensor, attention_mask_tensor, factor_tensor, title_id_tensor)

    Returns:
        dict: A dictionary with the following keys, where values are batched tensors:
            - 'input_ids': Batched input_ids tensor.
            - 'labels': Batched labels tensor.
            - 'attention_mask': Batched attention_mask tensor.
            - 'factor': Batched tensor of factors (-1.0 or +1.0) for each sample.
            - 'title_id': Batched tensor of title IDs for each sample.
    """
    if not samples:
        return {} # Handle empty batch case

    # Ensure all samples have the expected number of elements
    expected_elements = 5
    if any(len(sample) != expected_elements for sample in samples):
        raise ValueError(f"All samples must be tuples of length {expected_elements}")

    # Stack the tensors from each sample
    input_ids = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    attention_mask = torch.stack([sample[2] for sample in samples])
    # Factors are already tensors (sample[3]), just stack them
    factors = torch.stack([sample[3] for sample in samples])
    # Title IDs are already tensors (sample[4]), just stack them
    title_ids = torch.stack([sample[4] for sample in samples])

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'factor': factors,
        'title_id': title_ids  # Add the title_id field
    }




def custom_gd_collator_forget(samples):
    """
    Custom data collator for forget and retain data

    Args:
        samples: list of tuples (forget_data, retain_data) from the DualDataset class

    Returns:
        rets: list of tuples (input_ids, labels, attention_mask)
        example output for batch size 2
        
        [(  #forget data for batch of 2
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # input_ids
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # labels
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # attention_mask
            ),
            (  #retain data for batch of 2
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # input_ids
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # labels
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), # attention_mask
            ),
        ]

    """

    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets




