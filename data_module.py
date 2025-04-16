from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import PreTrainedTokenizer
from typing import Tuple
import math
import pandas as pd
from typing import Optional


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
            fraction = -1.0
        else:
            # Retain sample
            if self.num_retain == 0:
                 raise RuntimeError("Attempting to fetch retain sample but retain dataset is empty.") # Should not happen
            # Calculate the overall retain index based on blocks and position within retain part
            pos_in_retain_part = pos_in_effective_block - self.n_forget_per_block
            effective_idx = (block_index * self.n_retain_per_block + pos_in_retain_part) % self.num_retain
            sample_row = self.retain.iloc[effective_idx]
            fraction = 1.0

        # Process the text using the conversion function
        processed_sample = convert_raw_data_to_model_qa( # Assuming this exists and works
            self.tokenizer,
            self.max_length,
            sample_row[self.qk],
            sample_row[self.ak],
            self.template_format
        )

        input_ids, labels, attention_mask = processed_sample
        return (input_ids, labels, attention_mask, fraction)
    



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
            fraction = -1.0 # Use float
        else:
            # Retain sample
            if self.num_retain == 0 or self.retain_per_block <= 0:
                 # Should not happen if len is correct and n < bs
                 raise RuntimeError("Trying to get retain sample, but retain dataset is empty or n>=bs.")
            # Correct index for retain part
            effective_idx = (block_index * self.retain_per_block + (pos_in_block - self.n)) % self.num_retain
            sample_row = self.retain.iloc[effective_idx]
            fraction = 1.0 # Use float

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
        return (input_ids, labels, attention_mask, fraction)


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




