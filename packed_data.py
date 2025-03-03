import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from packed_collate import collate_with_constrained_packing, dual_collate_with_constrained_packing

def convert_raw_data_to_model_qa(tokenizer, question, answer, template_format=None):
    """
    Prepares input and labels for the model based on the specified format.
    
    Args:
        tokenizer: The tokenizer to use for encoding
        question: The question text
        answer: The answer text
        template_format: Optional, format to structure the input
        
    Returns:
        Tuple of (input_ids, labels, attention_mask) as tensors with their actual length
    """
    # Use the provided template or default to a simple format
    if template_format:
        new_question = template_format.format(instruction=question)
    else:
        new_question = f"Question: {question}\nAnswer:"
    
    full_text = new_question + answer 
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=False,  # We'll handle truncation separately
    )

    input_ids = encoded['input_ids'] + [tokenizer.eos_token_id]
    attention_mask = encoded['attention_mask'] 
    
    # Create labels, masking out question tokens
    labels = input_ids.copy()
    for i in range(num_question_tokens):
        labels[i] = -100

    return {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels),
        'attention_mask': torch.tensor(attention_mask),
        'length': len(input_ids)
    }



class SingleDatasetPacked(Dataset):
    def __init__(self, data_path, tokenizer, template_format=None):
        """
        Initializes the dataset for gradient ascent finetuning with constrained packing
        
        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            template_format (str, optional): format template for structuring input
        """
        super(SingleDatasetPacked, self).__init__()
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.template_format = template_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]['question']
        answer = self.data.iloc[idx]['answer']
        return convert_raw_data_to_model_qa(
            tokenizer=self.tokenizer, 
            question=question, 
            answer=answer,
            template_format=self.template_format
        )

    def get_dataloader(self, batch_size=16, shuffle=True, num_workers=0, max_seq_len=2046):
        """
        Creates a DataLoader with the constrained packing collate function
        """
        return DataLoader(
            self,
            batch_size=batch_size,  # This is just initial grouping before packing
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda batch: collate_with_constrained_packing(
                batch, max_seq_len=max_seq_len, pad_token_id=self.tokenizer.pad_token_id
            )
        )


class DualDatasetPacked(Dataset): 
    """
    Dataset class for creating data for forget and retain (used by gradient difference)
    with constrained sequence packing
    
    Args:
        forget_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for forgetting
        retain_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for retaining
        tokenizer: tokenizer instance to process text
        template_format (str, optional): format template for structuring input
    """
    def __init__(self, forget_data, retain_data, tokenizer, template_format=None):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.template_format = template_format

    def __len__(self):
        return max(len(self.forget), len(self.retain))
    
    def __getitem__(self, idx):
        # Cyclic rotation of data
        forget_idx = idx % len(self.forget)
        retain_idx = idx % len(self.retain)

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer,
            self.forget.iloc[forget_idx]['question'],
            self.forget.iloc[forget_idx]['answer'],
            self.template_format
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer,
            self.retain.iloc[retain_idx]['question'],
            self.retain.iloc[retain_idx]['answer'],
            self.template_format
        )

        return (forget_data, retain_data)

    def get_dataloader(self, batch_size=16, shuffle=True, num_workers=0, max_seq_len=2046):
        """
        Creates a DataLoader with the constrained dual packing collate function
        """
        return DataLoader(
            self,
            batch_size=batch_size,  # This is just initial grouping before packing
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda batch: dual_collate_with_constrained_packing(
                batch, max_seq_len=max_seq_len, pad_token_id=self.tokenizer.pad_token_id
            )
        )