import torch
from transformers import  default_data_collator
from typing import Tuple, Any
import pandas as pd
from typing import Dict, List, Tuple 



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


def custom_data_collator_interleaved(samples):
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




# def dpo_retain_collator(samples: list[dict]) -> dict[str, torch.Tensor]:
#     """
#     Collates samples from CombinedForgetRetainDataset.
#     Each sample is a dict:
#     {
#         'answer_input_ids': Tensor, 'answer_labels': Tensor, 'answer_attention_mask': Tensor,
#         'idk_input_ids': Tensor, 'idk_labels': Tensor, 'idk_attention_mask': Tensor,
#         'factor': float
#     }
#     Returns a batch dict with stacked tensors.
#     """
#     if not samples:
#         return {}

#     keys = samples[0].keys()
#     batch = {}

#     for key in keys:
#         if key == 'factor':
#             batch[key] = torch.tensor([sample[key] for sample in samples], dtype=torch.float)
#         elif isinstance(samples[0][key], torch.Tensor):
#             batch[key] = torch.stack([sample[key] for sample in samples])
#         else:
#             batch[key] = [sample[key] for sample in samples] 
            
#     return batch


def dpo_retain_collator(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collates samples from CombinedForgetRetainDataset.
    Each sample is a dict, potentially including:
    {
        'answer_input_ids': Tensor, 'answer_labels': Tensor, 'answer_attention_mask': Tensor,
        'idk_input_ids': Tensor, 'idk_labels': Tensor, 'idk_attention_mask': Tensor,
        'factor': float,
        'original_index': Tensor (scalar, long)
    }
    Returns a batch dict with stacked tensors. 'factor' is converted to a float tensor.
    """
    if not samples:
        return {}

    # Initialize the batch dictionary
    batch = {}
    # Get all keys from the first sample to ensure consistency
    # (though all samples from your dataset should have the same keys)
    first_sample_keys = samples[0].keys()

    for key in first_sample_keys:
        # Get all values for the current key from all samples
        values = [sample[key] for sample in samples]

        if key == 'factor':
            # Special handling for 'factor': convert list of floats to a tensor
            batch[key] = torch.tensor(values, dtype=torch.float)
        elif isinstance(values[0], torch.Tensor):
            # If the value is a tensor (e.g., input_ids, labels, attention_mask, original_index), stack them
            batch[key] = torch.stack(values)
        elif isinstance(values[0], (int, float, bool, str)):
            # If it's a basic Python type, you might want to convert to a tensor
            # or handle as per your model's needs. For 'original_index' if it were not a tensor,
            # you'd convert it here. But since we made it a tensor in __getitem__, this case
            # might not be hit for 'original_index'.
            # Let's assume for now other non-tensor items are not expected or are handled by HF Trainer later.
            # If you had other scalar metadata you wanted as tensors:
            # if all(isinstance(v, (int, float)) for v in values):
            #     batch[key] = torch.tensor(values)
            # else:
            #     batch[key] = values # Keep as list if mixed or string
            batch[key] = values # Default to keeping as a list if not tensor and not factor
        else:
            # Fallback for other types (e.g., list of strings if not handled above)
            batch[key] = values
            
    return batch