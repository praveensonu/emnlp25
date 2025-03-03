import torch

def constrained_pack_sequences(batch_data, max_seq_len=2046, pad_token_id=0):
    """
    Packs sequences with constraint that no question-answer pair is split.
    
    Args:
        batch_data: List of dictionaries with input_ids, labels, etc.
        max_seq_len: Maximum sequence length (default 2046)
        pad_token_id: Token ID to use for padding
        
    Returns:
        List of packed sequence dictionaries
    """
    packed_batches = []
    current_batch = {
        'input_ids': [],
        'labels': [],
        'attention_mask': []
    }
    current_length = 0
    
    for item in batch_data:
        # If this item would exceed max length, finalize current batch and start new one
        if current_length + item['length'] > max_seq_len and current_length > 0:
            # Convert lists to tensors
            for key in ['input_ids', 'labels', 'attention_mask']:
                current_batch[key] = torch.tensor(current_batch[key])
            packed_batches.append(current_batch)
            
            # Start a new batch
            current_batch = {
                'input_ids': [],
                'labels': [],
                'attention_mask': []
            }
            current_length = 0
        
        # If single item exceeds max length, pad it individually
        if item['length'] > max_seq_len:
            truncated_input_ids = item['input_ids'][:max_seq_len].tolist()
            truncated_labels = item['labels'][:max_seq_len].tolist()
            truncated_attention_mask = item['attention_mask'][:max_seq_len].tolist()
            
            packed_batches.append({
                'input_ids': torch.tensor(truncated_input_ids),
                'labels': torch.tensor(truncated_labels),
                'attention_mask': torch.tensor(truncated_attention_mask)
            })
        else:
            # Add this item to current batch
            current_batch['input_ids'].extend(item['input_ids'].tolist())
            current_batch['labels'].extend(item['labels'].tolist())
            current_batch['attention_mask'].extend(item['attention_mask'].tolist())
            current_length += item['length']
    
    # Don't forget the last batch if it has data
    if current_length > 0:
        for key in ['input_ids', 'labels', 'attention_mask']:
            current_batch[key] = torch.tensor(current_batch[key])
        packed_batches.append(current_batch)
    
    return packed_batches


def collate_with_constrained_packing(batch, max_seq_len=2046, pad_token_id=0):
    """
    Custom collate function that packs sequences with constraints
    
    Args:
        batch: List of samples from dataset
        max_seq_len: Maximum sequence length (default 2046)
        pad_token_id: Token ID to use for padding
        
    Returns:
        List of dictionaries with packed tensors
    """
    return constrained_pack_sequences(batch, max_seq_len, pad_token_id)


def dual_collate_with_constrained_packing(batch, max_seq_len=2046, pad_token_id=0):
    """
    Custom collate function for DualDataset with constrained packing
    
    Args:
        batch: List of (forget_data, retain_data) tuples
        max_seq_len: Maximum sequence length
        pad_token_id: Token ID to use for padding
        
    Returns:
        Tuple of (forget_batches, retain_batches)
    """
    forget_items = [item[0] for item in batch]
    retain_items = [item[1] for item in batch]
    
    forget_batches = constrained_pack_sequences(forget_items, max_seq_len, pad_token_id)
    retain_batches = constrained_pack_sequences(retain_items, max_seq_len, pad_token_id)
    
    return forget_batches, retain_batches