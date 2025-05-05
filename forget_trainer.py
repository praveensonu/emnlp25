from transformers import Trainer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from transformers.trainer_utils import is_main_process
from typing import Dict, Union, Any, Optional, List, Tuple, Type



accelerator = Accelerator()
def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss



class GATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        """
        Computes the gradient ascent loss for the model
        """
        #if self.loss_type == 'grad_ascent':
        # unpack the forget inputs
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        attention_mask = inputs['attention_mask']

        # forward pass
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        forget_loss = outputs.loss * -1 # gradient ascent is negating the loss

        loss = forget_loss
        return (loss, outputs) if return_outputs else loss
    
    

class GradDiffTrainer(Trainer):
    ## since outputs.loss gives mean loss over a batch, we need to compute loss over a sequence and avg it.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1) extract the +1/-1 factors
        factor = inputs.pop("factor")   # [B]
        
        # 2) forward pass (will return mean loss over batch, but we'll ignore it)
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits                               # [B, L, Vocab]
        labels = inputs["labels"]                             # [B, L]

        # 3) shift for causal LM (predict token i from all tokens < i)
        shift_logits = logits[..., :-1, :].contiguous()       # [B, L-1, V]
        shift_labels = labels[..., 1:].contiguous()           # [B, L-1]

        # 4) token-level loss, no reduction
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),    # [B*(L-1), V]
            shift_labels.view(-1)                             # [B*(L-1)]
        )                                                     # [B*(L-1)]
        
        # 5) back to [B, L-1], average per example
        token_loss = token_loss.view(shift_labels.size())    # [B, L-1]
        example_loss = token_loss.mean(dim=1)                # [B]

        # 6) weight by factor and average
        weighted_loss = (example_loss * factor).mean()       # scalar

        return (weighted_loss, outputs) if return_outputs else weighted_loss


def process_inputs(inputs):
    """
    Modifies the inputs dictionary by removing 'factor' and 'title_id'
    if they exist. Returns the potentially popped values.
    """
    # Attempt to pop 'factor'. If it doesn't exist, factor_value will be None.
    factor_value = inputs.pop("factor", None)

    # Attempt to pop 'title_id'. If it doesn't exist, title_id_value will be None.
    title_id_value = inputs.pop("title_id", None)

    print(f"Attempted to pop 'factor'. Value obtained: {factor_value}")
    print(f"Attempted to pop 'title_id'. Value obtained: {title_id_value}")
    print(f"Remaining keys in inputs: {list(inputs.keys())}")

    # You can now use factor_value and title_id_value if they are not None
    # The 'inputs' dictionary has been modified in place.

    return factor_value, title_id_value

class BatchGradDiffTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the adjusted loss for gradient difference unlearning,
        scaling each sequence's loss by the provided factor (-1 or +1).
        Handles the presence of 'title_id' in inputs.
        """
        # Extract the factors and title_ids (per-sample) and remove them from inputs.
        
        factors, title_ids = process_inputs(inputs)  # Expected shape: (batch_size,) - Not used in loss calc but needs removal

        # Forward pass: model should return logits and any additional outputs.
        outputs = model(**inputs)
        logits = outputs.logits

        # Assume standard causal LM setup:
        #   - logits: (batch_size, seq_len, vocab_size)
        #   - labels: (batch_size, seq_len)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous() # Use labels from inputs dict

        # Define loss function with no reduction to keep per-token loss.
        loss_fct = CrossEntropyLoss(reduction="none")

        # Flatten the logits and labels to compute loss for each token.
        # Need to handle potential empty tensors if seq_len <= 1 after shift
        if shift_logits.size(1) == 0:
            # If sequence length is too short, loss is 0 or handle as error
            # For now, return 0 loss for this case.
            adjusted_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            print("Warning: Sequence length <= 1 after shifting, resulting in 0 loss for this batch.")
            return (adjusted_loss, outputs) if return_outputs else adjusted_loss

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Reshape loss to (batch_size, seq_len - 1)
        loss = loss.view(shift_logits.size(0), -1)

        # Compute the count of valid tokens for each sequence (ignoring tokens with label -100)
        # Ensure valid_counts is float for division and handle division by zero
        valid_counts = (shift_labels != -100).sum(dim=-1).float()
        valid_counts = torch.max(valid_counts, torch.tensor(1.0, device=valid_counts.device)) # Avoid division by zero

        # Calculate the average loss for each sequence independently.
        per_sequence_loss = loss.sum(dim=-1) / valid_counts  # Shape: (batch_size,)

        # Scale each sequence's loss by its corresponding factor (-1 or +1).
        scaled_losses = per_sequence_loss * factors

        # Average the scaled per-sequence losses to get a single scalar value.
        adjusted_loss = scaled_losses.mean()

        return (adjusted_loss, outputs) if return_outputs else adjusted_loss
    

