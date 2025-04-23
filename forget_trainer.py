from transformers import Trainer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from transformers.trainer_utils import is_main_process


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
    

## not using this trainer
class VanillaGradDiffTrainer(Trainer): 
    
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        forget_inputs, retain_inputs = inputs
        input_ids, labels, attention_mask = forget_inputs

        ## gradient ascent on the forget
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        forget_loss = outputs.loss
        forget_loss = forget_loss * -1

        ## gradient descent on the retain
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
        retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
        retain_loss = retain_outputs.loss
        loss = forget_loss + retain_loss

        return (loss, outputs) if return_outputs else loss
    

class GradDiffTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the adjusted loss for gradient difference unlearning,
        scaling each sequence's loss by the provided factor (-1 or +1).

        Steps:
          - Pops the "factor" field from inputs.
          - Performs a forward pass of the model.
          - Calculates per-token cross-entropy loss with reduction set to "none".
          - Reshapes and averages the token losses to compute per-sequence loss.
          - Scales each sequence's loss by its factor.
          - Returns the mean of the scaled per-sequence losses.
        """
        # Extract the factors (per-sample) and remove it from inputs.
        factors = inputs.pop("factor")  # Expected shape: (batch_size,)
        
        # Forward pass: model should return logits and any additional outputs.
        outputs = model(**inputs)
        logits = outputs.logits

        # For language modeling tasks, typically you shift the logits and labels by one.
        # Adjust these operations based on how your inputs and model are set up.
        # Here we assume:
        #   - logits: (batch_size, seq_len, vocab_size)
        #   - labels: (batch_size, seq_len)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        # Define loss function with no reduction to keep per-token loss.
        loss_fct = CrossEntropyLoss(reduction="none")
        # Flatten the logits and labels to compute loss for each token.
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Reshape to (batch_size, seq_len - 1)
        loss = loss.view(shift_logits.size(0), -1)
        
        # Compute the count of valid tokens for each sequence (ignoring tokens with label -100)
        valid_counts = (shift_labels != -100).sum(dim=-1).float()
        
        # Calculate the average loss for each sequence independently.
        per_sequence_loss = loss.sum(dim=-1) / valid_counts  # Shape: (batch_size,)
        
        # Scale each sequence's loss by its corresponding factor (-1 or +1).
        scaled_losses = per_sequence_loss * factors
        
        # Average the scaled per-sequence losses to get a single scalar value.
        adjusted_loss = scaled_losses.mean()
        
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss


class BatchGradDiffTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the adjusted loss for gradient difference unlearning,
        scaling each sequence's loss by the provided factor (-1 or +1).
        Handles the presence of 'title_id' in inputs.
        """
        # Extract the factors and title_ids (per-sample) and remove them from inputs.
        factors = inputs.pop("factor")      # Expected shape: (batch_size,)
        title_ids = inputs.pop("title_id")  # Expected shape: (batch_size,) - Not used in loss calc but needs removal

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
    


class NPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta=0.01, **kwargs):
        if ref_model is None:
            raise ValueError("Reference model must be provided.")
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        ref_device = next(self.ref_model.parameters()).device
        current_process_device = f"cuda:{self.args.local_rank}"

        if str(ref_device) != current_process_device:
            print(f"WARNING (Process {self.args.local_rank}): ref_model device ({ref_device}) "
                   f"might not match expected DDP device ({current_process_device}). Check device_map.")
            try:
                self.ref_model.to(current_process_device)
                ref_device = next(self.ref_model.parameters()).device
                print(f"moved ref_model to {ref_device}")
            except Exception as e:
                print(f'ERROR moving ref_model: {e}')

        self.ref_model.eval()  # Set reference model to evaluation mode
        for param in self.ref_model.parameters():
            param.requires_grad = False # freezing the ref_model params explicitly 


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
             
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        attention_mask = inputs['attention_mask']

        outputs = model(input_ids, labels=labels,
                        attention_mask=attention_mask)
        loss_current = get_batch_loss(outputs.logits, labels)

        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, labels=labels,
                                    attention_mask=attention_mask)
            loss_ref = get_batch_loss(ref_outputs.logits, labels)

        log_probs_policy = -loss_current
        log_probs_ref = -loss_ref
        logits = log_probs_policy - log_probs_ref

        if is_main_process(self.args.local_rank):
         # Print stats for the first element in the batch for inspection
            print(f"\n--- compute_loss Debug (Rank {self.args.local_rank}) ---")
            print(f"  loss_current[0]: {loss_current[0].item()}")
            print(f"  loss_ref[0]: {loss_ref[0].item()}")
            print(f"  log_probs_policy[0]: {log_probs_policy[0].item()}")
            print(f"  log_probs_ref[0]: {log_probs_ref[0].item()}")
            print(f"  logits[0]: {logits[0].item()}")
            # Check for NaNs/Infs *before* logsigmoid
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("  WARNING: logits contains NaN or Inf!")
            # Calculate the term inside logsigmoid
            inner_term = -self.beta * logits
            print(f"  inner_term[0] (-beta * logits): {inner_term[0].item()}")
            if torch.isnan(inner_term).any() or torch.isinf(inner_term).any():
                print("  WARNING: inner_term contains NaN or Inf!")
        
        # Check for stability BEFORE the main calculation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"ERROR (Rank {self.args.local_rank}): NaN/Inf detected in logits. Returning dummy loss 0.")
            dummy_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (dummy_loss, outputs) if return_outputs else dummy_loss
        
        npo_loss_terms = -F.logsigmoid(inner_term)
        if torch.isnan(npo_loss_terms).any() or torch.isinf(npo_loss_terms).any():
            print(f"ERROR (Rank {self.args.local_rank}): NaN/Inf detected in npo_loss_terms. Returning dummy loss 0.")
            dummy_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (dummy_loss, outputs) if return_outputs else dummy_loss

        # Final check
        if torch.isnan(npo_loss_terms).any() or torch.isinf(npo_loss_terms).any():
            print(f"ERROR (Rank {self.args.local_rank}): NaN/Inf detected in npo_loss_terms. Returning dummy loss 0.")
            dummy_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (dummy_loss, outputs) if return_outputs else dummy_loss

        npo_loss = npo_loss_terms.mean()

        if is_main_process(self.args.local_rank):
            print(f"  npo_loss (mean): {npo_loss.item()}")
            print(f"--- End compute_loss Debug ---\n")

        if torch.isnan(npo_loss) or torch.isinf(npo_loss):
            print(f"ERROR (Rank {self.args.local_rank}): Final npo_loss is NaN/Inf. Returning dummy loss 0.")
            dummy_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (dummy_loss, outputs) if return_outputs else dummy_loss

        loss = npo_loss
        return (loss, outputs) if return_outputs else loss
    
    
class NPO_GradDiffTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = self.npo_coeff * forget_loss + self.grad_diff_coeff * retain_loss

            return loss