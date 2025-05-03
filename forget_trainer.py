from transformers import Trainer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from transformers.trainer_utils import is_main_process
from typing import Dict, Union, Any, Optional, List, Tuple, Type
from transformers.modeling_outputs import CausalLMOutputWithPast
import copy
import math


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
    

    

class NPOTrainer_og(Trainer):
    def compute_loss(model, ref_model, inputs, beta=0.1):
        forget_inputs = inputs[0]
        input_ids, labels, attention_mask = forget_inputs

        outputs = model(input_ids, labels=labels,
                        attention_mask=attention_mask)
        loss_current = get_batch_loss(outputs.logits, labels)

        with torch.no_grad():
            ref_outputs = ref_model(input_ids, labels=labels,
                                    attention_mask=attention_mask)
            loss_ref = get_batch_loss(ref_outputs.logits, labels)

        neg_log_ratios = loss_current - loss_ref
        loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

        return loss


class InterleavedNPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[nn.Module, Type[nn.Module]],
        ref_model: Optional[Union[nn.Module, Type[nn.Module]]] = None,
        beta: float = 1.0,
        alpha: float = 1.0, # Weight for retain loss
        gamma: float = 1.0, # Weight for forget loss (NPO component)
        retain_loss_type: str = "NLL", # "NLL" or "KL"
        *args,
        **kwargs,
    ):
        """
        Trainer for Negative Preference Optimization (NPO).

        Args:
            model: The model to train.
            ref_model: The reference model (frozen). Required for NPO and KL retain loss.
            beta: Temperature parameter for the NPO loss component.
            alpha: Weight factor for the retain loss component.
            gamma: Weight factor for the forget (NPO) loss component.
            retain_loss_type: Type of loss for the retain set ("NLL" or "KL").
            *args, **kwargs: Standard Hugging Face Trainer arguments.
        """
        if ref_model is None:
            raise ValueError("NPOTrainer requires a reference model (`ref_model`).")

        # Check if ref_model is a class or instance
        if isinstance(ref_model, type):
             raise ValueError("ref_model should be an initialized model instance, not a class.")

        super().__init__(model=model, *args, **kwargs)

        self.ref_model = ref_model
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.retain_loss_type = retain_loss_type.upper()

        if self.retain_loss_type not in ["NLL", "KL"]:
             raise ValueError(f"Unknown retain_loss_type: {retain_loss_type}. Choose 'NLL' or 'KL'.")

        # Ensure ref_model is not trainable and on the correct device
        # (Trainer __init__ handles the main model device placement via args.device/local_rank)
        # User should handle initial ref_model placement as in their example code.
        # We double-check requires_grad here.
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        self.ref_model.eval() # Ensure it's in eval mode

        # Verify ref_model device matches model device if possible (might not be fully prepared yet)
        # This check is best effort during init. The critical check is at compute_loss time.
        try:
            model_device = next(self.model.parameters()).device
            ref_device = next(self.ref_model.parameters()).device
            if model_device != ref_device:
                 print(f"Warning: NPOTrainer init: model device ({model_device}) and ref_model device ({ref_device}) differ. Ensure ref_model is correctly placed before training loop.")
        except StopIteration:
             print("Warning: Could not verify model/ref_model device during NPOTrainer init.")


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=False):
        """
        Computes the combined NPO loss and retain loss for an interleaved batch.

        Args:
            model: The model to train. The trainer automatically handles DDP wrapping.
            inputs: A dictionary from the data collator, expected to contain:
                    - "input_ids"
                    - "attention_mask"
                    - "labels" (used for both NPO NLL and retain NLL)
                    - "factor": A tensor indicating sample type (-1.0 for forget, 1.0 for retain).
            return_outputs: Whether to return model outputs (not fully supported here).

        Returns:
            The total computed loss (scalar tensor).
        """
        # Ensure ref_model is on the same device as the current model replica
        # This is crucial in DDP where model parts are on different GPUs.
        current_model_device = next(model.parameters()).device
        if next(self.ref_model.parameters()).device != current_model_device:
            self.ref_model.to(current_model_device)
            # print(f"Rank {self.accelerator.process_index}: Moved ref_model to {current_model_device} in compute_loss")


        if "factor" not in inputs:
            raise ValueError("Input dictionary must contain a 'factor' key for NPOTrainer.")
        if "labels" not in inputs:
             raise ValueError("Input dictionary must contain a 'labels' key.")

        factors = inputs["factor"]
        forget_mask = (factors == -1.0)
        retain_mask = (factors == 1.0)

        total_loss = torch.tensor(0.0, device=current_model_device)
        forget_loss = torch.tensor(0.0, device=current_model_device)
        retain_loss = torch.tensor(0.0, device=current_model_device)

        # --- 1. Compute Forget Loss (NPO Component) ---
        if forget_mask.any():
            # Select only the forget samples for NPO calculation
            # Important: Create copies or views for the sub-batch dictionary
            forget_inputs = {}
            for key, value in inputs.items():
                 if isinstance(value, torch.Tensor) and value.shape[0] == factors.shape[0]:
                     forget_inputs[key] = value[forget_mask].clone() # Use clone if modifications happen downstream
                 elif key != "factor": # Keep other non-tensor inputs if any
                      forget_inputs[key] = value # Be careful if these are modified

            if not forget_inputs or "input_ids" not in forget_inputs or forget_inputs["input_ids"].shape[0] == 0:
                 print("Warning: No forget samples found in batch or forget_inputs malformed.")
            else:
                # NPO uses DPO loss with only "lose" inputs (the forget data)
                npo_loss_component, _, _ = compute_dpo_loss(
                    policy_model=model, # Pass the currently processed model replica
                    ref_model=self.ref_model,
                    win_inputs=None,
                    lose_inputs=forget_inputs,
                    beta=self.beta,
                )
                forget_loss = npo_loss_component

        # --- 2. Compute Retain Loss ---
        if retain_mask.any():
            # Select only the retain samples
            retain_inputs = {}
            for key, value in inputs.items():
                 if isinstance(value, torch.Tensor) and value.shape[0] == factors.shape[0]:
                     retain_inputs[key] = value[retain_mask].clone()
                 elif key != "factor":
                      retain_inputs[key] = value

            if not retain_inputs or "input_ids" not in retain_inputs or retain_inputs["input_ids"].shape[0] == 0:
                print("Warning: No retain samples found in batch or retain_inputs malformed.")
            else:
                if self.retain_loss_type == "NLL":
                    # Standard Causal LM loss on the retain set
                    # model(**retain_inputs).loss should return the mean loss for the retain sub-batch
                    outputs = model(**retain_inputs)
                    retain_loss = outputs.loss
                    if retain_loss is None: # Handle models not returning loss directly
                         print("Warning: model(**retain_inputs) did not return a loss for NLL retain calculation.")
                         retain_loss = torch.tensor(0.0, device=current_model_device)

                elif self.retain_loss_type == "KL":
                    # KL divergence between model and ref_model on the retain set
                    kl_loss = compute_kl_divergence(
                        policy_model=model,
                        ref_model=self.ref_model,
                        inputs=retain_inputs,
                        return_outputs=False
                    )
                    retain_loss = kl_loss

        # --- 3. Combine Losses ---
        # Ensure losses are valid scalars before combining
        if not isinstance(forget_loss, torch.Tensor): forget_loss = torch.tensor(forget_loss, device=current_model_device)
        if not isinstance(retain_loss, torch.Tensor): retain_loss = torch.tensor(retain_loss, device=current_model_device)

        if torch.isnan(forget_loss) or torch.isinf(forget_loss):
            print("Warning: NPO/Forget loss is NaN or Inf. Setting to 0.")
            forget_loss = torch.tensor(0.0, device=current_model_device)
        if torch.isnan(retain_loss) or torch.isinf(retain_loss):
            print("Warning: Retain loss is NaN or Inf. Setting to 0.")
            retain_loss = torch.tensor(0.0, device=current_model_device)

        total_loss = self.gamma * forget_loss + self.alpha * retain_loss

        # The Trainer framework handles gradient accumulation and DDP reduction.
        # We just need to return the scalar loss for this micro-batch on this rank.

        # return (total_loss, outputs) if return_outputs else total_loss # outputs would be tricky here
        return total_loss

class DPOTrainer(Trainer):
     def compute_loss(model, ref_model, inputs, beta=0.1, return_outputs=False, num_items_in_batch=None):
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            with torch.no_grad():
                idk_outputs_oracle = ref_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = ref_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits
            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
          
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            print(loss.item())
            outputs = forget_outputs
            return loss
     
    
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