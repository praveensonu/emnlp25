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



def compute_batch_nll(model, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Tuple[torch.Tensor, CausalLMOutputWithPast]:
    """
    Calculates the negative log-likelihood loss for each sequence in a batch.

    Args:
        model: The model to compute NLL for.
        inputs: Dictionary containing model inputs (e.g., "input_ids", "attention_mask", "labels").
                "labels" are required.

    Returns:
        A tuple containing:
        - loss: A tensor of shape (batch_size,) with the sum NLL loss for each sequence.
        - outputs: The raw outputs from the model.
    """
    if "labels" not in inputs:
        raise ValueError("compute_batch_nll requires 'labels' in inputs.")

    # Perform forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    # --- Shift logic: predict next token ---
    # Ensure logits and labels are compatible shapes for shifting
    # Generally, labels sequence length should match logits sequence length
    if logits.shape[1] != labels.shape[1]:
         # This case might indicate a problem with data preparation or model config
         print(f"Warning: Logits sequence length ({logits.shape[1]}) differs from labels sequence length ({labels.shape[1]})")
         # Decide how to handle: error, truncation, padding?
         # For now, let's try to align based on the shorter length if possible, but it's risky.
         # A safer default might be to raise an error.
         min_len = min(logits.shape[1], labels.shape[1])
         if min_len <= 1:
              # Cannot shift if sequence length is 0 or 1 after potential truncation
              print("Error: Sequence length too short for shifting after potential mismatch.")
              # Return zero loss or raise error
              return torch.zeros(logits.size(0), device=logits.device), outputs
         logits = logits[:, :min_len, :]
         labels = labels[:, :min_len]
         print(f"Warning: Truncated logits and labels to min length {min_len}")


    # Handle sequences too short to shift (length 0 or 1)
    if logits.shape[1] <= 1:
        print(f"Warning: Sequence length ({logits.shape[1]}) too short to compute shifted NLL. Returning zero loss.")
        return torch.zeros(logits.size(0), device=logits.device), outputs

    # Perform the shift
    shifted_labels = labels[..., 1:].contiguous()
    logits_shifted = logits[..., :-1, :].contiguous()

    # --- Calculate Loss ---
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    # Get dimensions *from the shifted tensors* that determine the loss size
    batch_size, seq_len_shifted, vocab_size = logits_shifted.shape

    # Calculate loss per token, ignoring padding (-100)
    # Reshape for CrossEntropyLoss: (Batch * SeqLenShifted, VocabSize) vs (Batch * SeqLenShifted,)
    loss = loss_function(logits_shifted.view(-1, vocab_size),
                         shifted_labels.view(-1))

    # loss now has shape (batch_size * seq_len_shifted,)

    # --- Reshape the loss tensor ---
    # Use the dimensions derived *directly* from the shifted tensors
    # Target shape is (batch_size, seq_len_shifted)
    try:
        loss = loss.view(batch_size, seq_len_shifted)
    except RuntimeError as e:
        # Add more debug info if it still fails
        print(f"Internal Shape Error during view: loss size {loss.numel()}, target view ({batch_size}, {seq_len_shifted})")
        print(f"Original logits shape: {outputs.logits.shape}, Original labels shape: {inputs['labels'].shape}")
        print(f"Shifted logits shape: {logits_shifted.shape}, Shifted labels shape: {shifted_labels.shape}")
        raise e # Re-raise the error

    # Sum loss over the sequence dimension for each item in the batch
    # loss now has shape (batch_size, seq_len_shifted)
    loss = loss.sum(dim=-1) # Sum over the sequence length dimension -> shape (batch_size,)

    return loss, outputs



def compute_dpo_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    win_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
    lose_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    Computes the DPO loss.

    Args:
        policy_model: The model being trained.
        ref_model: The frozen reference model.
        win_inputs: Batch of preferred examples. Dict format for compute_batch_nll.
        lose_inputs: Batch of dispreferred examples. Dict format for compute_batch_nll.
        beta: Temperature parameter for DPO.

    Returns:
        A tuple containing:
        - loss: The computed DPO loss.
        - policy_logps: Tuple of (policy_win_logps, policy_lose_logps).
        - ref_logps: Tuple of (ref_win_logps, ref_lose_logps).
    """
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs cannot be None")

    # --- Calculate Log Probs ---
    policy_win_logps, ref_win_logps = None, None
    policy_lose_logps, ref_lose_logps = None, None

    if win_inputs is not None:
        # Policy model NLL (negative log prob)
        policy_win_nll, _ = compute_batch_nll(policy_model, win_inputs)
        policy_win_logps = -policy_win_nll # Convert NLL to LogProb

        # Reference model NLL (negative log prob)
        with torch.no_grad():
            ref_win_nll, _ = compute_batch_nll(ref_model, win_inputs)
            ref_win_logps = -ref_win_nll # Convert NLL to LogProb

    if lose_inputs is not None:
        # Policy model NLL
        policy_lose_nll, _ = compute_batch_nll(policy_model, lose_inputs)
        policy_lose_logps = -policy_lose_nll

        # Reference model NLL
        with torch.no_grad():
            ref_lose_nll, _ = compute_batch_nll(ref_model, lose_inputs)
            ref_lose_logps = -ref_lose_nll

    # --- Combine Log Probs ---
    # Initialize log ratios to 0 or a default large negative value if one side is missing
    pi_logratios = torch.tensor(0.0, device=policy_model.device)
    ref_logratios = torch.tensor(0.0, device=policy_model.device)

    if policy_win_logps is not None and policy_lose_logps is not None:
        # DPO uses logp(y_w) - logp(y_l)
        pi_logratios = policy_win_logps - policy_lose_logps
    elif policy_win_logps is not None: # Only win inputs
        # Often handled by specific logic in methods like IPO, not standard DPO
        # For NPO (win=None), this case isn't hit for pi_logratios calculation
        # If DPO needs to handle only win, needs modification
        pi_logratios = policy_win_logps # Or adjust based on specific algorithm needs
    elif policy_lose_logps is not None: # Only lose inputs (like in NPO context)
         # DPO loss expects difference. If only lose, the "win" part is implicitly 0?
         # The NPO usage handles this by having win_log_ratio = 0 later.
         # Let's calculate as if win_logp = 0 for symmetry:
         pi_logratios = 0.0 - policy_lose_logps # Equivalent to policy_lose_nll

    if ref_win_logps is not None and ref_lose_logps is not None:
        ref_logratios = ref_win_logps - ref_lose_logps
    elif ref_win_logps is not None:
        ref_logratios = ref_win_logps
    elif ref_lose_logps is not None:
        # Consistent with pi_logratios:
        ref_logratios = 0.0 - ref_lose_logps # Equivalent to ref_lose_nll


    # Calculate the final DPO log ratio difference used in the loss
    # This is log( (pi(win)/ref(win)) / (pi(lose)/ref(lose)) )
    # = (log pi(win) - log ref(win)) - (log pi(lose) - log ref(lose))
    # = (log pi(win) - log pi(lose)) - (log ref(win) - log ref(lose))
    # = pi_logratios - ref_logratios

    # Handle cases where only one type of input was provided
    if win_inputs is None: # NPO case: lose_inputs provided, win_inputs=None
        # policy_win_logps and ref_win_logps are None
        # pi_logratios = 0 - policy_lose_logps = policy_lose_nll
        # ref_logratios = 0 - ref_lose_logps = ref_lose_nll
        # loss based on -(policy_lose_nll - ref_lose_nll) = ref_lose_nll - policy_lose_nll
        # This represents log( policy(lose) / ref(lose) )
        logits = beta * (pi_logratios - ref_logratios) # beta * (policy_lose_nll - ref_lose_nll) WRONG SIGNAGE
        # Let's directly use the NPO formulation's input to logsigmoid:
        # NPO wants log( policy(lose) / ref(lose) )
        # = log policy(lose) - log ref(lose)
        # = -policy_lose_nll - (-ref_lose_nll) = ref_lose_nll - policy_lose_nll
        log_ratio = ref_lose_nll - policy_lose_nll # Shape (batch_size_lose,)
        logits = -beta * log_ratio # We want to minimize this ratio, loss uses -logsigmoid(-beta * ratio)

    elif lose_inputs is None: # Only win_inputs provided
        # pi_logratios = policy_win_logps
        # ref_logratios = ref_win_logps
        # We want log ( policy(win) / ref(win) )
        log_ratio = policy_win_logps - ref_win_logps # Shape (batch_size_win,)
        logits = beta * log_ratio # Maximize this ratio, loss uses -logsigmoid(beta * ratio)

    else: # Both win and lose inputs provided (Standard DPO)
         # log [ (p_pi(w)/p_ref(w)) / (p_pi(l)/p_ref(l)) ]
         # = (log p_pi(w) - log p_ref(w)) - (log p_pi(l) - log p_ref(l))
         win_log_ratio  = policy_win_logps - ref_win_logps
         lose_log_ratio = policy_lose_logps - ref_lose_logps
         # Ensure they are broadcastable if batch sizes differ (shouldn't happen with typical paired data)
         logits = beta * (win_log_ratio - lose_log_ratio) # Maximize this diff, loss uses -logsigmoid(...)


    # DPO Loss: -logsigmoid(logits)
    # Note: Original code had -2/beta * logsigmoid(...). Let's stick to standard DPO loss definition first: -logsigmoid(beta * ratio_diff)
    # loss = -F.logsigmoid(logits).mean()
    # Let's match the original GitHub code's loss scaling (-2/beta * ...), assuming it's intentional for NPO.
    # The formula seems designed for the (win_log_ratio - lose_log_ratio) argument structure.
    # Re-evaluate the input to logsigmoid based on original structure: beta * (win_log_ratio - lose_log_ratio)

    win_log_ratio_val = 0.0
    lose_log_ratio_val = 0.0

    if win_inputs is not None:
        policy_win_nll, _ = compute_batch_nll(policy_model, win_inputs)
        with torch.no_grad():
            ref_win_nll, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio_val = -(policy_win_nll - ref_win_nll) # = ref_nll - policy_nll = log(policy/ref)

    if lose_inputs is not None:
        policy_lose_nll, _ = compute_batch_nll(policy_model, lose_inputs)
        with torch.no_grad():
            ref_lose_nll, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio_val = -(policy_lose_nll - ref_lose_nll) # = ref_nll - policy_nll = log(policy/ref)

    # Now calculate the argument for logsigmoid as in the original code
    logsigmoid_arg = beta * (win_log_ratio_val - lose_log_ratio_val)

    loss = -2 / beta * F.logsigmoid(logsigmoid_arg).mean() # Match original formula scaling

    # Return logps for potential analysis (or None if not computed)
    policy_logps = (policy_win_logps, policy_lose_logps)
    ref_logps = (ref_win_logps, ref_lose_logps)

    # Note: Original code returned model outputs. compute_batch_nll returns them,
    # but we aren't capturing/returning them here to keep it simpler. Add if needed.
    # Returning logps instead as they are intermediate results.
    return loss, policy_logps, ref_logps


def compute_kl_divergence(
    policy_model: nn.Module,
    ref_model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    return_outputs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, CausalLMOutputWithPast]]:
    """
    Computes KL divergence loss between policy and reference model logits.
    KL(ref || policy) = sum P_ref(x) * log(P_ref(x) / P_policy(x))

    Args:
        policy_model: The model being trained.
        ref_model: The frozen reference model.
        inputs: Dictionary containing model inputs (e.g., "input_ids", "attention_mask").
                Labels are NOT used for KL calculation directly but might be in inputs.
        return_outputs: Whether to return the policy model's outputs.

    Returns:
        KL loss (scalar tensor), or (KL loss, policy_outputs) if return_outputs is True.
    """
    # Get policy logits
    policy_outputs = policy_model(**inputs, output_hidden_states=False, output_attentions=False)
    policy_logits = policy_outputs.logits # Shape: (batch_size, seq_len, vocab_size)

    # Get reference logits
    with torch.no_grad():
        ref_outputs = ref_model(**inputs, output_hidden_states=False, output_attentions=False)
        ref_logits = ref_outputs.logits

    # --- KL Calculation ---
    # Use shifted logits like in NLL if comparing next-token prediction distributions
    policy_logits_shifted = policy_logits[..., :-1, :].contiguous()
    ref_logits_shifted = ref_logits[..., :-1, :].contiguous()

    # Calculate log probabilities (log softmax)
    policy_log_probs = F.log_softmax(policy_logits_shifted, dim=-1)
    ref_probs = F.softmax(ref_logits_shifted, dim=-1) # P_ref(x)
    # Alternative: Use ref log_probs directly in kl_div if using that formula
    # ref_log_probs = F.log_softmax(ref_logits_shifted, dim=-1)

    # KL divergence: sum_i P_ref_i * (log P_ref_i - log P_policy_i)
    # PyTorch's kl_div expects: input=log P_policy, target=P_ref
    # It computes: sum target * (log target - input) = sum P_ref * (log P_ref - log P_policy)
    # Need to handle padding. Assume attention_mask exists.
    if "attention_mask" not in inputs:
        raise ValueError("KL divergence calculation requires 'attention_mask' in inputs.")

    attention_mask = inputs["attention_mask"]
    attention_mask_shifted = attention_mask[..., 1:].contiguous() # Match shifted logits

    # Reshape for kl_div: (Batch * SeqLen, VocabSize)
    batch_size, seq_len_shifted, vocab_size = policy_log_probs.shape
    policy_log_probs_flat = policy_log_probs.view(-1, vocab_size)
    ref_probs_flat = ref_probs.view(-1, vocab_size)

    kl_loss_fn = nn.KLDivLoss(reduction="none", log_target=False) # Input is logP_policy, target is P_ref
    kl_per_token = kl_loss_fn(policy_log_probs_flat, ref_probs_flat).sum(dim=-1) # Sum over vocab dimension

    # Reshape back to (Batch, SeqLen)
    kl_per_token = kl_per_token.view(batch_size, seq_len_shifted)

    # Mask out padding tokens
    kl_per_token = kl_per_token * attention_mask_shifted

    # Average KL loss over non-masked tokens in the batch
    # Sum KL over sequence and batch, divide by number of non-masked tokens
    total_kl = kl_per_token.sum()
    num_tokens = attention_mask_shifted.sum()

    if num_tokens == 0:
       kl_loss = torch.tensor(0.0, device=policy_model.device)
    else:
       kl_loss = total_kl / num_tokens

    if return_outputs:
        return kl_loss, policy_outputs
    else:
        return kl_loss
    

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