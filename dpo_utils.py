import logging
from copy import deepcopy
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import Trainer
from typing import Any, Dict, List, Optional,  Union
import torch
import ast

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_prompt_response_from_chat(chat_data: list, tokenizer: AutoTokenizer):
    # ... (keep the function as defined before)
    if not chat_data or not isinstance(chat_data[-1], dict) or chat_data[-1].get('role') != 'assistant': # Added type check
        log.warning(f"Chat data is not a list ending with an assistant dict: {chat_data}")
        return None, None
    # ... rest of the function
    final_response = chat_data[-1]['content']
    prompt_chat = chat_data[:-1] # Everything before the last assistant message
    formatted_prompt = tokenizer.apply_chat_template(
        prompt_chat,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted_prompt, final_response



# Inside dpo_utils.py

def process_chat_data(df, tokenizer, log):
    # ... (previous code for parsing) ...
    processed_data = []
    for index, row in df.iterrows():
        try:
            # ... (parsing logic with ast.literal_eval) ...
            chosen_chat_list = ast.literal_eval(chosen_chat_str)
            rejected_chat_list = ast.literal_eval(rejected_chat_str)
            # ... (checks for list type) ...

            formatted_prompt_chosen, final_response_chosen = extract_prompt_response_from_chat(chosen_chat_list, tokenizer)
            formatted_prompt_rejected, final_response_rejected = extract_prompt_response_from_chat(rejected_chat_list, tokenizer)

            # --- Crucial Check ---
            if formatted_prompt_chosen is None or formatted_prompt_rejected is None:
                log.warning(f"Skipping row {index} due to invalid chat format detected by extract function.")
                continue
            if formatted_prompt_chosen != formatted_prompt_rejected:
                log.warning(f"Skipping row {index} due to mismatched prompts after formatting.")
                continue

            # --- ADD THIS CHECK ---
            # Ensure final responses are not empty strings, as this leads to issues in collator
            if not final_response_chosen or not isinstance(final_response_chosen, str):
                log.warning(f"Skipping row {index} due to empty or invalid 'chosen' response: {final_response_chosen}")
                continue
            if not final_response_rejected or not isinstance(final_response_rejected, str):
                log.warning(f"Skipping row {index} due to empty or invalid 'rejected' response: {final_response_rejected}")
                continue
            # --- END OF ADDED CHECK ---


            processed_data.append({
                'prompt': formatted_prompt_chosen,
                'chosen': final_response_chosen,
                'rejected': final_response_rejected
            })

        # ... (except blocks remain the same) ...
        except (ValueError, SyntaxError) as e:
             # ...
             continue
        except Exception as e:
             # ...
             continue

    return processed_data



def compute_batch_nll(model, inputs):
    """
    Compute the negative log-likelihood for each sequence in a batch independently.
    Args:
        model: The model to compute NLL with.
        inputs: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
                'labels' are expected to be the same as 'input_ids' with padding tokens (-100).
    Returns:
        A tensor of shape (batch_size,) containing the NLL for each sequence,
        and the model outputs.
    """
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    # Shift so that tokens < n predict n
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()

    # Calculate per-token loss
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # loss shape: (batch_size, seq_len - 1)
    loss = loss_function(shifted_logits.transpose(-1, -2), shifted_labels)

    # Sum loss over sequence dimension
    # seq_nll shape: (batch_size,)
    seq_nll = loss.sum(dim=-1)
    return seq_nll, outputs


def compute_dpo_loss(policy_model,
                     ref_model,
                     win_inputs,
                     lose_inputs,
                     beta=0.1):
    """
    Compute the DPO loss for a batch of paired examples.
    Args:
        policy_model: The model being trained.
        ref_model: The frozen reference model.
        win_inputs: Dictionary of inputs for the chosen responses.
        lose_inputs: Dictionary of inputs for the rejected responses.
        beta: The DPO temperature parameter.
    Returns:
        Scalar DPO loss, policy chosen outputs, policy rejected outputs.
    """

    # Policy model NLL calculations (requires gradients)
    win_policy_nll, win_policy_outputs = compute_batch_nll(policy_model, win_inputs)
    lose_policy_nll, lose_policy_outputs = compute_batch_nll(policy_model, lose_inputs)

    # Reference model NLL calculations (no gradients needed)
    with torch.no_grad():
        win_ref_nll, _ = compute_batch_nll(ref_model, win_inputs)
        lose_ref_nll, _ = compute_batch_nll(ref_model, lose_inputs)

    # Convert NLL to Log Probabilities (log P = -NLL)
    # Note: The signs are flipped compared to the original code's ratio calculation
    # because we directly use the definition: log P_pi(y) - log P_ref(y)
    win_logp_policy = -win_policy_nll
    win_logp_ref = -win_ref_nll
    lose_logp_policy = -lose_policy_nll
    lose_logp_ref = -lose_ref_nll

    # Calculate log ratios
    pi_logratios = win_logp_policy - lose_logp_policy # log(pi_policy(win) / pi_policy(lose))
    ref_logratios = win_logp_ref - lose_logp_ref    # log(pi_ref(win) / pi_ref(lose))

    # DPO loss formula (standard version)
    # log(sigmoid( beta * ( (logP_policy(win) - logP_ref(win)) - (logP_policy(lose) - logP_ref(lose)) ) ))
    # = log(sigmoid( beta * ( (logP_policy(win) - logP_policy(lose)) - (logP_ref(win) - logP_ref(lose)) ) ))
    # = log(sigmoid( beta * (pi_logratios - ref_logratios) ))
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean() # Average over the batch

    return loss, win_policy_outputs, lose_policy_outputs


class CustomDpoTrainer(Trainer):
    def __init__(self, *args, ref_model: nn.Module = None, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")

        self.ref_model = ref_model
        self.beta = beta

        # Ensure the reference model is on the right device and in eval mode
        # Trainer will handle device placement for the main 'model'
        # Place ref_model on the same device *once*
        # Note: If using FSDP or complex setups, this might need adjustment
        if self.args.world_size > 1:
             # If using DDP, ensure ref_model is on the correct local rank device
             self.ref_model = self.ref_model.to(self.args.device)

        self.ref_model.eval()


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By overriding this method, we implement DPO loss calculation.
        """
        # Ensure ref_model is on the correct device (important if devices change or in specific setups)
        # This might be slightly redundant if placed correctly in __init__, but safer.
        if self.ref_model.device != model.device:
             self.ref_model = self.ref_model.to(model.device)

        # Prepare inputs for chosen and rejected responses
        win_inputs = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"],
            "labels": inputs["chosen_labels"],
        }
        lose_inputs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"],
            "labels": inputs["rejected_labels"],
        }

        # Call the DPO loss function
        # 'model' is the policy model being trained (potentially DDP wrapped)
        loss, win_outputs, lose_outputs = compute_dpo_loss(
            policy_model=model,
            ref_model=self.ref_model,
            win_inputs=win_inputs,
            lose_inputs=lose_inputs,
            beta=self.beta
        )

        # If you need metrics or want to return outputs, handle them here
        # For standard training, just returning the loss is sufficient
        outputs = (win_outputs, lose_outputs) if return_outputs else None
        return (loss, outputs) if return_outputs else loss
    

    
@dataclass
class ChatDPODataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None # Max length for the *entire* sequence (prompt + response)
    max_prompt_length: Optional[int] = None # Optional: Max length for the prompt part
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        chosen_batch = []
        rejected_batch = []

        if not features:
            return batch

        for feature in features:
            prompt = feature['prompt'] # This is the already formatted prompt string
            chosen = feature['chosen']
            rejected = feature['rejected']

            # Tokenize prompt (without padding/truncation yet)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False) # Assuming template added special tokens
            # Apply prompt truncation if specified
            if self.max_prompt_length:
                prompt_tokens = {k: v[-self.max_prompt_length:] for k, v in prompt_tokens.items()}


            # Tokenize responses, adding EOS token
            # Important: Check if your tokenizer/chat template *already* adds EOS after assistant response
            # If it does, you might not need `+ self.tokenizer.eos_token` here. Test this!
            chosen_tokens = self.tokenizer(chosen + self.tokenizer.eos_token, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected + self.tokenizer.eos_token, add_special_tokens=False)

            # --- Combine prompt and response tokens ---
            chosen_sequence = {
                'input_ids': prompt_tokens['input_ids'] + chosen_tokens['input_ids'],
                'attention_mask': prompt_tokens['attention_mask'] + chosen_tokens['attention_mask']
            }
            rejected_sequence = {
                'input_ids': prompt_tokens['input_ids'] + rejected_tokens['input_ids'],
                'attention_mask': prompt_tokens['attention_mask'] + rejected_tokens['attention_mask']
            }

            # --- Create Labels (Mask Prompt Tokens) ---
            chosen_labels = [-100] * len(prompt_tokens['input_ids']) + chosen_tokens['input_ids']
            rejected_labels = [-100] * len(prompt_tokens['input_ids']) + rejected_tokens['input_ids']

            # --- Apply Max Length Truncation (Entire Sequence) ---
            if self.max_length:
                for seq in [chosen_sequence, rejected_sequence]:
                     for key in ['input_ids', 'attention_mask']:
                         seq[key] = seq[key][:self.max_length]
                chosen_labels = chosen_labels[:self.max_length]
                rejected_labels = rejected_labels[:self.max_length]


            chosen_batch.append({**chosen_sequence, 'labels': chosen_labels})
            rejected_batch.append({**rejected_sequence, 'labels': rejected_labels})


        # --- Padding ---
        # Pad the chosen sequences
        padded_chosen = self.tokenizer.pad(
            chosen_batch,
            padding=self.padding,
            max_length=self.max_length, # Use max_length for padding
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad the rejected sequences
        padded_rejected = self.tokenizer.pad(
            rejected_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Reconstruct the batch keys expected by the CustomDpoTrainer
        batch['chosen_input_ids'] = padded_chosen['input_ids']
        batch['chosen_attention_mask'] = padded_chosen['attention_mask']
        batch['chosen_labels'] = padded_chosen['labels']
        batch['rejected_input_ids'] = padded_rejected['input_ids']
        batch['rejected_attention_mask'] = padded_rejected['attention_mask']
        batch['rejected_labels'] = padded_rejected['labels']

        return batch