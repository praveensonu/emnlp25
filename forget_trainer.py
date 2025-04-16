from transformers import Trainer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss


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
    


class VanillaGradDiffTrainer(Trainer): ## not using this trainer
    
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



class NPOTrainer(Trainer):
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