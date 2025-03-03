from transformers import Trainer
import torch

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