import torch
from forget_trainer import get_batch_loss
import torch.nn.functional as F
import torch.nn as nn
from accelerate import Accelerator
from transformers import Trainer

accelerator = Accelerator()


def get_batch_loss(output, labels):
    # when passed a ModelOutput or tuple, extract the first item
    if not torch.is_tensor(output):
        if hasattr(output, "logits"):
            output = output.logits
        else:
            output = output[0]

    shifted_labels = labels[..., 1:].contiguous()
    output         = output[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss    = loss_fn(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss

def compute_retain_loss(model, retain_inputs):  
    retain_outputs = model(**retain_inputs)
    retain_loss = 0.0
    retain_loss += retain_outputs.loss
    return retain_loss



class SimNPO(Trainer):
    def __init__(self, delta = 0.0, beta = 1.0, **hf_trainer_kwargs):
        super().__init__(**hf_trainer_kwargs)
        self.delta = delta
        self.beta = beta

        # self.model = self.accelerator.prepare_model(
        #                     self.model, 
        #                     evaluation_mode=False
        # )

        # self.model.train()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs, retain_inputs = inputs
        f_input_ids, f_labels, f_attention_mask = forget_inputs
        loss_mask = f_labels != -100
        f_outputs = model(input_ids = f_input_ids, labels = f_labels, attention_mask = f_attention_mask)
        f_loss = get_batch_loss(f_outputs, f_labels)
        f_loss = f_loss / loss_mask.sum(-1) - self.delta
        f_loss = -F.logsigmoid(self.beta * f_loss).mean() * 2 / self.beta
        r_input_ids, r_labels, r_attention_mask = retain_inputs
        r_outputs = model(input_ids = r_input_ids, labels = r_labels, attention_mask = r_attention_mask)
        r_loss = r_outputs.loss 
        loss =  f_loss + r_loss
        return (loss, f_outputs) if return_outputs else loss
    
    

