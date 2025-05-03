from data_module import convert_raw_data_to_model_qa, SingleDataset
import torch
from forget_trainer import get_batch_loss
import torch.nn.functional as F
import torch.nn as nn
from accelerate import Accelerator
from transformers import Trainer

accelerator = Accelerator()


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def compute_dpo_loss(model, ref_model, win_inputs = None, lose_inputs = None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs cannot be None")
    
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None
    if win_outputs is None:
        win_outputs = model(**win_inputs)
        win_loss = get_batch_loss(win_outputs, win_inputs['labels'])
        with torch.no_grad():
            win_ref_outputs = ref_model(**win_inputs)
            win_ref_loss = get_batch_loss(win_ref_outputs, win_inputs['labels'])
        win_log_ratio = - (win_loss - win_ref_loss)

    if lose_outputs is None:
        lose_outputs = model(**lose_inputs)
        lose_loss = get_batch_loss(lose_outputs, lose_inputs['labels'])
        with torch.no_grad():
            lose_ref_outputs = ref_model(**lose_inputs)
            lose_ref_loss = get_batch_loss(lose_ref_outputs, lose_inputs['labels'])
        lose_log_ratio = - (lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)



class VanillaDPOTrainer(Trainer):
    def __init__(self, beta = 1.0, ref_model, model):
        super().__init__(model, ref_model)
        self.beta = beta
        if self.ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.ref_model = ref_model
        self.model = model

        ref_model = accelerator.prepare_model(ref_model, evaluation_mode = True)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        forget_inputs = 





    
