from data_module import convert_raw_data_to_model_qa
import torch
from forget_trainer import get_batch_loss
import torch.nn.functional as F
import torch.nn as nn
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from typing import Any
import pandas as pd
import copy
from dpo_data_module import TwoStreamBatchSampler, RetainOnlyDataset, ForgetIdkRetainDataset, mixed_collate_fn_dpo


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


def compute_dpo_loss(model, ref_model, win_inputs = None, lose_inputs = None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs cannot be None")
    
    win_log_ratio, lose_log_ratio = 0.0, 0.0

    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_outputs = model(**win_inputs)
        #print("policy.training:",    model.training)  
        win_logits = win_outputs.logits
        #print("win_logits.grad_fn:", win_logits.grad_fn)
        #print("win_logits.requires_grad:", win_logits.requires_grad)
        #print("  win_logits.requires_grad:", win_logits.requires_grad)
        win_loss = get_batch_loss(win_logits, win_inputs['labels'])
        with torch.no_grad():
            win_ref_outputs = ref_model(**win_inputs)
        win_ref_logits = win_ref_outputs.logits
        win_ref_loss = get_batch_loss(win_ref_logits, win_inputs['labels'])
        win_log_ratio = - (win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_outputs = model(**lose_inputs)
        lose_logits = lose_outputs.logits
        lose_loss = get_batch_loss(lose_logits, lose_inputs['labels'])
        with torch.no_grad():
            lose_ref_outputs = ref_model(**lose_inputs)
        lose_ref_logits = lose_ref_outputs.logits
        lose_ref_loss = get_batch_loss(lose_ref_logits, lose_inputs['labels'])
        lose_log_ratio = - (lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_retain_loss(model, retain_inputs):  
    retain_outputs = model(**retain_inputs)
    retain_loss = 0.0
    retain_loss += retain_outputs.loss
    return retain_loss

class VanillaDPOTrainer(Trainer):
    def __init__(self,
                 ref_model,        
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 **hf_trainer_kwargs 
                ):
        super().__init__(**hf_trainer_kwargs)

        self.beta  = beta
        self.gamma = gamma

        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 
        self.ref_model = self._prepare_ref_model(ref_model)
        

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids":      inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels":         inputs["answer_labels"],
        }
        idk_inputs = {
            "input_ids":      inputs["idk_input_ids"],
            "attention_mask": inputs["idk_attention_mask"],
            "labels":         inputs["idk_labels"],
        }

        forget_loss, forget_outputs = compute_dpo_loss(
            model      = model,
            ref_model  = self.ref_model,
            win_inputs = idk_inputs,
            lose_inputs=forget_inputs,
            beta       = self.beta,
        )
        loss = self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss


class VanillaNPOTrainer(Trainer):
    def __init__(self,
                 ref_model,         
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 **hf_trainer_kwargs 
                ):
        super().__init__(**hf_trainer_kwargs)

        self.beta  = beta
        self.gamma = gamma
        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 
        self.ref_model = self._prepare_ref_model(ref_model)
        

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids":      inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels":         inputs["answer_labels"],
        }
        idk_inputs = {
            "input_ids":      inputs["idk_input_ids"],
            "attention_mask": inputs["idk_attention_mask"],
            "labels":         inputs["idk_labels"],
        }

        forget_loss, forget_outputs = compute_dpo_loss(
            model      = model,
            ref_model  = self.ref_model,
            win_inputs = None,
            lose_inputs=forget_inputs,
            beta       = self.beta,
        )
        loss = self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss



class RetainDPOTrainer(Trainer):
    def __init__(self,
                 ref_model,        
                 beta: float = 0.1,
                 gamma: float = 1.0,
                 alpha: float = 1.0,
                 **hf_trainer_kwargs 
                ):
        super().__init__(**hf_trainer_kwargs)

        self.beta  = beta
        self.gamma = gamma
        self.alpha = alpha

        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 
        self.ref_model = self._prepare_ref_model(ref_model)
        

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids":      inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels":         inputs["answer_labels"],
        }
        idk_inputs = {
            "input_ids":      inputs["idk_input_ids"],
            "attention_mask": inputs["idk_attention_mask"],
            "labels":         inputs["idk_labels"],
        }
        retain_inputs = {
            "input_ids":      inputs["retain_input_ids"],
            "attention_mask": inputs["retain_attention_mask"],
            "labels":         inputs["retain_labels"],
        }

        forget_loss, forget_outputs = compute_dpo_loss(
            model      = model,
            ref_model  = self.ref_model,
            win_inputs = idk_inputs,
            lose_inputs=forget_inputs,
            beta       = self.beta,
        )

        retain_loss = compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
    


class RetainNPOTrainer(Trainer):
    def __init__(self,
                 ref_model,        
                 beta: float = 0.1,
                 gamma: float = 1.0,
                 alpha: float = 1.0,
                 **hf_trainer_kwargs 
                ):
        super().__init__(**hf_trainer_kwargs)

        self.beta  = beta
        self.gamma = gamma
        self.alpha = alpha

        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 
        self.ref_model = self._prepare_ref_model(ref_model)
        

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        return self.accelerator.prepare_model(ref_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids":      inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels":         inputs["answer_labels"],
        }
        idk_inputs = {
            "input_ids":      inputs["idk_input_ids"],
            "attention_mask": inputs["idk_attention_mask"],
            "labels":         inputs["idk_labels"],
        }
        retain_inputs = {
            "input_ids":      inputs["retain_input_ids"],
            "attention_mask": inputs["retain_attention_mask"],
            "labels":         inputs["retain_labels"],
        }

        forget_loss, forget_outputs = compute_dpo_loss(
            model      = model,
            ref_model  = self.ref_model,
            win_inputs = None,
            lose_inputs=forget_inputs,
            beta       = self.beta,
        )

        retain_loss = compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
    


class RetainDPOTrainerWithSampler(RetainDPOTrainer):
    def __init__(self, args: TrainingArguments = None,
                 forget_dataset: Dataset = None,
                 retain_dataset: Dataset = None,
                 n_forget_per_batch: int = 1,
                 batch_size: int = 8,
                 tokenizer = None,
                 **kwargs,):
        
        if forget_dataset is None or retain_dataset is None:
            raise ValueError("Both forget_dataset and retain_dataset must be provided.")

        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.n_forget_per_batch = n_forget_per_batch
        self._train_batch_size = batch_size

        self.combined_dataset = ConcatDataset([self.forget_dataset, self.retain_dataset])

        super().__init__(
            model = kwargs.get('model'),
            ref_model = kwargs.get('ref_model'),
            beta = kwargs.get('beta', 0.1),
            gamma = kwargs.get('gamma', 1.0),
            alpha = kwargs.get('alpha', 1.0),
            args = args,
            train_dataset = self.combined_dataset,
            data_collator = None,
            **{k:v for k,v in kwargs.items() if k not in ['model', 'ref_model', 'beta', 'gamma', 'alpha']}
        )
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Overrides the default method to use Twostreambatchsampler and the simple collate function
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        total_batch_size = self._train_batch_size
        num_forget_total = len(self.forget_dataset)
        num_retain_total = len(self.retain_dataset)

        n_retain_per_batch = total_batch_size - self.n_forget_per_batch


        if n_retain_per_batch < 0:
            raise ValueError("n_forget_per_batch cannot exceed total batch size.")

        if n_retain_per_batch == 0 and self.alpha > 0:
            print("Warning: n_retain_per_batch is 0, but forget loss weight (alpha) > 0")

        if self.n_forget_per_batch == 0 and self.gamma > 0:
            print("Warning: n_forget_per_batch is 0, but forget loss weight (gamma) > 0")

        
        # --- creating the sampler ---
        batch_sampler = TwoStreamBatchSampler(
                forget_idx=range(num_forget_total),
                retain_idx = range(num_retain_total),
                batch_size = total_batch_size,
                primary_batch_size = self.n_forget_per_batch,
            )

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": mixed_collate_fn_dpo,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        return DataLoader(
            self.combined_dataset,
            **dataloader_params,
        )

    
