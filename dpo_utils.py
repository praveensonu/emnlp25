import torch
from forget_trainer import get_batch_loss
import torch.nn.functional as F
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
import copy
from collators import dpo_retain_collator
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


def compute_dpo_loss(model, ref_model, win_inputs = None, lose_inputs = None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs cannot be None")
    
    win_log_ratio, lose_log_ratio = 0.0, 0.0

    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_outputs = model(**win_inputs)
        win_logits = win_outputs.logits
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
    



class BatchRetainDPOTrainer(Trainer):
    def __init__(self,
                 ref_model,
                 beta: float = 0.1,
                 gamma: float = 1.0,
                 alpha: float = 1.0,
                 **hf_trainer_kwargs):
        super().__init__(**hf_trainer_kwargs)

        self.beta   = beta
        self.gamma  = gamma
        self.alpha  = alpha

        if ref_model is None:
            raise ValueError("ref_model must be provided for DPO training.")
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 
        self.ref_model = self._prepare_ref_model(ref_model)

        if self.data_collator is None:
            self.data_collator = dpo_retain_collator

        
    def _prepare_ref_model(self, ref_model):
        if self.accelerator.is_local_main_process:
            print("Preparing reference model...")
    
        print(f"Rank {self.accelerator.process_index}: In _prepare_ref_model. Accelerator device: {self.accelerator.device}, torch.cuda.current_device(): {torch.cuda.current_device()}") 
        prepared_ref_model  = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        prepared_ref_model.eval()
        for param in prepared_ref_model.parameters():
            param.requires_grad = False
        if self.accelerator.is_local_main_process:
            print("Reference model prepared and set to eval mode.")
        return prepared_ref_model
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training ['torch.utils.data.DataLoader'] 
        will use no shuffle for this trainer to support our interleaving dataset
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        data_collator = self.data_collator if self.data_collator is not None else dpo_retain_collator

        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory" : self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

            if self.args.world_size > 1:
                dataloader_params["sampler"] = DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=False # for maintaining interleaved order across GPUs
                )
            else:
                dataloader_params["sampler"] = SequentialSampler(train_dataset) # doing this for single GPU

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        factors = inputs["factor"]
        device = factors.device

        total_forget_dpo_loss = torch.tensor(0.0, device=device)
        total_retain_ce_loss = torch.tensor(0.0, device=device)

        policy_outputs_dict = {}

        forget_mask = factors < 0.0
        if forget_mask.any():
            num_forget_samples = forget_mask.sum().item()
            #print(f"num_forget_samples: {num_forget_samples}")
            win_inputs_forget = {
                "input_ids":      inputs["idk_input_ids"][forget_mask],
                "attention_mask": inputs["idk_attention_mask"][forget_mask],
                "labels":         inputs["idk_labels"][forget_mask],
            }

            lose_inputs_forget = {
                "input_ids":      inputs["answer_input_ids"][forget_mask],
                "attention_mask": inputs["answer_attention_mask"][forget_mask],
                "labels":         inputs["answer_labels"][forget_mask],
            }

            if win_inputs_forget["input_ids"].shape[0] > 0:
                forget_loss_val, dpo_policy_outputs = compute_dpo_loss(
                    model      = model,
                    ref_model  = self.ref_model,
                    win_inputs = win_inputs_forget,
                    lose_inputs=lose_inputs_forget,
                    beta       = self.beta,
                )
                total_forget_dpo_loss = forget_loss_val

                if return_outputs:
                    policy_outputs_dict.update({f"dpo_{k}": v for k,v in dpo_policy_outputs.items()})
            else:
                if self.accelerator.is_local_main_process:
                    print("Warning: forget_mask was true, but resulted in 0 forget samples after indexing.")

        retain_mask = factors > 0.0
        if retain_mask.any():
            num_retain_samples = retain_mask.sum().item()
            #print(f"num_retain_samples: {num_retain_samples}")
            current_retain_inputs = {
                "input_ids":      inputs["answer_input_ids"][retain_mask],
                "attention_mask": inputs["answer_attention_mask"][retain_mask],
                "labels":         inputs["answer_labels"][retain_mask],
            }

            if current_retain_inputs["input_ids"].shape[0] > 0:
                retain_loss_val = compute_retain_loss(model = model, retain_inputs=current_retain_inputs)
                total_retain_ce_loss = retain_loss_val
            else:
                if self.accelerator.is_local_main_process:
                    print("Warning: retain_mask was true, but resulted in 0 retain samples after indexing.")

        
        loss = self.gamma * total_forget_dpo_loss + self.alpha * total_retain_ce_loss

        return (loss, policy_outputs_dict) if return_outputs else loss
    
