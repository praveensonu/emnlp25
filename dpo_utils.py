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

    loss =  -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
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
    

### this setting only works for per device train batch size of 1
## the idea is I have 2 gpus, per device is 1. so total batch size is 2. Grad accumulation steps is 4. 

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
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator # Assuming it's set

        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last, # Get from args
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                print(f"Rank {self.args.process_index}: Instantiating DistributedSampler.")
                sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=False,
                    drop_last=self.args.dataloader_drop_last # Pass drop_last here too
                )
                dataloader_params["sampler"] = sampler
                print(f"Rank {self.args.process_index}: Sampler type is {type(sampler)}")
            else:
                # This part is for single GPU, not relevant for your 2-GPU hang/ratio issue
                print(f"Rank {self.args.process_index}: Instantiating SequentialSampler.")
                dataloader_params["sampler"] = SequentialSampler(train_dataset)
        else: # IterableDataset
            dataloader_params["sampler"] = None # Sampler not used with IterableDataset usually

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params, shuffle=False))
        

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        factors = inputs["factor"]
        original_indices = inputs["original_index"]
        device = factors.device



        win_inputs_forget_path = {
        "input_ids": inputs["idk_input_ids"],
        "attention_mask": inputs["idk_attention_mask"],
        "labels": inputs["idk_labels"],
        }
        lose_inputs_forget_path = {
            "input_ids": inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels": inputs["answer_labels"],
        }

        # --- Prepare inputs for Retain CE path (answer) ---
        # These will be used by all ranks, but only contribute to loss if factor > 0
        retain_inputs_ce_path = {
            "input_ids": inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels": inputs["answer_labels"],
        }

        # --- 1. Compute "potential" DPO loss for ALL samples ---
        # The model and ref_model forward passes will occur for every sample.
        # Gradients will flow through these paths.
        current_global_step_being_accumulated_for = self.state.global_step if self.state else -1
        potential_dpo_loss_val, _ = compute_dpo_loss( # We don't need dpo_policy_outputs for this test
            model=model,
            ref_model=self.ref_model,
            win_inputs=win_inputs_forget_path,
            lose_inputs=lose_inputs_forget_path,
            beta=self.beta,
        )
        # Ensure it's a scalar tensor if compute_dpo_loss returns per-sample loss and batch_size > 1
        # For batch_size=1, it should already be a scalar or (1,) tensor. .mean() is safe.
        potential_dpo_loss_val = potential_dpo_loss_val.mean()


        # --- 2. Compute "potential" Retain CE loss for ALL samples ---
        # The model forward pass will occur for every sample.
        # Gradients will flow through this path.
        # compute_retain_loss already returns model_outputs.loss which should be a scalar
        potential_ce_loss_val = compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs_ce_path
        )
        potential_ce_loss_val = potential_ce_loss_val.mean()


        actual_dpo_contribution = torch.tensor(0.0, device=device, dtype=potential_dpo_loss_val.dtype)
        actual_ce_contribution = torch.tensor(0.0, device=device, dtype=potential_ce_loss_val.dtype)

        current_factor = factors.item()
        current_original_idx = original_indices.item()
    
        log_prefix = (f"GlobalStepAccumFor: {current_global_step_being_accumulated_for} "
                  f"Rank {self.accelerator.process_index} OrigIdx: {current_original_idx}")
        if current_factor < 0.0: # It's a forget sample
            actual_dpo_contribution = self.gamma * potential_dpo_loss_val
            print(f"{log_prefix} - DPO: {actual_dpo_contribution:.4f} (Factor: {current_factor})")
        
        elif current_factor > 0.0: # It's a retain sample
            actual_ce_contribution = self.alpha * potential_ce_loss_val
            print(f"{log_prefix} - CE: {actual_ce_contribution:.4f} (Factor: {current_factor})")

        loss = actual_dpo_contribution + actual_ce_contribution

        # --- Sanity Checks for NaN/Inf ---
        if not torch.isfinite(potential_dpo_loss_val):
            print(f"Rank {self.accelerator.process_index} - Potential DPO loss is NaN/Inf: {potential_dpo_loss_val} for factor {current_factor}")
            #  save inputs["idk_input_ids"] and inputs["answer_input_ids"] here
        if not torch.isfinite(potential_ce_loss_val):
            print(f"Rank {self.accelerator.process_index} - Potential CE loss is NaN/Inf: {potential_ce_loss_val} for factor {current_factor}")
            #  save inputs["answer_input_ids"] here
        if not torch.isfinite(loss):
            print(f"Rank {self.accelerator.process_index} - Final loss is NaN/Inf: {loss} for factor {current_factor}")
    
            # raise ValueError(f"Rank {self.accelerator.process_index} - NaN/Inf loss detected")

        
        policy_outputs_dict = {}
        return (loss, policy_outputs_dict) if return_outputs else loss
    


class BatchRetainNPOTrainer(BatchRetainDPOTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        factors = inputs["factor"]
        device = factors.device
        lose_inputs_forget_path = {
            "input_ids": inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels": inputs["answer_labels"],
        }

        # --- Prepare inputs for Retain CE path (answer) ---
        # These will be used by all ranks, but only contribute to loss if factor > 0
        retain_inputs_ce_path = {
            "input_ids": inputs["answer_input_ids"],
            "attention_mask": inputs["answer_attention_mask"],
            "labels": inputs["answer_labels"],
        }

        # --- 1. Compute "potential" DPO loss for ALL samples ---
        # The model and ref_model forward passes will occur for every sample.
        # Gradients will flow through these paths.
        potential_dpo_loss_val, _ = compute_dpo_loss( # We don't need dpo_policy_outputs for this test
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=lose_inputs_forget_path,
            beta=self.beta,
        )
        potential_dpo_loss_val = potential_dpo_loss_val.mean()


        # --- 2. Compute "potential" Retain CE loss for ALL samples ---
        # The model forward pass will occur for every sample.
        # Gradients will flow through this path.
        # compute_retain_loss already returns model_outputs.loss which should be a scalar
        potential_ce_loss_val = compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs_ce_path
        )
        potential_ce_loss_val = potential_ce_loss_val.mean()
        actual_dpo_contribution = torch.tensor(0.0, device=device, dtype=potential_dpo_loss_val.dtype)
        actual_ce_contribution = torch.tensor(0.0, device=device, dtype=potential_ce_loss_val.dtype)
        current_factor = factors.item()
        if current_factor < 0.0: # It's a forget sample
            actual_dpo_contribution = self.gamma * potential_dpo_loss_val 
        elif current_factor > 0.0: # It's a retain sample
            actual_ce_contribution = self.alpha * potential_ce_loss_val
    
        loss = actual_dpo_contribution + actual_ce_contribution

        # --- Sanity Checks for NaN/Inf ---
        if not torch.isfinite(potential_dpo_loss_val):
            print(f"Rank {self.accelerator.process_index} - Potential DPO loss is NaN/Inf: {potential_dpo_loss_val} for factor {current_factor}")
        if not torch.isfinite(potential_ce_loss_val):
            print(f"Rank {self.accelerator.process_index} - Potential CE loss is NaN/Inf: {potential_ce_loss_val} for factor {current_factor}")
        if not torch.isfinite(loss):
            print(f"Rank {self.accelerator.process_index} - Final loss is NaN/Inf: {loss} for factor {current_factor}")
            # raise ValueError(f"Rank {self.accelerator.process_index} - NaN/Inf loss detected")

        
        policy_outputs_dict = {}
        return (loss, policy_outputs_dict) if return_outputs else loss