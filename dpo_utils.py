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
        data_collator = self.data_collator 
        dataloader_params = {
            "batch_size": self.args.train_batch_size, 
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
           
            if self.accelerator.num_processes <= 1: 
                print(f"Rank {self.accelerator.process_index}: Instantiating SequentialSampler for single GPU.")
                
                dataloader_params["sampler"] = None 
                dataloader_params["shuffle"] = False 
            else: 
                print(f"Rank {self.accelerator.process_index}: Instantiating DistributedSampler.")
     
                dataloader_params["sampler"] = DistributedSampler(
                    train_dataset,
                    num_replicas=self.accelerator.num_processes,
                    rank=self.accelerator.process_index,
                    shuffle=False,
                    drop_last=self.args.dataloader_drop_last
                )
                dataloader_params["shuffle"] = False 
        else: 
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
 
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        factors = inputs["factor"] 
        device = factors.device
        batch_size_on_device = factors.shape[0]
        total_loss_for_batch = torch.tensor(0.0, device=device, dtype=torch.float32) 
        actual_samples_processed_for_loss = 0        
        forget_mask = factors < 0.0
        if forget_mask.any():
            num_forget_samples = forget_mask.sum().item()
            
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
                forget_loss_val_mean, dpo_policy_outputs = compute_dpo_loss(
                    model=model,
                    ref_model=self.ref_model,
                    win_inputs=win_inputs_forget,
                    lose_inputs=lose_inputs_forget,
                    beta=self.beta,
                )
                
                total_loss_for_batch += self.gamma * forget_loss_val_mean * num_forget_samples
                actual_samples_processed_for_loss += num_forget_samples

        retain_mask = factors > 0.0
        if retain_mask.any():
            num_retain_samples = retain_mask.sum().item()

            current_retain_inputs = {
                "input_ids":      inputs["answer_input_ids"][retain_mask],
                "attention_mask": inputs["answer_attention_mask"][retain_mask],
                "labels":         inputs["answer_labels"][retain_mask],
            }

            if current_retain_inputs["input_ids"].shape[0] > 0: 
                retain_outputs = model(**current_retain_inputs) 
                retain_loss_val_mean = retain_outputs.loss 
                
                total_loss_for_batch += self.alpha * retain_loss_val_mean * num_retain_samples
                actual_samples_processed_for_loss += num_retain_samples


        if actual_samples_processed_for_loss > 0:
            final_loss = total_loss_for_batch / actual_samples_processed_for_loss
        else:

            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        if 'original_index' in inputs:
           current_original_indices = inputs["original_index"].tolist()
           factors_list = factors.tolist()
           print_info = [f"Idx:{idx}(F:{f:.0f})" for idx, f in zip(current_original_indices, factors_list)]
           print(f"SingleGPU Step {self.state.global_step if self.state else -1} BatchOrigIndices: {print_info} -> FinalLoss: {final_loss.item():.4f}")


        policy_outputs_dict = {}
        return (final_loss, policy_outputs_dict) if return_outputs else final_loss
    


class BatchRetainNPOTrainer(Trainer):
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
        data_collator = self.data_collator 
        dataloader_params = {
            "batch_size": self.args.train_batch_size, 
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
           
            if self.accelerator.num_processes <= 1: 
                print(f"Rank {self.accelerator.process_index}: Instantiating SequentialSampler for single GPU.")
                
                dataloader_params["sampler"] = None 
                dataloader_params["shuffle"] = False 
            else: 
                print(f"Rank {self.accelerator.process_index}: Instantiating DistributedSampler.")
     
                dataloader_params["sampler"] = DistributedSampler(
                    train_dataset,
                    num_replicas=self.accelerator.num_processes,
                    rank=self.accelerator.process_index,
                    shuffle=False,
                    drop_last=self.args.dataloader_drop_last
                )
                dataloader_params["shuffle"] = False 
        else: 
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
 
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        factors = inputs["factor"] 
        device = factors.device
        batch_size_on_device = factors.shape[0]
        total_loss_for_batch = torch.tensor(0.0, device=device, dtype=torch.float32) 
        actual_samples_processed_for_loss = 0        
        forget_mask = factors < 0.0
        if forget_mask.any():
            num_forget_samples = forget_mask.sum().item()
            
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
                forget_loss_val_mean, dpo_policy_outputs = compute_dpo_loss(
                    model=model,
                    ref_model=self.ref_model,
                    win_inputs=None,
                    lose_inputs=lose_inputs_forget,
                    beta=self.beta,
                )
                
                total_loss_for_batch += self.gamma * forget_loss_val_mean * num_forget_samples
                actual_samples_processed_for_loss += num_forget_samples

        retain_mask = factors > 0.0
        if retain_mask.any():
            num_retain_samples = retain_mask.sum().item()

            current_retain_inputs = {
                "input_ids":      inputs["answer_input_ids"][retain_mask],
                "attention_mask": inputs["answer_attention_mask"][retain_mask],
                "labels":         inputs["answer_labels"][retain_mask],
            }

            if current_retain_inputs["input_ids"].shape[0] > 0: 
                retain_outputs = model(**current_retain_inputs) 
                retain_loss_val_mean = retain_outputs.loss 
                
                total_loss_for_batch += self.alpha * retain_loss_val_mean * num_retain_samples
                actual_samples_processed_for_loss += num_retain_samples


        if actual_samples_processed_for_loss > 0:
            final_loss = total_loss_for_batch / actual_samples_processed_for_loss
        else:

            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
 #       if 'original_index' in inputs:
 #          current_original_indices = inputs["original_index"].tolist()
 #          factors_list = factors.tolist()
 #          print_info = [f"Idx:{idx}(F:{f:.0f})" for idx, f in zip(current_original_indices, factors_list)]
 #          print(f"SingleGPU Step {self.state.global_step if self.state else -1} BatchOrigIndices: {print_info} -> FinalLoss: {final_loss.item():.4f}")


        policy_outputs_dict = {}
        return (final_loss, policy_outputs_dict) if return_outputs else final_loss
