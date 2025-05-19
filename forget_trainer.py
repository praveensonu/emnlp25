from transformers import Trainer
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from collators import custom_data_collator_interleaved




accelerator = Accelerator()
def get_batch_loss(output_logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output_logits_for_loss = output_logits[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output_logits_for_loss.transpose(-1,-2), shifted_labels).sum(dim=-1)

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


# using this for the gradient descent method applied in literature
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
    
    

def process_inputs_with_title(inputs):
    """
    Modifies the inputs dictionary by removing 'factor' and 'title_id'
    if they exist. Returns the potentially popped values.
    """
    factor_value = inputs.pop("factor", None)
    title_id_value = inputs.pop("title_id", None)
    return factor_value, title_id_value


def process_inputs_without_title(inputs_dict):
    """
    Helper to pop 'factor' and other non-model args from inputs before passing to model.
    """
    factor_value = inputs_dict.pop("factor", None)
    return factor_value #, title_id_value


# we use this for batch gradient ascent forget : retain ratio. 
class BatchGradDiffTrainer(Trainer):

    def __init__(self,model,
                 **hf_trainer_kwargs):
        super().__init__(model= model, **hf_trainer_kwargs)
        self.model = model
        self.model = self.accelerator.prepare_model(
                        self.model, 
                        evaluation_mode=False   
            )
        self.model.train() 

        if self.data_collator is None:
            self.data_collator = custom_data_collator_interleaved

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
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        """
        Computes the adjusted loss, scaling each sequence's loss by 'factor'.
        """
        factors = process_inputs_without_title(inputs) 
        

        if factors is None:
            if self.is_in_train:
                raise ValueError("Factors are missing from training inputs. Ensure your dataset and collator provide them.")
            else: 
                print("Warning: 'factor' not found in inputs during evaluation. Computing standard loss.")
                outputs = model(**inputs)
                if "loss" in outputs:
                     loss = outputs.loss
                else:
                    logits = outputs.logits
                    labels_for_loss = inputs["labels"]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels_for_loss[..., 1:].contiguous()
                    loss_fct_eval = CrossEntropyLoss()
                    loss = loss_fct_eval(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return (loss, outputs) if return_outputs else loss
            
        #current_global_step_being_accumulated_for = self.state.global_step if self.state else -1
        outputs = model(**inputs)
        logits = outputs.logits 
        labels = inputs['labels']
        sum_loss_per_seq = get_batch_loss(logits, labels)
        shift_labels_for_count = labels[..., 1:].contiguous()
        if logits.size(1) <= 1:
            adjusted_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        

        valid_token_counts = (shift_labels_for_count != -100).sum(dim=-1).float()
        valid_token_counts = torch.max(valid_token_counts, torch.tensor(1.0, device = valid_token_counts.device))

        mean_loss_per_sequence = sum_loss_per_seq / valid_token_counts

        scaled_losses = mean_loss_per_sequence * factors

        adjusted_loss = scaled_losses.mean()



        #log_prefix = (f"GlobalStepAccumFor: {current_global_step_being_accumulated_for} "
        #          f"Rank {self.accelerator.process_index}")
        #print(f"{log_prefix} - Adjusted Loss: {adjusted_loss.item()} - factors: {factors.tolist()} - ")

        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

