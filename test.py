import os
import json
from config import Config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer   
cfg = Config()

# def update_json_dict(data_path, new_results):
#     # Check if the file exists; if not, start with an empty dictionary
#     if os.path.exists(data_path):
#         with open(data_path, 'r') as f:
#             try:
#                 data = json.load(f)
#             except json.JSONDecodeError:
#                 data = {}
#     else:
#         data = {}
    
#     # Expecting new_results to be a dictionary
#     if not isinstance(new_results, dict):
#         raise ValueError("new_results must be a dictionary when updating a dict-based JSON")
    
#     # Merge new_results into the existing data
#     data.update(new_results)
    
#     # Write the updated dictionary back to the file
#     with open(data_path, 'w') as f:
#         json.dump(data, f, indent=4)


# qa_perplexity_forget = torch.tensor(11.2437)
# results = {'grad_ascent': 
#            {'qa_perplexity_forget': qa_perplexity_forget.item(),
#            'exp_type': cfg.exp_type,
#            'model_id': cfg.model_id,
#            'batch_size': cfg.batch_size,
#            'num_epochs': cfg.num_epochs,
#            'lr': cfg.lr,
#            'weight_decay': cfg.weight_decay,
#            'LoRA_r': cfg.LoRA_r,
#            'LoRA_alpha': cfg.LoRA_alpha,}}


# update_json_dict(cfg.results_path, results)





