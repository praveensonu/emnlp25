class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.loss_type      = 'grad_diff' #or 'grad_ascent' or 'grad_diff'
        self.model_id       = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.LoRA_r         = 8
        self.LoRA_alpha     = 32
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-5
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj']
        self.batch_size     = 16
        self.gradient_accumulation_steps = 1
        self.num_epochs     = 20
        self.overwrite_dir  = True
        self.weight_decay   = 0.01 
        self.exp_type       = 'dob'
        self.save_dir       = f'outputs/{self.loss_type}_{self.exp_type}_model'
        self.access_token   = ''
        self.forget_path    = '/home/praveen/theoden/emnlp_25/dataset/forget_dob.csv'
        self.retain_path    = '/home/praveen/theoden/emnlp_25/dataset/retain_dob.csv'
        self.results_path   = f'/home/praveen/theoden/emnlp_25/results/mcq_{self.exp_type}_results.json'


class Config_ft:
    def __init__(self):
        super(Config_ft, self).__init__()
        self.model_id       = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.LoRA_r         = 32
        self.LoRA_alpha     = 64
        self.LoRA_dropout   = 0.1
        self.lr             = 1e-5
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj']
        self.batch_size     = 32
        self.gradient_accumulation_steps = 1
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01 
        self.save_dir       = f'outputs/reinforced_model'
        self.access_token   = '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'




class Config_eval:
    def __init__(self):
        super(Config_eval, self).__init__()
        self.loss_type      = 'grad_ascent' #or 'grad_ascent' or 'grad_diff'
        self.model_id       = '/home/praveen/theoden/emnlp_25/outputs/grad_ascent_dob_model' # 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.data_path      = '/home/praveen/theoden/emnlp_25/dataset/cloze_data.csv'
        self.LoRA_r         = 8
        self.LoRA_alpha     = 32
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-5
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj']
        self.batch_size     = 16
        self.gradient_accumulation_steps = 1
        self.num_epochs     = 20
        self.overwrite_dir  = True
        self.weight_decay   = 0.01 
        self.exp_type       = 'dob' #or 'dob' 'entity'
        self.save_dir       = f'outputs/{self.loss_type}_{self.exp_type}_model'
        self.access_token   = ''
        self.forget_path    = '/home/praveen/theoden/emnlp_25/dataset/forget_dob.csv'
        self.retain_path    = '/home/praveen/theoden/emnlp_25/dataset/retain_dob.csv'
        self.results_path   = f'/home/praveen/theoden/emnlp_25/results/cloze_{self.exp_type}_results.json'