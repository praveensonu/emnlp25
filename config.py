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
        self.access_token   = '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'
        self.forget_path    = '/home/praveen/theoden/emnlp_25/dataset/forget_dob.csv'
        self.retain_path    = '/home/praveen/theoden/emnlp_25/dataset/retain_dob.csv'
        self.results_path   = f'/home/praveen/theoden/emnlp_25/results/mcq_{self.exp_type}_results.json'