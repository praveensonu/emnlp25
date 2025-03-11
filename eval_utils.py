import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm
import pandas as pd
from torch import nn
from scipy.stats import ks_2samp, hmean
from typing import Generator, Tuple, Union
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def convert_raw_data_to_model_format(tokenizer: PreTrainedTokenizer, 
                                     max_length: int, 
                                     question : str, 
                                     answer : str,
                                     template_format = None) -> torch.Tensor:
    
    """
    Tokenizes question answer pair and returns input_ids, labels, and attention_mask into SFT format.
    
    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize the input.
        max_length (int): Maximum sequence length. This includes max_new_tokens + token length of question.
        question (str): Question to be tokenized.
        answer (str): Answer to be tokenized.
        question_start_token (str): Start token for question.
        question_end_token (str): End token for question.
        answer_token (str): Start token for answer.
    
    Returns:
        torch.Tensor: Each input_ids, labels, and attention_mask in their own tensor.
    """
    if template_format:
        new_question = template_format.format(instruction=question)
    else:
        new_question = f"Question: {question}\nAnswer:"

     
    
    full_text = new_question + answer

    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded["input_ids"])
    pad_input_ids = encoded['input_ids'] + [128009] + [128004] * (pad_length -1)
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded['input_ids']) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
    
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset

class TextDatasetQA(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 model, 
                 max_length=500, 
                 split = None, 
                 question_key='question', 
                 answer_key='answer', 
                 template_format = None,): 
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = pd.read_csv(data_path)

        self.data = add_dataset_index(self.data)
        #self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key
        self.template_format = template_format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, 
                                                              question, 
                                                              answer, 
                                                              self.template_format)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def create_dataloader(dataset, tokenizer, model, answer_key, batch_size=32, max_length=200):
    """
    Args:
        dataset: The dataset to use.
        tokenizer: Tokenizer instance.
        model: Model instance.
        answer_key (str): Key for the answer type (e.g., 'paraphrased_answer' or 'perturbed_answer').
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 30//4.
        max_length (int, optional): Maximum token length. Defaults to 200.

    Returns:
        DataLoader: The DataLoader for the given dataset.
    """
    text_dataset = TextDatasetQA(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        question_key='question',
        answer_key=answer_key,
        question_start_token="Question: ",
        question_end_token="\n",
        answer_token="Answer: "
    )
    
    return DataLoader(text_dataset, batch_size=batch_size, collate_fn=custom_data_collator_with_indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss


def get_eval_logs(para_data, perturbed_data, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(para_data, perturbed_data)):
        input_ids, labels, attention_mask, indices = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2] #shape of [7,5,200] => [7,5]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1

        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), 
                         "labels": perturb_labels.view(bsz*seq_len, -1), 
                         "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)} #shape of [7,5,200] => [35,200] (7*5 = 35) basically flattening it
        
        #to device
        for k,v in batch.items():
            batch[k] = v.to(model.device)
        for k,v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)
        
        #compute log probabilities (cross entropy loss)
        para_loss = get_batch_loss(outputs.logits, batch["labels"])
        perturb_loss = get_batch_loss(perturb_outputs.logits, 
                                      perturb_batch["labels"]).view(bsz, seq_len)

        #compute number of valid tokens (excluding padding/masked ones)
        num_token_para = (batch["labels"] != -100).sum(dim=-1)
        num_token_perturb = (perturb_batch["labels"] != -100).view(bsz, seq_len, -1).sum(-1)

        #compute per-token loss
        para_loss_per_token = para_loss / num_token_para
        perturb_loss_per_token = perturb_loss / num_token_perturb

        #zip index and each stat into the dict
        para_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), para_loss_per_token.cpu().numpy().tolist()))
        perturb_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token.cpu().numpy().tolist()))

        # store in the dict
        if 'average_para_loss' not in eval_logs:
            eval_logs['average_para_loss'] = {}
        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}

        eval_logs['average_para_loss'].update(para_loss_per_token)
        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)

    return eval_logs


def cal_truth_ratio(unlearn_logs):

    unlearn_para_npvalues = np.array(list(unlearn_logs['average_para_loss'].values()))
    unlearn_pert_npvalues = np.array(list(unlearn_logs['average_perturb_loss'].values()))
    unlearn_pert_npvalues = unlearn_pert_npvalues.mean(axis = -1)

    unlearn_truth_ratio = np.exp(unlearn_pert_npvalues - unlearn_para_npvalues)

    return unlearn_truth_ratio



def cal_forget_quality(unlearn_logs, retain_logs):

    unlearn_para_npvalues = np.array(list(unlearn_logs['average_para_loss'].values()))
    unlearn_pert_npvalues = np.array(list(unlearn_logs['average_perturb_loss'].values()))
    unlearn_pert_npvalues = unlearn_pert_npvalues.mean(axis = -1)


    retain_para_npvalues = np.array(list(retain_logs['average_para_loss'].values()))
    retain_pert_npvalues = np.array(list(retain_logs['average_perturb_loss'].values()))
    retain_pert_npvalues = retain_pert_npvalues.mean(axis = -1)

    unlearn_truth_ratio = np.exp(unlearn_pert_npvalues - unlearn_para_npvalues)
    retain_truth_ratio = np.exp(retain_pert_npvalues - retain_para_npvalues)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    
    return {'forget_quality': test_res.pvalue,
            'KS Test Pval Forget': test_res.pvalue,
            'KS Test Forget': test_res.statistic}



def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gen_outputs, ground_truths)

    return rouge_scores['rouge1'].recall, rouge_scores['rougeL'].recall


def eval_cosine_similarity(gen_outputs, ground_truths, model_name, device):
    model = SentenceTransformer(model_name, device=device)
    with torch.no_grad():
        gen_embedding = model.encode(gen_outputs, show_progress_bar=False)
        gt_embedding = model.encode(ground_truths, show_progress_bar=False)
        cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
        similarity = cosine_sim.item()
    return max(0, similarity)


def get_probs(outputs, labels):
    loss = get_batch_loss(outputs.logits, labels) # we use batch loss, which return sum loss of the sequence
    ## since get batch_loss give sum loss for each sequence in a batch
    cond_probs = torch.exp(-loss)
    return cond_probs


def calculate_cond_prob(prompt, answer, tokenizer, model, device):
    full_text = prompt + " " + answer + tokenizer.eos_token
    encoded = tokenizer(full_text, return_tensors='pt', add_special_tokens=True).to(device)
    full_input_ids = encoded['input_ids']
    prompt_encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).to(device)
    prompt_len = prompt_encoded['input_ids'].size(1)
    labels = full_input_ids.clone()
    labels[0, :prompt_len] = -100
    with torch.no_grad():
        outputs = model(full_input_ids)
    cond_probs = get_probs(outputs, labels)
    probs = cond_probs.to('cpu').float().numpy()
    return probs.item()



def generate_outputs(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    outputs = model.generate(**inputs)
    full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = full_output.split("assistant\n\n")[-1]
    return answer


def compute_forget_efficacy(forget_path, model, tokenizer, retriever_model, device, template):
    """
    Evaluate the forget efficacy by generating answers for each row in the provided DataFrame,
    computing evaluation metrics, and updating the DataFrame with generated answers.

    Parameters:
        forget (pd.DataFrame): DataFrame containing at least 'question' and 'answer' columns.
        model: The model used for generating answers.
        tokenizer: The tokenizer corresponding to the model.
        cfg: Configuration object that must include 'retriever_model' for cosine similarity evaluation.
        device: The device to run model computations on (e.g., "cpu" or "cuda").

    Returns:
        tuple: A tuple containing:
            - forget (pd.DataFrame): The updated DataFrame with a new 'gen_answer' column.
            - forget_efficacy (float): The computed forget efficacy score.
    """
    # Initialize the 'gen_answer' column and lists for evaluation metrics
    forget = pd.read_csv(forget_path)
    forget['gen_answer'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in forget.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Format prompt using a global template (assumed defined elsewhere)
        prompt = template.format(instruction=question)
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, gen_answer, tokenizer, model, device)

        # Update DataFrame and store metric scores
        forget.loc[i, 'gen_answer'] = gen_answer
        probas.append(prob)
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    print(all_scores)
    forget_efficacy = 1.0 - np.mean(all_scores)
    
    return forget, forget_efficacy



def compute_model_utility(retain_path, model, tokenizer, retriever_model, device, template):
    """
    Evaluate the forget efficacy by generating answers for each row in the provided DataFrame,
    computing evaluation metrics, and updating the DataFrame with generated answers.

    Parameters:
        forget (pd.DataFrame): DataFrame containing at least 'question' and 'answer' columns.
        model: The model used for generating answers.
        tokenizer: The tokenizer corresponding to the model.
        cfg: Configuration object that must include 'retriever_model' for cosine similarity evaluation.
        device: The device to run model computations on (e.g., "cpu" or "cuda").

    Returns:
        tuple: A tuple containing:
            - forget (pd.DataFrame): The updated DataFrame with a new 'gen_answer' column.
            - forget_efficacy (float): The computed forget efficacy score.
    """
    retain = pd.read_csv(retain_path)
    # Initialize the 'gen_answer' column and lists for evaluation metrics
    retain['gen_answer'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in retain.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Format prompt using a global template (assumed defined elsewhere)
        prompt = template.format(instruction=question)
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, gen_answer, tokenizer, model, device)

        # Update DataFrame and store metric scores
        retain.loc[i, 'gen_answer'] = gen_answer
        probas.append(prob)
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    model_utility = hmean(all_scores)
    
    return retain, model_utility


