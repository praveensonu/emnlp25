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
                                    question: str, 
                                    answer: str,
                                    template_format=None) -> torch.Tensor:
    """
    Tokenizes question answer pair and returns input_ids, labels, and attention_mask into SFT format.
    
    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize the input.
        max_length (int): Maximum sequence length. This includes max_new_tokens + token length of question.
        question (str): Question to be tokenized.
        answer (str): Answer to be tokenized.
        template_format (str, optional): Custom template format. If None, will use the tokenizer's chat template.
    
    Returns:
        torch.Tensor: Each input_ids, labels, and attention_mask in their own tensor.
    """
    # Format the question using either custom template or chat template
    if template_format:
        new_question = template_format.format(instruction=question)
    else:
        # Use the tokenizer's built-in chat template
        messages = [{"role": "user", "content": question}]
        new_question = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    full_text = new_question + answer
    
    # Get the number of tokens in the question part
    prompt_inputs = tokenizer(new_question, return_tensors="pt")
    num_question_tokens = prompt_inputs["input_ids"].size(1)
    
    # Tokenize the full text
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    
    # Padding logic
    pad_length = max_length - len(encoded["input_ids"])
    
    # Use the tokenizer's pad token instead of hardcoded values if available
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Create padded input_ids
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] + [pad_token_id] * (pad_length - 1) if pad_length > 0 else encoded['input_ids']
    
    # Create padded attention mask
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length if pad_length > 0 else encoded['attention_mask']
    
    # Create labels, masking the prompt tokens
    if len(encoded['input_ids']) == max_length:
        label = encoded['input_ids'].copy()
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
    
    # Mask prompt tokens in labels
    for i in range(num_question_tokens):
        label[i] = -100
    
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


## only for tofu
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
    T = (labels[0] != -100).sum().float()  # convert count to float
    loss_avg = loss / T  # average loss per answer token
    
    # Compute conditional probability p(y|x) = exp(-T * loss_avg)
    p_y_given_x = torch.exp(-T * loss_avg)
    
    return p_y_given_x, T.item(), loss_avg.item()


def calculate_cond_prob(prompt, answer, tokenizer, model, device):
    """
    Calculate conditional probability of the answer given the prompt.
    """
    # Format the prompt using the chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Create the full text with the formatted prompt and answer
    full_text = formatted_prompt + answer + tokenizer.eos_token
    
    # Tokenize the full text
    encoded = tokenizer(full_text, return_tensors='pt', add_special_tokens=True).to(device)
    full_input_ids = encoded['input_ids']
    
    # Tokenize just the prompt to get its length
    prompt_encoded = tokenizer(formatted_prompt, return_tensors='pt', add_special_tokens=True).to(device)
    prompt_len = prompt_encoded['input_ids'].size(1)
    
    # Create labels, masking the prompt tokens
    labels = full_input_ids.clone()
    labels[0, :prompt_len] = -100
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(full_input_ids)
    
    # Calculate probabilities
    p_y_given_x, T, loss_avg = get_probs(outputs, labels)
    
    return p_y_given_x



def generate_outputs(text, model, tokenizer, device):
    """
    Generate model outputs for the given text using the chat template.
    """
    # Format the input using the chat template
    messages = [{"role": "user", "content": text}]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted input
    inputs = tokenizer(formatted_input, return_tensors="pt", add_special_tokens=True).to(device)
    
    # Generate outputs
    outputs = model.generate(**inputs, max_new_tokens = 50)
    
    # Decode the outputs
    full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Extract the answer part
    # This assumes the assistant's response comes after the formatted prompt
    # The exact split might need adjustment based on the specific chat template used
    assistant_prefix = "assistant\n\n"
    if assistant_prefix in full_output:
        answer = full_output.split(assistant_prefix)[-1]
    else:
        # If the assistant prefix isn't found, try to extract the answer portion
        # based on the length of the formatted input
        prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if full_output.startswith(prompt_text):
            answer = full_output[len(prompt_text):].strip()
        else:
            answer = full_output  # Fallback to returning the full output
    
    return answer

# def generate_outputs(text, model, tokenizer, device):
#     inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
#     outputs = model.generate(**inputs)
#     full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     answer = full_output.split("assistant\n\n")[-1]
#     return answer



def compute_forget_efficacy(forget_path, model, tokenizer, retriever_model, device):
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
    forget['probs'] = ''
    forget['rouge_l'] = ''
    forget['cos_sim'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in forget.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Format prompt using a global template (assumed defined elsewhere)
        #prompt = template.format(instruction=question)
        prompt = question
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, answer, tokenizer, model, device) #or gen_answer?

        # Update DataFrame and store metric scores
        forget.loc[i, 'gen_answer'] = gen_answer
        forget.loc[i, 'probs'] = prob.item()
        forget.loc[i, 'rouge_l'] = rougel
        forget.loc[i, 'cos_sim'] = cosine_sim

        probas.append(prob.item())
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    
    forget_efficacy = 1.0 - np.mean(all_scores)
    print('forget_efficacy scores:',forget_efficacy)
    return forget, all_scores, forget_efficacy


def compute_model_utility_retain(retain_path, model, tokenizer, retriever_model, device):
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
    retain['gen_answer'] = ''
    retain['probs'] = ''
    retain['rouge_l'] = ''
    retain['cos_sim'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in retain.iterrows():
        question = row['question']
        answer = row['answer']
        
        #prompt = template.format(instruction=question)
        prompt = question
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, answer, tokenizer, model, device)  #or gen_answer?

        retain.loc[i, 'gen_answer'] = gen_answer
        retain.loc[i, 'probs'] = prob.item()
        retain.loc[i, 'rouge_l'] = rougel
        retain.loc[i, 'cos_sim'] = cosine_sim
        probas.append(prob.item())
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    
    model_utility_retain = hmean(all_scores)
    #model_utility = np.mean(all_scores)
    print('Model Utility scores:',model_utility_retain)
    
    return retain, all_scores, model_utility_retain



def compute_model_utility_test(test_path, model, tokenizer, retriever_model, device):
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
    
    test = pd.read_csv(test_path)
    test['gen_answer'] = ''
    test['probs'] = ''
    test['rouge_l'] = ''
    test['cos_sim'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in test.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Format prompt using a global template (assumed defined elsewhere)
        #prompt = template.format(instruction=question)
        prompt = question
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, answer, tokenizer, model, device)  #or gen_answer?

        # Update DataFrame and store metric scores
        test.loc[i, 'gen_answer'] = gen_answer
        test.loc[i, 'probs'] = prob.item()
        test.loc[i, 'rouge_l'] = rougel
        test.loc[i, 'cos_sim'] = cosine_sim
        probas.append(prob.item())
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    
    model_utility_test = hmean(all_scores)
    #model_utility = np.mean(all_scores)
    print('Model Utility scores:',model_utility_test)
    
    return test, all_scores, model_utility_test


def calculate_conditional_perplexity(question, model, tokenizer, device, max_new_tokens=100, sys_text=""):
    """
    Generate an answer to a question and calculate its conditional perplexity.
    
    This calculates the conditional perplexity of the answer given the question prompt.
    
    Args:
        question (str): The user question
        model: The language model
        tokenizer: The tokenizer for the model
        device: The device to run the model on
        max_new_tokens (int): Maximum tokens to generate
        sys_text (str): Optional system prompt
        
    Returns:
        float: The conditional perplexity value
        str: The generated answer
    """
    # Prepare the prompt with optional system text
    if sys_text:
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": question}
        ]
    else:
        messages = [{"role": "user", "content": question}]
    
    # Format input using chat template
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted input
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].size(1)
    
    # Generate the answer
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True
        )
    
    # Get the full sequence (prompt + generated answer)
    full_sequence = generation_output.sequences[0]
    
    # Extract just the generated part (without the prompt)
    generated_ids = full_sequence[input_length:]
    
    # Decode the answer
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Calculate perplexity on only the generated part
    # Create new input that includes both the prompt and generated answer
    full_text = formatted_input + answer
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    # Create labels, masking prompt tokens (-100 is the ignore index)
    labels = full_inputs["input_ids"].clone()
    labels[:, :input_length] = -100
    
    # Forward pass with labels to compute loss
    with torch.no_grad():
        outputs = model(full_inputs["input_ids"], labels=labels)
    
    # Get the loss value (negative log likelihood)
    nll = outputs.loss.item()
    
    # Calculate perplexity: exp(average negative log likelihood)
    perplexity = torch.exp(torch.tensor(nll)).item()
    
    return perplexity, answer


