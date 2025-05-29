import torch
import math
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch.nn.functional as F
from scipy.stats import hmean
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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



@torch.no_grad()
def get_probs_ppl(question : str,
                  answer : str,
                  model : PreTrainedModel,
                  tokenizer : PreTrainedTokenizer,
                  device):
    """
    given a question and its gt answer, this function will return 
    p_mean = (1/T) * sum_{t=1}^T p(y_t|x, y_{<t})
    perplexity (ppl) = exp(-(1/T) * sum_{t=1}^T log p(y_t | x, y_{<t}))
    """
    full_text = question + answer

    full_enc = tokenizer(full_text, return_tensors='pt', add_special_tokens=False).to(device)
    questions_enc = tokenizer(question, return_tensors='pt', add_special_tokens=False).to(device)

    full_ids = full_enc['input_ids']
    questions_ids = questions_enc['input_ids']

    question_len = questions_ids.size(1)

    outputs = model(input_ids =full_ids, attention_mask = full_enc['attention_mask'])
    logits = outputs.logits
    
    answer_logits = logits[0, question_len: -1, :]
    answer_ids = full_ids[0, question_len+1:,]

    log_probs = F.log_softmax(answer_logits, dim = -1)

    selected = log_probs[torch.arange(log_probs.size(0)), answer_ids]
    probs = selected.exp()

    T = probs.size(0)
    p_mean = probs.mean().item()
    avg_nll = -1.0 * selected.mean().item()
    preplexity = math.exp(avg_nll)

    return p_mean, preplexity



@torch.no_grad()
def generate_outputs(question :str, model, tokenizer, device, max_new_tokens: int = 50):
    inputs = tokenizer(
        question, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample = False,
        return_dict_in_generate = False
    )
    full_seq = out[0]
    input_ids = inputs['input_ids']
    gen_ids = full_seq[input_ids.size(1):]
    answer = tokenizer.decode(
        gen_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return answer



def compute_forget_efficacy(forget, model, tokenizer, retriever_model, device):
    forget['gen_answer'] = ''
    forget['probs'] = ''
    forget['rouge_l'] = ''
    forget['cos_sim'] = ''
    forget['ppl'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []
    ppls = []
    for i, row in forget.iterrows():
        question = row['question']
        answer = row['answer']
        probs, ppl = get_probs_ppl(question, answer, model, tokenizer, device=device)
        gen_answer = generate_outputs(question, model, tokenizer, device=device)
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        forget.loc[i, 'gen_answer'] = gen_answer
        forget.loc[i, 'probs'] = probs
        forget.loc[i, 'rouge_l'] = rougel
        forget.loc[i, 'cos_sim'] = cosine_sim
        forget.loc[i, 'ppl'] = ppl
        probas.append(probs)
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)
        ppls.append(ppl)
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    forget_efficacy = 1.0 - np.mean(all_scores)
    print('forget_efficacy scores:',forget_efficacy)
    return forget, all_scores, forget_efficacy, np.mean(ppls)



def compute_model_utility_retain(retain, model, tokenizer, retriever_model, device):
    retain['gen_answer'] = ''
    retain['probs'] = ''
    retain['rouge_l'] = ''
    retain['cos_sim'] = ''
    retain['ppl'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []
    ppls = []
    for i, row in retain.iterrows():
        question = row['question']
        answer = row['answer']
        probs, ppl = get_probs_ppl(question, answer, model, tokenizer, device=device)
        gen_answer = generate_outputs(question, model, tokenizer, device=device)
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        retain.loc[i, 'gen_answer'] = gen_answer
        retain.loc[i, 'probs'] = probs
        retain.loc[i, 'rouge_l'] = rougel
        retain.loc[i, 'cos_sim'] = cosine_sim
        retain.loc[i, 'ppl'] = ppl
        probas.append(probs)
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)
        ppls.append(ppl)
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    model_utility_retain = hmean(all_scores)
    print('Model Utility scores:',model_utility_retain)
    
    return retain, all_scores, model_utility_retain, np.mean(ppls)



def compute_model_utility_test(test, model, tokenizer, retriever_model, device):
    test['gen_answer'] = ''
    test['probs'] = ''
    test['rouge_l'] = ''
    test['cos_sim'] = ''
    test['ppl'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []
    ppls = []

    for i, row in test.iterrows():
        question = row['question']
        answer = row['answer']
        probs, ppl = get_probs_ppl(question, answer, model, tokenizer, device=device)
        gen_answer = generate_outputs(question, model, tokenizer, device=device)
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        test.loc[i, 'gen_answer'] = gen_answer
        test.loc[i, 'probs'] = probs
        test.loc[i, 'rouge_l'] = rougel
        test.loc[i, 'cos_sim'] = cosine_sim
        test.loc[i, 'ppl'] = ppl
        probas.append(probs)
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)
        ppls.append(ppl)
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    
    model_utility_test = hmean(all_scores)
    print('Model Utility scores:',model_utility_test)
    
    return test, all_scores, model_utility_test, np.mean(ppls)





