# WiC zero_shot_new (2025-01-11 08:16:11)
import pandas
import time, ray
import openai
#
import os, itertools

# zero-shot prompt template 2023-09-09 16:13:56
zero_shot_template = '''
Your task is to identify if the meanings of the target word \"{word}\" in the following c1 and c2 sentences correspond to {adj} meanings or not.
That is, it is the Word-in-Context task.

Please simply answer T, if the meanings correspond to {adj} meanings.
Otherwise, simply answer F.
[Question]
Target word: {word}
c1: {c1}
c2: {c2}
Answer: 
'''

# zero-shot anti-prompt template 2023-11-11 08:41:37
zero_shot_anti_template = '''
Your task is to identify if the meanings of the target word \"{word}\" in the following c1 and c2 sentences correspond to {adj} meanings or not.
That is, it is the Word-in-Context task.

Please simply answer F, if the meanings correspond to {adj} meanings.
Otherwise, simply answer T.
[Question]
Target word: {word}
c1: {c1}
c2: {c2}
Answer: 
'''

class prompt_template:
    def __init__(self, name, template):
        self.name = name
        self.template_str = template
zero_shot_prompt = prompt_template('zero-shot-default', zero_shot_template)
zero_shot_anti_prompt = prompt_template('zero-shot-anti', zero_shot_anti_template)

###
def eval_wic_fs(adj='identical', 
                target_dataset='test', s=0, e=0,
                pr_template=zero_shot_prompt, 
                llm='gpt-3.5-turbo-1106',
                run_id=0,
                save_tsv=True, save_summary=True, 
                temperature=0, max_tokens=384, nc=8, verbose=True):
    # globals
    global _adj, _target_dataset, _pr_template, _llm, _run_id
    _adj=adj; _target_dataset=target_dataset; _pr_template=pr_template
    _llm=llm; _run_id=run_id
    #
    if e==0: 
        target_df = make_df(target_dataset)[s:]
        e = len(target_df)
    else:
        target_df = make_df(target_dataset)[s:e]
    target_df['id'] = range(s, e)
    golds = target_df['label']
    #
    if verbose: print('>>> Target datasets:', target_dataset)
    pred_list_ = wic(adj, target_df, s, e, 
                     pr_template.template_str, 
                     temperature, max_tokens, nc, verbose)
    pred_list = [repair_answer(_) for _ in pred_list_]
    accuracy, summary = make_results_summary(pred_list, golds)
    #
    if verbose: print('Accuracy:', accuracy)
    if save_summary:
        save_summary_result(llm, adj, target_dataset, accuracy, summary, run_id=run_id)
    if save_tsv: save_preds(llm, adj, target_dataset, target_df, pred_list, run_id=run_id)
    #
    return pred_list, golds

import random
def repair_answer(x):
    if not x: 
        return random.sample(['T','F'], k=1)[0]
    if len(x)==1: return(x)
    if x[0]=='F':
        return 'F'
    elif x[0]=='T':
        return 'T'
    else:
        print('???', x)
        return 'F'

def wic(adj, target_df, s, e,
        pr_template, temperature, max_tokens, nc, verbose):
    words = target_df['word']; pos_list = target_df['pos']
    c1_list = target_df['c1']; c2_list = target_df['c2']
    golds = target_df['label']
    #
    ray.shutdown()
    ray.init(num_cpus=nc)
    #
    okay = ng = 0
    if verbose: print('\nStart WiC with zero-shot setting >', time.ctime())
    begin_t = time.time()
    pred_list = []
    for i, (word, pos, c1, c2) in enumerate(zip(words, pos_list, c1_list, c2_list)):
        print('\n---------------')
        for trial in range(10):
            wic_res = wic_(adj, s+i, word, pos, c1, c2, 
                           pr_template, target_df, 
                           temperature=temperature, max_tokens=max_tokens, nc=nc, verbose=verbose)
            if wic_res: 
                break
            else:
                print('>>> retrying wic_ >>>', trial+1)
        pred_list.append(wic_res)
        print('finished: i, pred, gold:', s+i, wic_res, golds[s+i])
        if wic_res==golds[s+i]: okay += 1
        else: ng += 1
        print('Acc so far:', okay/(okay+ng))
    #
    end_t = time.time()
    if verbose:
        print('>>> All finished! Total elapsed (sec):', end_t-begin_t)
        print(time.ctime())
        print('\n')
    #
    return pred_list

#
def wic_(adj, target_id, word, pos, c1, c2, 
         pr_template_body, target_df, 
         temperature, max_tokens, nc, verbose,
         timeout=3, trial_limit=3):
    time.sleep(1)
    begin_t = time.time()
    #
    query = pr_template_body.format(adj=adj, word=word, c1=c1, c2=c2)
    if verbose: print(query)
    #
    r_ = rgpt.remote(query, temperature, max_tokens)
    #
    cnt = 1
    _finished, _not_finished = ray.wait([r_], num_returns=1, timeout=timeout)
    while _not_finished and (cnt < trial_limit):
        cnt += 1
        print('>>> wic_: incomplete', cnt)
        print('    ... finished items:', len(_finished), 1)
        print('    ... elapsed time:', time.time()-begin_t)
        time.sleep(2)        
        _finished, _not_finished = ray.wait([r_], num_returns=1, timeout=timeout)
    #
    if cnt >= trial_limit:
        print('<<< wic_ trial count exceedsd <<<')
        return []
    #
    try:
        results = ray.get(_finished)
    except openai.error.RateLimitError as err:
        print('<<< RateLimitError <<< sleep for: 10 seconds')
        time.sleep(10)
        return []
    except openai.error.APIError as err:
        print('<<< APIError <<< sleep for: 10 seconds')
        time.sleep(10)
        return []
    #
    if len(results)!=1:
        print('# of not finished calls:', 1-len(results))
    #
    end_t = time.time()
    print('Elapsed (sec):', end_t-begin_t)
    #
    pred_list = [c["choices"][0]["message"]["content"] for c in results]
    #
    if not pred_list:
        return []
    #
    return pred_list[0]

### WiC dataset
def make_df(dataset='train'):
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'data.txt', encoding="utf-8") as f:
        data_df = pandas.read_csv(f, delimiter='\t', na_filter=None, names=['word', 'pos', 'index', 'c1', 'c2'])
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'gold.txt',  encoding="utf-8") as f:
        gold_df = pandas.read_csv(f, delimiter='\t', na_filter=None, header=None)
    data_df['label'] = gold_df
    return data_df

### results summary
from sklearn.metrics import confusion_matrix, classification_report
def make_results_summary(predicted_list, gold_list):
    conf = str(confusion_matrix(gold_list, predicted_list))
    rep = classification_report(gold_list, predicted_list)
    summ = conf + '\n' + rep
    #
    corr = 0
    for p, g in zip(predicted_list, gold_list):
        if p[:2] == g[:2]: corr += 1
    return corr/len(predicted_list), summ

def save_summary_result(llm, adj, target_dataset, accuracy, summary, run_id):
    fname = './{dir}/summ_{llm}_{adj}_{ds}_{id}.txt'.format(dir='zs_tsv', llm=llm, adj=adj, ds=target_dataset, id=run_id)
    with open(fname, 'w') as f:
        f.write(summary + '\n')
        f.write('Accuracy: ' + str(accuracy) + '\n')

### save prediction results (2023-11-11 08:56:52)
def save_preds(llm, adj, target_dataset, target_df, preds, run_id):
    words = target_df['word']; pos_list = target_df['pos']
    c1_list = target_df['c1']; c2_list = target_df['c2']
    golds = target_df['label']; ids = target_df['id']
    #
    fname = './{dir}/{llm}_{adj}_{ds}_{id}.tsv'.format(dir='zs_tsv', llm=llm, adj=adj, ds=target_dataset, id=run_id)
    with open(fname, 'w') as f:
        header = '\t'.join(['id', 'w', 'p', 'l', 'c1', 'c2', 'pred']) + '\n'
        lines = [header]
        for id, word, pos, gold, c1, c2, pred in zip(ids, words, pos_list, golds, c1_list, c2_list, preds):
            line = '\t'.join([str(id), word, pos, gold, c1, c2, pred]) + '\n'
            lines.append(line)
        f.writelines(lines)
    #

### OpenAI API with ray support
@ray.remote
def rgpt(query, temperature, max_tokens, trial_limit=5, first_wait_time=10):
    global _llm; global _run_id
    for i in range(trial_limit):
        try:
            api_res = openai.ChatCompletion.create(
                model = _llm,
                messages = [{'role':'user', 'content':query}],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=_run_id,
                #logprobs=1,
            )
            return api_res
        except openai.error.ServiceUnavailableError as err:
            if i==trial_limit - 1:
                raise
            print(f"Error: {err}")
            wait_time_seconds = first_wait_time * (2**i)
            print(f"Waiting for {wait_time_seconds} secs.")
            time.sleep(wait_time_seconds)

#####
import argparse, time

def main():
    global wic_with_bert_model
    #
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument('--adj', type=str, default='identical', help='adjective to be included in the prompt')
    arg_p.add_argument('--llm', type=str, default='gpt-3.5-turbo-1106', help='gpt model')
    arg_p.add_argument('--template', type=str, default='default', help='prompt template')
    arg_p.add_argument('--dataset', type=str, default='test', help='train, dev, test')
    arg_p.add_argument('--run_id', type=int, default=23, help='seed')
    arg_p.add_argument('--verbose', type=str, default='False', help='verbose')
    arg_p.add_argument('--start', type=int, default=0, help='start instance') # 2024-06-27 10:42:06
    arg_p.add_argument('--end', type=int, default=0, help='end instance; 0 means till end') # 2024-06-27 10:42:09

    args = arg_p.parse_args()
    #
    if args.verbose=='True':
        verbose_ = True
    else:
        verbose_ = False
    if args.template=='anti':
        X = eval_wic_fs(adj=args.adj, llm=args.llm, run_id=args.run_id, target_dataset=args.dataset, pr_template=zero_shot_anti_prompt, 
                        s=args.start, e=args.end,  # 2024-06-27 10:41:53
                        verbose=verbose_)
    else:
        X = eval_wic_fs(adj=args.adj, llm=args.llm, run_id=args.run_id, target_dataset=args.dataset, 
                        s=args.start, e=args.end, # added 2024-06-27 10:41:12
                        verbose=verbose_)
    time.sleep(1)
    return X

#
if __name__ == '__main__':
    main()
