##### Mistral-7B-Instruct-v0.3 for 2024-07-31 09:40:34 #####
import os, itertools, time, argparse, random
import pandas, torch
from transformers import AutoTokenizer, set_seed
import transformers
import datasets

set_seed(23)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe_gen = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    add_special_tokens=True,
    #temperature=0.1,
)

# zero-shot prompt template: 2024-07-02 12:07:28
zero_shot_template = '''The tesk is to identify if the meanings of the target word \"{word}\" in the following c1 and c2 sentences correspond to {adj} meanings or not.
Target word: {word}
c1: {c1}
c2: {c2}
If the meanings correspond to {adj} meanings, Answer Yes. Otherwise, Answer No.
Only provide the answer, no explanations required.
The Answer (Yes or No) is:'''

# zero-shot anti-prompt template: 2024-07-02 12:07:22
zero_shot_anti_template = '''The task is to identify if the meanings of the target word \"{word}\" in the following c1 and c2 sentences correspond to {adj} meanings or not.
Target word: {word}
c1: {c1}
c2: {c2}
If the meanings correspond to {adj} meanings, Answer Yes. Otherwise, Answer No.
Only provide the answer, no explanations required.
The Answer (Yes or No) is:'''

class prompt_template:
    def __init__(self, name, template):
        self.name = name
        self.template_str = template
zero_shot_prompt = prompt_template('zero-shot-default', zero_shot_template)
zero_shot_anti_prompt = prompt_template('zero-shot-anti', zero_shot_anti_template)

### WiC dataset
def make_df(dataset='train'):
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'data.txt', encoding="utf-8") as f:
        data_df = pandas.read_csv(f, delimiter='\t', na_filter=None, names=['word', 'pos', 'index', 'c1', 'c2'])
    with open('./WiC_dataset/'+dataset+'/'+dataset+'.'+'gold.txt',  encoding="utf-8") as f:
        gold_df = pandas.read_csv(f, delimiter='\t', na_filter=None, header=None)
    data_df['label'] = gold_df
    return data_df

###
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

def save_summary_result(llm, adj, target_ds, accuracy, summary, run_id):
    fname = './{dir}/summ_{llm}_{adj}_{ds}_{id}.txt'.format(dir='zs_tsv', llm=llm, adj=adj, ds=target_ds, id=run_id)
    with open(fname, 'w') as f:
        f.write(summary + '\n')
        f.write('Accuracy: ' + str(accuracy) + '\n')

### save prediction results (not tinydb, but tsv: 2023-11-11 08:56:52)
def save_preds(llm, adj, target_ds, target_df, preds, run_id):
    words = target_df['word']; pos_list = target_df['pos']
    c1_list = target_df['c1']; c2_list = target_df['c2']
    golds = target_df['label']; ids = target_df['id']
    #
    fname = './{dir}/{llm}_{adj}_{ds}_{id}.tsv'.format(dir='zs_tsv', llm=llm, adj=adj, ds=target_ds, id=run_id)
    with open(fname, 'w') as f:
        header = '\t'.join(['id', 'w', 'p', 'l', 'c1', 'c2', 'pred']) + '\n'
        lines = [header]
        for id, word, pos, gold, c1, c2, pred in zip(ids, words, pos_list, golds, c1_list, c2_list, preds):
            line = '\t'.join([str(id), word, pos, gold, c1, c2, pred]) + '\n'
            lines.append(line)
        f.writelines(lines)

###
def main():
    #
    arg_p = argparse.ArgumentParser()
    arg_p.add_argument('--adj', type=str, default='identical', help='adjective to be included in the prompt')
    arg_p.add_argument('--llm', type=str, default='mistral-7B', help='mistral model')
    arg_p.add_argument('--template', type=str, default='default', help='prompt template')
    arg_p.add_argument('--dataset', type=str, default='test', help='train, dev, test')
    arg_p.add_argument('--run_id', type=int, default=23, help='seed')
    arg_p.add_argument('--verbose', type=str, default='True', help='verbose')
    arg_p.add_argument('--start', type=int, default=0, help='start instance') # 2024-06-27 10:42:06
    arg_p.add_argument('--end', type=int, default=0, help='end instance; 0 means till end') # 2024-06-27 10:42:09
    arg_p.add_argument('--save_summary', type=str, default='True', help='save results summary') 
    arg_p.add_argument('--save_tsv', type=str, default='True', help='save detailed results') 
    #
    args = arg_p.parse_args()
    if args.verbose: print(args)
    #
    if args.template=='anti':
        pr_template=zero_shot_anti_prompt
    else:
        pr_template=zero_shot_prompt
    #
    target_df = make_df(args.dataset)
    if args.end==0: args.end = len(target_df)
    target_df = target_df[args.start:args.end]
    target_df['id'] = range(args.start, args.end)
    #
    target_df = datasets.Dataset.from_pandas(target_df)
    #
    if args.verbose=='True':
        args.verbose = True
    else:
        args.verbose = False
    #
    pred_list, golds, target_df = wic_mistral(pipe=pipe_gen, adj=args.adj, 
                                              llm=args.llm, run_id=args.run_id, 
                                              target_ds=args.dataset, 
                                              target_df=target_df,
                                              s=args.start, e=args.end,
                                              pr_template=pr_template,
                                              temperature=0.0001, max_tokens=20,
                                              verbose=args.verbose)
    accuracy, summary = make_results_summary(pred_list, golds)
    #
    if args.verbose: print('Accuracy:', accuracy)
    if args.save_summary!='False':
        save_summary_result(args.llm, args.adj, args.dataset, accuracy, summary, run_id=args.run_id)
    if args.save_tsv!='False': save_preds(args.llm, args.adj, args.dataset, target_df, pred_list, run_id=args.run_id)
    #
    return pred_list

###
from tqdm import tqdm
def wic_mistral(pipe=pipe_gen, adj='same', 
                llm='mistral-7B', run_id=23,
                target_ds='dev', 
                #target_df=make_df('dev'),
                target_df='',
                s=0, e=20,
                pr_template=zero_shot_prompt, temperature=0.0001, max_tokens=20, verbose=True):
    #
    if target_df=='': target_df = make_df(target_ds)
    if e==0: e = len(target_df)
    words = target_df['word'][s:e]; pos_list = target_df['pos'][s:e]
    c1_list = target_df['c1'][s:e]; c2_list = target_df['c2'][s:e]
    golds = target_df['label'][s:e]; 
    pr_template_body = pr_template.template_str
    if verbose:
        print('Prompt template:', pr_template_body)
    #
    queries = []
    for i, (word, pos, c1, c2) in enumerate(zip(words, pos_list, c1_list, c2_list)):
        query = pr_template_body.format(adj=adj, word=word, c1=c1, c2=c2)
        queries.append(query)
    #
    if verbose:
        completions = []
        for i in tqdm(range(len(queries))):
            completions.append(pipe(
                queries[i],
                do_sample=True,
                top_k=1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                ))
    else:
        completions = pipe(
            queries,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    #
    pred_list = parse_mistral_results(completions, pr_template.name)
    golds = list(golds)
    if verbose:
        print('Accuracy:', comp_acc(pred_list, golds))
    return pred_list, golds, target_df

def parse_mistral_results(completions, pr_name, anti=False):
    if pr_name=='zero-shot-anti': anti=True
    def parse_response(gen_text):
        lines = gen_text.split('\n')
        answered = False
        for l in lines:
            if l.startswith('The Answer (Yes or No) is: Yes'):
                if anti: result = 'F'
                else: result = 'T'
                answered = True
                break
            elif l.startswith('The Answer (Yes or No) is: No'):
                if anti: result = 'T'
                else: result = 'F'
                answered = True
                break
        if not answered:
            result = '?'
        return result
    return [parse_response(compl[0]['generated_text']) for compl in completions]

def comp_acc(x, y):
    c = 0
    for i in range(len(x)):
        if x[i]==y[i]: c += 1
    return c/len(x)

def test(s, e=0, pipe=pipe_gen, adj='same', ds='dev', pr=zero_shot_prompt, max_tokens=10):
    ds = make_df(ds)
    if e==0: e = len(ds)
    X = wic_mistral(pipe, adj, ds, s=s, e=e, pr_template=pr, max_tokens=max_tokens)
    print(X[0])
    for i, x in enumerate(X[0]):
        #res = x[0]['generated_text']
        print()
        print(x)
        print()
        res = x[0]['generated_text']
        answered = False
        for r in res.split('\n'):
            if r.startswith('The Answer (Yes or No) is:'):
                print(i, '>', r)
                answered = True
        if not answered:
            print(i, '?', r)
    return X

###
if __name__ == '__main__':
    main()
