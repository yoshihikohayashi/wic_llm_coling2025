# 2025-01-11 08:47:10
import numpy as np
from io import StringIO

def get_zs_tsv_dir(llm):
    return './zs_tsv/'

def summ_file_name(llm, adj, data_split, run_id='323'):
    fn = '_'.join(['summ', llm, adj, data_split, run_id]) + '.txt'
    return fn

def get_conf_mat(llm, adj, data_split):
    zs_tsv_dir = get_zs_tsv_dir(llm)
    with open(zs_tsv_dir + summ_file_name(llm, adj, data_split), 'r') as f:
        lines = f.readlines()
    x = lines[0] + lines[1]
    d = StringIO(x.replace('[', '').replace(']', ''))
    return np.loadtxt(d, dtype=int)

def p_r_f1(llm, adj, ds, delm=' & '):
    conf = get_conf_mat(llm, adj, ds)
    f_r = conf[0,0] / (conf[0,0]+conf[0,1])
    t_r = conf[1,1] / (conf[1,0]+conf[1,1])
    f_p = conf[0,0] / (conf[0,0]+conf[1,0])
    t_p = conf[1,1] / (conf[0,1]+conf[1,1])

    acc = (conf[0,0]+conf[1,1]) / (conf[0,0]+conf[0,1]+conf[1,0]+conf[1,1])
    
    f_f1 = 2*f_r*f_p / (f_r+f_p)
    t_f1 = 2*t_r*t_p / (t_r+t_p)
    
    f_p, f_r, f_f1 = "{:.3f}".format(f_p), "{:.3f}".format(f_r), "{:.3f}".format(f_f1)
    t_p, t_r, t_f1 = "{:.3f}".format(t_p), "{:.3f}".format(t_r), "{:.3f}".format(t_f1)
    acc = "{:.3f}".format(acc)
    
    line = delm.join([adj, f_p, f_r, f_f1, t_p, t_r, t_f1, acc])
    return line

positives = ['identical', 'the same', 'similar', 'related']
negatives = ['distinct', 'different', 'dissimilar', 'unrelated']
llms = ['gpt-3.5-turbo-1106', 'gpt-4o-2024-05-13', 'llama3-8B', 'mistral-7B']

def comp_for_adjs(llm, adjs=positives+negatives,
                  ds='test', delm=' & ', verbose=True):
    lines = []
    if verbose:
        print('LLM:', llm)
        print(delm.join(['adjective', 'F/P', 'F/R', 'F/F1', 'T/P', 'T/R', 'T/F1', 'Acc']) + ' \\\\')
    for adj in adjs:
        l = p_r_f1(llm, adj, ds=ds)
        l = l.replace('nan', '0.000')
        if verbose: print(l + ' \\\\')
        lines.append(l)
    return lines

def dp3(r):
    return str("{:.3g}".format(r))
def dp5(r):
    return str("{:.5g}".format(r))

import statistics 
def get_accuracy(llm, adj, ds='test'):
    conf = get_conf_mat(llm, adj, ds)
    acc = (conf[0,0]+conf[1,1]) / (conf[0,0]+conf[0,1]+conf[1,0]+conf[1,1])
    return acc

def get_accuracies(llms=llms, adjs=positives+negatives, dss=['train', 'dev', 'test']):
    acc_table = {}
    for llm in llms:
        for adj in adjs:
            for ds in dss:
                acc_table[(llm, adj, ds)] = get_accuracy(llm, adj, ds)
    return acc_table

accuracy_table = get_accuracies()

def get_average_variance(llm, adjs=positives+negatives, ds='test'):
    l = []
    for adj in adjs:
        l.append(accuracy_table[(llm, adj, ds)])
    ave = statistics.mean(l)
    var = statistics.variance(l)
    return float(dp3(ave)), float(dp5(var))

def for_paper(llms=llms, adjs=positives+negatives, dss=['train', 'dev', 'test']):
    lines = []
    for adj in adjs:
        adj_items = []
        for llm in llms:
            for ds in dss:
                adj_items.append(dp3(accuracy_table[(llm, adj, ds)]))
        adj_line = ' & '.join([adj]+adj_items) + '\\\\'
        #print(adj_line)
        lines.append(adj_line)
    #
    pos_ave_line = []
    for llm in llms:
        for ds in dss:
            pos_ave_line.append(get_average_variance(llm, adjs=positives, ds=ds))
    neg_ave_line = []
    for llm in llms:
        for ds in dss:
            neg_ave_line.append(get_average_variance(llm, adjs=negatives, ds=ds))
    all_ave_line = []
    for llm in llms:
        for ds in dss:
            all_ave_line.append(get_average_variance(llm, adjs=positives+negatives, ds=ds))
    #
    for line in lines[0:4]:
        print(line)
    print(' & '.join(['pos.avg.'] + [str(_[0]) for _ in pos_ave_line]) + '\\\\')
    #
    for line in lines[4:]:
        print(line)
    print(' & '.join(['neg.avg.'] + [str(_[0]) for _ in neg_ave_line]) + '\\\\')
    print(' & '.join(['all avg.'] + [str(_[0]) for _ in all_ave_line]) + '\\\\')
    
def accuracies(llm, adjs=positives+negatives, ds='test'):
    acc_list = [float(l.split('&')[-1]) for l in comp_for_adjs(llm, adjs=adjs, ds=ds, verbose=False)]
    acc_mean = statistics.mean(acc_list)
    acc_var = statistics.variance(acc_list)
    return float(dp3(acc_mean)), float(dp5(acc_var))

def acc_by_llms(llms = ['gpt-3.5-turbo-1106', 'gpt-4o-2024-05-13', 'llama3-8B', 'mistral-7B'], ds='test'):
    for llm in llms:
        print(llm)
        pos = accuracies(llm, adjs=positives, ds=ds)
        print('positives:', pos)
        neg = accuracies(llm, adjs=negatives, ds=ds)
        print('negatives:', neg)
        all = accuracies(llm, ds=ds)
        print('all:', all)
        print()
