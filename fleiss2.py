import itertools

all_predictors = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                        ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))
wo_gpt4_predictors = list(itertools.product(['gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                            ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))

wo_gpt_predictors = list(itertools.product(['llama3-8B', 'mistral-7B'],
                                            ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))

gpt_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106'],
                                         ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))

gpt4_predictors  = list(itertools.product(['gpt-4o-2024-05-13'],
                                         ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))

gpt3_predictors  = list(itertools.product(['gpt-3.5-turbo-1106'],
                                         ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))

llama3_predictors  = list(itertools.product(['llama3-8B'],
                                         ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))

mistral_predictors  = list(itertools.product(['mistral-7B'],
                                         ['identical', 'the same', 'similar', 'related', 'distinct', 'different', 'dissimilar', 'unrelated']))

#
identical_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['identical']))

same_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['the same']))

similar_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['similar']))

related_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['related']))

distinct_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['distinct']))

different_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['different']))

dissimilar_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['dissimilar']))

unrelated_predictors  = list(itertools.product(['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B'],
                                         ['unrelated']))

#
import pandas as pd
import statsmodels.stats.inter_rater as ir
import itertools

def get_zs_tsv_dir(llm):
    return './zs_tsv/'

def count_f(l):
    c = 0
    for _ in l:
        if _=='F': c+=1
    return c

def getname(var):
    for k, v in globals().items():
        if v==var: return k

#
def fleiss(data_split, predictors, model_id=323, verbose=False):
    fname_list = []
    for llm, adj in predictors:
        zs_tsv_dir = get_zs_tsv_dir(llm)
        fname_list.append(zs_tsv_dir + '_'.join([llm, adj, data_split, str(model_id)]) + '.tsv')
    preds = pd.DataFrame()
    
    for fname in fname_list:
        df = pd.read_csv(fname, delimiter='\t')['pred']
        preds = pd.concat((preds, df), axis=1)
    #
    ratings = []
    for index, p in preds.iterrows():
        c_f = count_f(list(p))
        ratings.append([c_f, len(list(p))-c_f])
    kappa = ir.fleiss_kappa(ratings, method='fleiss')
    if verbose: print('Fleiss kappa:', kappa)
    return kappa, ratings

def fleiss2(data_split, predictors, model_id=323, verbose=False):
    fname_list = []
    for llm, adj in predictors:
        zs_tsv_dir = get_zs_tsv_dir(llm)
        fname_list.append(zs_tsv_dir + '_'.join([llm, adj, data_split, str(model_id)]) + '.tsv')

    preds = pd.DataFrame()
    golds = pd.DataFrame()    
    for fname in fname_list:
        pred_df = pd.read_csv(fname, delimiter='\t')['pred']
        gold_df = pd.read_csv(fname, delimiter='\t')['l']
        preds = pd.concat((preds, pred_df), axis=1)
        golds = pd.concat((golds, gold_df), axis=1)

    ind_tbl = {'TT':0, 'TF':1, 'FT':2, 'FF':3}    
    ratings = []
    for (p_ind, p), (g_ind, g) in zip(preds.iterrows(), golds.iterrows()):
        p_list = list(p); g_list = list(g)
        gp_s= [x+y for (x, y) in list(zip(g_list, p_list))]
        r = [0,0,0,0]
        for gp in gp_s:
            r[ind_tbl[gp]] += 1
            ratings.append(r)
    #
    kappa = ir.fleiss_kappa(ratings, method='fleiss')
    if verbose: print('Fleiss kappa2:', kappa)
    return kappa, ratings

predictor_comb_set = [llama3_predictors, mistral_predictors, gpt3_predictors, gpt4_predictors, wo_gpt_predictors, wo_gpt4_predictors]
predictor_adj_set = [identical_predictors, same_predictors, similar_predictors, related_predictors, 
                     distinct_predictors, different_predictors, dissimilar_predictors, unrelated_predictors]

def main_(predictors_set=predictor_comb_set, data_split='test'):
    print('data split:', data_split)
    print()
    for ps in predictors_set:
        ps_ = getname(ps)
        k1, r1 = fleiss(data_split, ps, model_id=323)
        k2, r2 = fleiss2(data_split, ps, model_id=323)
        print('predictors set:', ps_)
        print('Kappa1, Kapp2, Diff:', k1, k2, k2-k1)
        print()

def dp3(r):
    return str("{:.3f}".format(r))

def for_paper(predictors_set=predictor_comb_set, data_split='test', delm=' & '):
    if predictors_set==predictor_comb_set:
        print(delm.join(['LLM', 'Kapp1', 'Kapp2', 'Diff']) + ' \\\\')
    else:
        print(delm.join(['Adjective', 'Kapp1', 'Kapp2', 'Diff']) + ' \\\\')
    #
    for ps in predictors_set:
        ps_ = getname(ps)
        k1, r1 = fleiss(data_split, ps, model_id=323)
        k2, r2 = fleiss2(data_split, ps, model_id=323)

        line = delm.join([ps_, dp3(k1), dp3(k2), dp3(k2-k1)]) + ' \\\\'
        print(line)
