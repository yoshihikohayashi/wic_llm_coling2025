#
import pandas as pd

def get_zs_tsv_dir(llm):
    return './zs_tsv/'

def get_df(llm, adj, dataset='dev', model_id=323):
    zs_tsv_dir = get_zs_tsv_dir(llm)
    fname = zs_tsv_dir + '_'.join([llm, adj, dataset, str(model_id)]) + '.tsv'
    df = pd.read_csv(fname, delimiter='\t')
    return df

def apply_cond(df, gold, pred):
    return df[(df['l']==gold) & (df['pred']==pred)]

def compare(llm1, adj1, gold1, pred1, llm2, adj2, gold2, pred2, dataset='dev', model_id=323, save=False):
    df1 = get_df(llm1, adj1, dataset, model_id)
    df2 = get_df(llm2, adj2, dataset, model_id)
    df1_ids = set(apply_cond(df1, gold1, pred1)['id'])
    df2_ids = set(apply_cond(df2, gold2, pred2)['id'])
    conj_ids = df1_ids.intersection(df2_ids)
    df1_sel = df1[df1['id'].isin(conj_ids)]
    if save:
        fname = './comp_res/' + '_'.join([llm1, adj1, gold1, pred1]) + '__' + '_'.join([llm2, adj2, gold2, pred2]) + '.tsv'
        df1_sel.to_csv(fname, sep='\t')
    return df1_sel

llms = ['gpt-4o-2024-05-13', 'gpt-3.5-turbo-1106', 'llama3-8B', 'mistral-7B']
adjs = ['identical', 'the same', 'similar', 'related']
gs = ['T', 'F']
ps = ['T', 'F']

def main__(llms=llms, dataset='dev', model_id=323):
    for llm in llms:
        for i, adj in enumerate(adjs):
            if i==len(adjs)-1: break
            for g in gs:
                df = compare(llm, adjs[i], g, 'F', llm, adjs[i+1], g, 'T', dataset=dataset)
                print(llm, adjs[i], g, 'F', adjs[i+1], 'T', len(df))
                df = compare(llm, adjs[i], g, 'T', llm, adjs[i+1], g, 'F', dataset=dataset)
                print(llm, adjs[i], g, 'T', adjs[i+1], 'F', len(df))
