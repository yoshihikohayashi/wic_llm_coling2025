# rank_corr.py (2025-01-11 09:00:17)
from scipy.stats import spearmanr, kendalltau
import pandas as pd

#
def list2ranks(list, reverse=True):
    sorted_list = sorted(list, reverse=reverse)
    ranks = {v: i+1 for i, v in enumerate(sorted_list)}
    rank_list = [ranks[v] for v in list]
    return rank_list

def pr_table2corrs(table_df):
    norm_ranks = list(range(4))
    rev_ranks = list(range(4)).reverse()
    #
    pos_fp = list2ranks(list(table_df['F/P'][:4]), reverse=False)
    pos_fr = list2ranks(list(table_df['F/R'][:4]), reverse=True)
    pos_tp = list2ranks(list(table_df['T/P'][:4]), reverse=True)
    pos_tr = list2ranks(list(table_df['T/R'][:4]), reverse=False)
    #
    neg_fp = list2ranks(list(table_df['F/P'][4:]), reverse=False)
    neg_fr = list2ranks(list(table_df['F/R'][4:]), reverse=True)
    neg_tp = list2ranks(list(table_df['T/P'][4:]), reverse=True)
    neg_tr = list2ranks(list(table_df['T/R'][4:]), reverse=False)
    #
    correlations = []
    for ranks in [pos_fp, pos_fr, pos_tp, pos_tr, neg_fp, neg_fr, neg_tp, neg_tr]:
        correlations.append((spearmanr(norm_ranks, ranks)[0],
                             kendalltau(norm_ranks, ranks)[0]))
    return correlations

def main(llm, llms_pr_dir='./llms_pr'):
    fname = llms_pr_dir + '/' + llm + '.tsv'
    pr_df = pd.read_csv(fname, delimiter='\t')
    return pr_table2corrs(pr_df)
