# 2024-08-04 09:56:28
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
import pandas as pd
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.linear_model import LogisticRegression as lgrc
from sklearn.neural_network import MLPClassifier as nnc

##
def comp_accuracy(golds, preds):
    c = 0
    for g, p in zip(golds, preds):
        if g==p: c += 1
    return c/len(golds)
    
def get_zs_tsv_dir(llm):
    return './zs_tsv/'

def get_gold_labels(data_split, model_id=323):
    llm, adj = 'mistral-7B', 'identical' # anything is okay
    zsv_dir = get_zs_tsv_dir(llm)
    fname = zsv_dir + '_'.join([llm, adj, data_split, str(model_id)]) + '.tsv'
    with open(fname, 'r') as fn:
        df = pd.read_csv(fn, delimiter='\t')
    golds = df['l']
    return golds

def get_pred_labels(predictor, data_split, model_id=323):
    llm, adj = predictor # predictor: tuple-of llm and adj
    zsv_dir = get_zs_tsv_dir(llm)
    fname = zsv_dir + '_'.join([llm, adj, data_split, str(model_id)]) + '.tsv'
    with open(fname, 'r') as fn:
        df = pd.read_csv(fn, delimiter='\t')
    golds = df['pred']
    return golds

def get_candidate_predictors_results(predictors, data_split='train'):
    preds_list = [get_pred_labels(pr, data_split) for pr in predictors]
    return predictors, preds_list

##
def classify_instances(golds, preds):
    T_OK = []; T_NG = []; F_OK = []; F_NG = []
    for i, (g, p) in enumerate(zip(golds, preds)):
        if g=='T':
            if g==p: T_OK.append(i)
            else: T_NG.append(i)
        else:
            if g==p: F_OK.append(i)
            else: F_NG.append(i)
    return set(T_OK), set(T_NG), set(F_OK), set(F_NG)

##
def select_adding_predictor(golds, meta_predictor_preds, cand_predictors, cand_predictors_preds_list, selected_predictors):
    meta_T_ok, meta_T_ng, meta_F_ok, meta_F_ng = classify_instances(golds, meta_predictor_preds)
    selected_predictor = None
    max_score_sofar = -9999
    #
    for cand_predictor, cand_predictor_preds in zip(cand_predictors, cand_predictors_preds_list):
        if cand_predictor in selected_predictors: continue
        T_ok, T_ng, F_ok, F_ng = classify_instances(golds, cand_predictor_preds)
        ok_instances = T_ok.union(F_ok)
        ng_instances = T_ng.union(F_ng)        
        only_cand_ok = ok_instances.difference(meta_T_ok.union(meta_F_ok))
        only_meta_ok = (meta_T_ok.union(meta_F_ok)).difference(ok_instances)
        meta_ok = meta_T_ok.union(meta_F_ok)
        both_ok = ok_instances.intersection(meta_ok)
        #
        score = (len(meta_ok) + len(only_cand_ok)) * len(both_ok) # 理想的なケースも最悪?のケースもOKを最大化したい
        #
        if score > max_score_sofar:
            selected_predictor = cand_predictor
            selected_predictor_preds = cand_predictor_preds
            max_score_sofar = score
    if not selected_predictor: return None, None
    return selected_predictor, selected_predictor_preds

#
def show_best_results(best_initial_index, best_ensemble_index, results):
    print('*****************')
    print('Best single predictor:', best_initial_index, results[best_initial_index][0], 'test acc:', results[best_initial_index][3][0], 
          'test macro avg f1:', results[best_initial_index][3][2]['macro avg']['f1-score'],
          'test macro avg p-r diff:', abs(results[best_initial_index][3][2]['macro avg']['precision']-results[best_initial_index][3][2]['macro avg']['recall']),
    )
    print('Best ensemble predictors:', best_ensemble_index, results[best_ensemble_index][1], 'test acc:',results[best_ensemble_index][5][0], 
          'test macro avg f1:', results[best_ensemble_index][5][2]['macro avg']['f1-score'],
          'test macro avg p-r diff:', abs(results[best_ensemble_index][5][2]['macro avg']['precision']-results[best_ensemble_index][5][2]['macro avg']['recall']),
    )

def retrieve_best_cases(results):
    best_initial_test_acc = best_ensemble_test_acc = 0.0
    best_initial_index = best_ensemble_index = -1
    for i, r_tuple in enumerate(results):
        if r_tuple[3][0] > best_initial_test_acc:
            best_initial_index = i
            best_initial_test_acc = r_tuple[3][0]
        if r_tuple[5][0] > best_ensemble_test_acc:
            best_ensemble_index = i
            best_ensemble_test_acc = r_tuple[5][0]
    return best_initial_index, best_ensemble_index

#
def greedy_ensemble(initial_predictor, predictors, data_split='dev', add_corr=False, k=4):
    meta_predictor, meta_predictor_preds, selected_predictors, initial_summ, initial_test_summ, summ \
        = greedy_search_predictors_combination(initial_predictor, predictors, data_split, add_corr, k)
    #
    test_golds = get_gold_labels('test')
    if not meta_predictor: # only initial predictor
        print('!!! Only the initial predictor yielded best dev result!')
        initial_preds = get_pred_labels(initial_predictor, 'test')
        test_acc = comp_accuracy(test_golds, initial_preds)
        test_summ = make_results_summary(test_golds, initial_preds)
    else:
        test_X = []
        for predictor in selected_predictors:
            test_X.append(le.fit_transform(get_pred_labels(predictor, 'test', model_id=323).to_numpy()))
        test_X = np.array(test_X, dtype=np.float32).T
        if add_corr: test_X = corr_predictors(test_X)
        test_preds = meta_predictor.predict(test_X)
        test_acc = comp_accuracy(test_golds, test_preds)
        test_summ = make_results_summary(test_golds, test_preds)
    return initial_predictor, selected_predictors, initial_summ, initial_test_summ, summ, test_summ

def greedy_search_predictors_combination(initial_predictor, predictors, data_split='dev', add_corr=False, k=4):
    golds = get_gold_labels(data_split)
    test_golds = get_gold_labels('test')
    initial_preds = get_pred_labels(initial_predictor, data_split)
    initial_test_pred = get_pred_labels(initial_predictor, 'test')
    initial_acc = comp_accuracy(golds, initial_preds)
    initial_summ = make_results_summary(golds, initial_preds)
    initial_test_acc = comp_accuracy(test_golds, initial_test_pred)
    initial_test_summ = make_results_summary(test_golds, initial_test_pred)
    best_acc_sofar = initial_acc
    best_summ_sofar = initial_summ
    cand_predictors, cand_predictors_preds_list = get_candidate_predictors_results(predictors, data_split=data_split)
    selected_predictors = [initial_predictor] 
    examined_predictors = [initial_predictor] 
    _, _meta_predictor_preds_list = get_candidate_predictors_results(selected_predictors, data_split=data_split)
    meta_predictor = None
    meta_predictor_preds = _meta_predictor_preds_list[0]
    #
    iter_count = 0
    while (1):
        iter_count += 1
        to_be_added_predictor, to_be_added_predictor_result \
            = select_adding_predictor(golds, meta_predictor_preds, cand_predictors, cand_predictors_preds_list, examined_predictors)
        if not to_be_added_predictor: break
        examined_predictors.append(to_be_added_predictor)
        new_meta_predictor, new_meta_predictor_preds = train_new_meta_predictor(selected_predictors+[to_be_added_predictor], add_corr=add_corr)
        new_accuracy = comp_accuracy(golds, new_meta_predictor_preds)
        new_summ = make_results_summary(golds, new_meta_predictor_preds)
        #
        if new_accuracy > best_acc_sofar:
            selected_predictors.append(to_be_added_predictor)
            meta_predictor = new_meta_predictor
            meta_predictor_preds = new_meta_predictor_preds
            best_acc_sofar = new_accuracy
            best_summ_sofar = new_summ
        #
        if meet_stopping_cond(golds, meta_predictor_preds, new_meta_predictor_preds, iter_count, k=k):
            break
    #
    acc = comp_accuracy(golds, meta_predictor_preds)
    summ = make_results_summary(golds, meta_predictor_preds)
    return meta_predictor, meta_predictor_preds, selected_predictors, initial_summ, initial_test_summ, summ

#
def meet_stopping_cond(golds, meta_predictor_preds, new_meta_predictor_preds, iter_count, k=8):
    if iter_count >= k-1: return True
    return False

def train_new_meta_predictor(predictors, add_corr):
    train_golds = get_gold_labels('train')
    dev_golds = get_gold_labels('dev')
    model, result = train_eval(predictors, train_golds, dev_golds, 'train', 'dev', add_corr=add_corr)
    #
    return model, result

#
def train_eval(predictors, train_golds, dev_golds, train_split, dev_split, add_corr):
    model = eval("nnc(solver='sgd', hidden_layer_sizes=(32,128,32), max_iter=10000, shuffle=True, random_state=323)")
    # train
    train_X = []
    for predictor in predictors:
        train_X.append(le.fit_transform(get_pred_labels(predictor, train_split, model_id=323).to_numpy()))
    train_X = np.array(train_X, dtype=np.float32).T
    # dev
    dev_X = []
    for predictor in predictors:
        dev_X.append(le.fit_transform(get_pred_labels(predictor, dev_split, model_id=323).to_numpy()))
    dev_X = np.array(dev_X, dtype=np.float32).T
    #
    if add_corr:
        train_X = corr_predictors(train_X)
        dev_X = corr_predictors(dev_X)
    # fit and predict
    trained_model = model.fit(train_X, train_golds)
    predicted = trained_model.predict(dev_X)
    #
    return trained_model, predicted

def corr_predictors(X):
    n_inst, n_cl = X.shape
    C = []
    for i in range(X.shape[0]):
        combs = list(itertools.combinations(X[i], 2))
        c = []
        for comb in combs:
            if comb[0]==comb[1]:
                c.append(1)
            else:
                c.append(0)
        C.append(c)
    return np.concatenate([X, C], axis=1)

### results summary
from sklearn.metrics import confusion_matrix, classification_report
def make_results_summary(gold_list, predicted_list):
    conf = confusion_matrix(gold_list, predicted_list)
    rep = classification_report(gold_list, predicted_list, zero_division=0.0, output_dict=True)
    #
    corr = 0
    for p, g in zip(predicted_list, gold_list):
        if p[:2] == g[:2]: corr += 1
    return corr/len(predicted_list), conf, rep

###
import pickle

def getname(var):
    for k, v in globals().items():
        if v==var: return k

def main(add_corr=True):
    save_dir = './greedy_ensemble_results/'
    # preditors_sets = [llama3_predictors, mistral_predictors, gpt3_predictors, gpt4_predictors, wo_gpt_predictors, wo_gpt4_predictors]
    preditors_sets = [all_predictors, wo_gpt_predictors]
    for ps in preditors_sets:
        if add_corr:
            fname = save_dir + getname(ps) + '_add_corr' + '.pkl'
        else:
            fname = save_dir + getname(ps) + '.pkl'
        print('Start:', ps)
        results, best_initial_index, best_ensemble_index = main_(ps, add_corr=add_corr)
        with open(fname, 'wb') as fn:
            pickle.dump((best_initial_index, best_ensemble_index, results), fn)
        print('Saved:', fname)
        print()

def main_(testing_predictors, add_corr=False, verbose=True):
    if verbose: print('Testing predictors:', testing_predictors, '\n')
    #
    results = []
    for i, predictor in enumerate(testing_predictors):
        r_tuple = greedy_ensemble(predictor, testing_predictors, add_corr=add_corr, k=32)
        results.append(r_tuple)
        if verbose:
            print('Iteration:', i)
            print('Initial:', r_tuple[0], 'dev acc:', r_tuple[2][0], 'test acc:', r_tuple[3][0])
            print('Selected predictors:', len(r_tuple[1]), 'dev acc:', r_tuple[4][0], 'test acc:', r_tuple[5][0])    
            print(r_tuple[1])
            print()
    best_initial_index, best_ensemble_index = retrieve_best_cases(results)
    if verbose:
        show_best_results(best_initial_index, best_ensemble_index, results)
    return results, best_initial_index, best_ensemble_index

def show_(testing_predictors=[all_predictors, wo_gpt_predictors], add_corr=True, verbose=True):
    if verbose: print('Testing predictors:', testing_predictors, '\n')
    save_dir = './greedy_ensemble_results/'
    preditors_sets = testing_predictors
    for ps in preditors_sets:
        if add_corr:
            fname = save_dir + getname(ps) + '_add_corr' + '.pkl'
        else:
            fname = save_dir + getname(ps) + '.pkl'
        print('Results:', fname)
        with open(fname, 'rb') as fn:
            best_initial_index, best_ensemble_index, results = pickle.load(fn)
        #
        best_initial_test_acc = results[best_initial_index][3][2]['accuracy']
        best_initial_test_f1 = results[best_initial_index][3][2]['macro avg']['f1-score']
        best_initial_test_pr_diff = abs(results[best_initial_index][3][2]['macro avg']['precision'] - results[best_initial_index][3][2]['macro avg']['recall'])
        #
        best_ensemble_test_acc = results[best_ensemble_index][5][2]['accuracy']
        best_ensemble_test_f1 = results[best_ensemble_index][5][2]['macro avg']['f1-score']
        best_ensemble_test_pr_diff = abs(results[best_ensemble_index][5][2]['macro avg']['precision'] - results[best_ensemble_index][5][2]['macro avg']['recall'])
        #
        if best_ensemble_test_acc > best_initial_test_acc: print('Accuracy improved', best_ensemble_test_acc - best_initial_test_acc)
        if best_ensemble_test_f1 > best_initial_test_f1: print('F1 improved!:', best_ensemble_test_f1 - best_initial_test_f1)
        if best_ensemble_test_pr_diff < best_initial_test_pr_diff: print('P-R Difference improved:', best_initial_test_pr_diff - best_ensemble_test_pr_diff)
        show_best_results(best_initial_index, best_ensemble_index, results)
        print()
