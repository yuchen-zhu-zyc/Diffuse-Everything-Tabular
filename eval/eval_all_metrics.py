import numpy as np
import pandas as pd
import os 
import torch
import json
import time
from tqdm import tqdm
from copy import deepcopy

from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
pd.options.mode.chained_assignment = None

# Metrics
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

from mle.mle import get_evaluator
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TabularDiT import TabularDiT
from sampler import EM_TauLeap_Sampler_VP, EM_TauLeap_Sampler_VE, Heun_TauLeap_Sampler_VE
from noise import LogLinearNoise, PolynomialNoise, LinearBeta, LinearBeta2
from dataset import preprocess, TabularDataset

import warnings
warnings.filterwarnings("ignore")

import argparse

# sampling function #
def get_syn_target(info, x_num_final, x_cat_final, 
                   num_inverse, cat_inverse, 
                   input_standardize=False, X_train_mean=None, X_train_std=None):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    if not input_standardize:
        syn_num_bf_inv = x_num_final
    elif input_standardize:
        syn_num_bf_inv = x_num_final * X_train_std + X_train_mean
    syn_cat_bf_inv = x_cat_final

    syn_num_af_inv = num_inverse(syn_num_bf_inv).astype(np.float32)
    syn_cat_af_inv = cat_inverse(syn_cat_bf_inv)
    
    if info['task_type'] == 'regression':
        syn_target_bf_inv = syn_num_bf_inv[:, :len(target_col_idx)]
        syn_target_af_inv = syn_num_af_inv[:, :len(target_col_idx)]
        syn_num_bf_inv    = syn_num_bf_inv[:, len(target_col_idx):]
        syn_num_af_inv    = syn_num_af_inv[:, len(target_col_idx):]
    
    else:
        syn_target_bf_inv = syn_cat_bf_inv[:, :len(target_col_idx)]
        syn_target_af_inv = syn_cat_af_inv[:, :len(target_col_idx)]
        syn_cat_bf_inv    = syn_cat_bf_inv[:, len(target_col_idx):]
        syn_cat_af_inv    = syn_cat_af_inv[:, len(target_col_idx):]

    return syn_num_bf_inv, syn_num_af_inv, syn_cat_bf_inv, syn_cat_af_inv, syn_target_bf_inv, syn_target_af_inv


def recover_data(syn_num, syn_cat, syn_target, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df


def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata


# evaluation metrics: density # 
def eval_density(real_data, syn_data, info):

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    qual_report = QualityReport()
    qual_report.generate(new_real_data, new_syn_data, metadata)

    diag_report = DiagnosticReport()
    diag_report.generate(new_real_data, new_syn_data, metadata)

    quality =  qual_report.get_properties()
    diag = diag_report.get_properties()

    Shape = quality['Score'][0]
    Trend = quality['Score'][1]

    return Shape, Trend


# evaluation metrics: mle #
def eval_mle(train, test, info):
    task_type = info['task_type']
    print(task_type)
    evaluator = get_evaluator(task_type)

    if task_type == 'regression':
        best_r2_scores, best_rmse_scores = evaluator(train, test, info)
        
        overall_scores = {}
        best_rmse_score = float("inf")
        for score_name in ['best_r2_scores', 'best_rmse_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 
                best_rmse_score = min(best_rmse_score, method["RMSE"])
        print("best_rmse_score: ", best_rmse_score)

        return best_rmse_score


    else:
        best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info)

        overall_scores = {}
        best_auroc_score = 0
        for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method
                best_auroc_score = max(best_auroc_score, method["roc_auc"])
        print("best_auroc_score: ", best_auroc_score)
        return best_auroc_score


# evaluation metrics: quality #
def eval_quality(real_data, syn_data, info):
    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx
        
    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')
        

    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    num_syn_data_np = num_syn_data.to_numpy()
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)

    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis = 1)).astype(float)
    le_real_num = pd.DataFrame(num_real_data_np).astype(float)
    le_real_cat = pd.DataFrame(cat_real_data_oh).astype(float)


    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis = 1)).astype(float)
    le_syn_num = pd.DataFrame(num_syn_data_np).astype(float)
    le_syn_cat = pd.DataFrame(cat_syn_data_oh).astype(float)

    np.set_printoptions(precision=4)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation of AlphaPrecision
    qual_score = np.mean(list(qual_res.values()))

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive'] ))

    Alpha_Precision_all = qual_res['delta_precision_alpha_naive']
    Beta_Recall_all = qual_res['delta_coverage_beta_naive']

    return Alpha_Precision_all, Beta_Recall_all



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # dataset name
    parser.add_argument('--dataname', type=str, default='adult')

    # model parameters
    parser.add_argument('--model', type=str, default='TabularDiT')
    parser.add_argument('--ckpt_path', type=str, default="./exp/adult/000/model_0029999.pt")
    parser.add_argument('--model_hidden_size', type=int, default=24, help='TabularDiT, hidden size.')
    parser.add_argument('--model_depth', type=int, default=4, help='TabularDiT, depth.')
    parser.add_argument('--model_num_heads', type=int, default=4, help='TabularDiT, num_heads.')
    parser.add_argument('--model_mlp_ratio', type=int, default=4, help='TabularDiT, mlp_ratio.')
    parser.add_argument('--two_time', default=False, type=lambda x: (str(x).lower() == 'true'), help="If standardize the dataset.")


    parser.add_argument('--SDE', type=str, default="VP-LinearBeta", choices=["VP-LinearBeta", "VP-LinearBeta2"], help='SDE type')

    # sample parameters
    parser.add_argument('--sample_batch_size', type=int, default=7000, help='Number of samples in one batch.')
    parser.add_argument('--sampling_steps', type=int, default=1000, help='Number of function evaluations.')

    # evaluation times
    parser.add_argument('--num_eval_runs', type=int, default=20, help="[Evaluation] Number of eval runs to take the average and std.")

    # gpu ID
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    dataname = args.dataname
    data_dir = f'data/{dataname}' 
    args.data_dir = f'data/{args.dataname}'

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    real_path = f'synthetic/{dataname}/real.csv'
    test_path = f'synthetic/{dataname}/test.csv'
    real_data = pd.read_csv(real_path)
    test = pd.read_csv(test_path).to_numpy()


    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    
    X_num, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(args.data_dir, task_type = info['task_type'], inverse = True)
    X_train_num, _ = X_num
    num_samples = X_train_num.shape[0]

    model = TabularDiT(categories, d_numerical, 
                    hidden_size = args.model_hidden_size, 
                    depth = args.model_depth, 
                    num_heads = args.model_num_heads, 
                    mlp_ratio = args.model_mlp_ratio,
                    two_time = args.two_time).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device)["ema"])
    model.eval()
    
    shape_list = []
    trend_list = []
    mle_list = []
    precision_list = []
    recall_list = []

    start_time = time.time()

    noise_cat = LogLinearNoise()
    if args.SDE == "VP-LinearBeta":
        noise_num = LinearBeta()
        sampler = EM_TauLeap_Sampler_VP(noise_num, noise_cat, categories, d_numerical, args.device, pfode = False)
    elif args.SDE == "VP-LinearBeta2":
        noise_num = LinearBeta2()
        sampler = EM_TauLeap_Sampler_VP(noise_num, noise_cat, categories, d_numerical, args.device, pfode = False)
        
    for run_idx in range(args.num_eval_runs):
        print('Start sampling {}...'.format(run_idx))
        with torch.no_grad():
            x_num_final = torch.zeros(num_samples, d_numerical)
            x_cat_final = torch.zeros(num_samples, len(categories))
            num_chunk = num_samples // args.sample_batch_size
            for chunk_idx in tqdm(range(num_chunk)):
                x_num_batch, x_cat_batch = sampler.sample(args.sample_batch_size, model, args.sampling_steps)
                x_num_final[(chunk_idx * args.sample_batch_size) : ((chunk_idx + 1) * args.sample_batch_size), :] = x_num_batch.detach().cpu().clone()
                x_cat_final[(chunk_idx * args.sample_batch_size) : ((chunk_idx + 1) * args.sample_batch_size), :] = x_cat_batch.detach().cpu().clone()
            # final batch
            if num_samples - (chunk_idx + 1) * args.sample_batch_size > 0:
                x_num_batch, x_cat_batch = sampler.sample(num_samples - (chunk_idx + 1) * args.sample_batch_size, model, args.sampling_steps)
                x_num_final[((chunk_idx + 1) * args.sample_batch_size):, :] = x_num_batch.detach().cpu().clone()
                x_cat_final[((chunk_idx + 1) * args.sample_batch_size):, :] = x_cat_batch.detach().cpu().clone()
            
        syn_num_bf_inv, syn_num_af_inv, syn_cat_bf_inv, syn_cat_af_inv, syn_target_bf_inv, syn_target_af_inv = get_syn_target(info, 
                            x_num_final, x_cat_final, num_inverse, cat_inverse, False, None, None) 

        syn_df_af_inv = recover_data(syn_num_af_inv, syn_cat_af_inv, syn_target_af_inv, info)
        print('Recovering syn_df...')

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df_af_inv.rename(columns = idx_name_mapping, inplace=True)

        end_time = time.time()
        print('Time for sampling:', end_time - start_time)

        syn_data = syn_df_af_inv.copy()
        train = syn_data.copy().to_numpy()

        with open(f'{data_dir}/info.json', 'r') as f:
            info = json.load(f)
        shape, trend = eval_density(real_data.copy(), syn_data.copy(), deepcopy(info))
        shape_list.append(shape*100)
        trend_list.append(trend*100)

        ## need to run eval_mle.py separately for datasets: default and news ##
        if dataname != "default" and dataname != "news":                      #
            with open(f'{data_dir}/info.json', 'r') as f:                     #
                info = json.load(f)                                           #
            mle = eval_mle(train, test, deepcopy(info))                       #
            mle_list.append(mle)                                              #
        ## for default and news, please run eval_mle.py separately           ##
        ## instructions are in: run_eval_mle_default_news.sh                 ##
        #######################################################################
        
        ap, br = eval_quality(real_data.copy(), syn_data.copy(), deepcopy(info))
        precision_list.append(ap*100)
        recall_list.append(br*100)

    shape_array = np.array(shape_list)
    trend_array = np.array(trend_list)
    mle_array = np.array(mle_list)
    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)

    print("#" * 50)
    print("Shape: mean: {:.6f}, std: {:.6f}, var: {:.6f}".format(np.mean(shape_array), np.std(shape_array), np.var(shape_array)))
    print("Trend: mean: {:.6f}, std: {:.6f}, var: {:.6f}".format(np.mean(trend_array), np.std(trend_array), np.var(trend_array)))
    print("MLE  : mean: {:.6f}, std: {:.6f}, var: {:.6f}".format(np.mean(mle_array),   np.std(mle_array),   np.var(mle_array)))
    print("Precision: mean: {:.6f}, std: {:.6f}, var: {:.6f}".format(np.mean(precision_array), np.std(precision_array), np.var(precision_array)))
    print("Recall:    mean: {:.6f}, std: {:.6f}, var: {:.6f}".format(np.mean(recall_array),    np.std(recall_array),    np.var(recall_array)))
    print("#" * 50)    





