import numpy as np
import torch 
import pandas as pd
import os 
import sys

import json
from mle.mle import get_evaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--model', type=str, default='real')
    parser.add_argument('--syn_path', type=str, default="")
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')
    args = parser.parse_args()

    dataname = args.dataname
    model = args.model
    

    train_path = args.syn_path
    test_path = f'synthetic/{dataname}/test.csv'

    train = pd.read_csv(train_path).to_numpy()
    test = pd.read_csv(test_path).to_numpy()

    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

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
        print(best_rmse_score)
                

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
        print(best_auroc_score)  