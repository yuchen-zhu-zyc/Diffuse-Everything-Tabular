import torch

import argparse
import warnings
import time
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from TabularDiT import TabularDiT
from sampler import EM_TauLeap_Sampler_VP, EM_TauLeap_Sampler_VE, Heun_TauLeap_Sampler_VE
from noise import LinearBeta, LinearBeta2, LogLinearNoise, PolynomialNoise

import dataset

warnings.filterwarnings('ignore')

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
    

def main(args):
    # data
    dataname = args.dataname
    data_dir = f'data/{dataname}'
    datainfo_path = f'data/{dataname}/info.json'
    with open(datainfo_path, 'r') as f:
        info = json.load(f)
    X_num, X_cat, categories, d_numerical = dataset.preprocess(data_dir, task_type = info['task_type'])
    _, _, categories, d_numerical, num_inverse, cat_inverse = dataset.preprocess(data_dir, task_type = info['task_type'], inverse = True)
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat
    
    if not args.imputation:
        num_samples = X_train_num.shape[0]
    if args.imputation:
        num_samples = X_test_num.shape[0]

    if args.input_standardize:
        X_train_mean = torch.from_numpy(np.mean(X_train_num, axis=0))
        X_train_std  = torch.from_numpy(np.std(X_train_num, axis=0))
    else:
        X_train_mean = None
        X_train_std = None

    # model
    ckpt_path = args.ckpt_path
    device = args.device
    steps = args.steps

    if args.save_path == '':
        save_path = '/'.join(ckpt_path.split('/')[:-1])
    else:
        save_path = args.save_path

    args.two_time = args.imputation
    print(f"args.two_time: {args.two_time}")
    model = TabularDiT(categories, d_numerical, 
                       hidden_size = args.model_hidden_size, 
                       depth = args.model_depth, 
                       num_heads = args.model_num_heads, 
                       mlp_ratio = args.model_mlp_ratio,
                       two_time = args.two_time).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    print("Pretrained model loaded.\nGenerating samples for {}...".format(ckpt_path))

    '''
        Generating samples    
    '''
    start_time = time.time()

    noise_cat = LogLinearNoise()
    if args.SDE == "VP-LinearBeta":
        noise_num = LinearBeta()
        sampler = EM_TauLeap_Sampler_VP(noise_num, noise_cat, categories, d_numerical, device, pfode = False)
    elif args.SDE == "VP-LinearBeta2":
        noise_num = LinearBeta2()
        sampler = EM_TauLeap_Sampler_VP(noise_num, noise_cat, categories, d_numerical, device, pfode = False)
    
    with torch.no_grad():
        x_num_final = torch.zeros(num_samples, d_numerical)
        x_cat_final = torch.zeros(num_samples, len(categories))
        num_chunk = num_samples // args.sample_batch_size
        for chunk_idx in tqdm(range(num_chunk)):
            x_num_batch, x_cat_batch = sampler.sample(args.sample_batch_size, model, steps)
            x_num_final[(chunk_idx * args.sample_batch_size) : ((chunk_idx + 1) * args.sample_batch_size), :] = x_num_batch.detach().cpu().clone()
            x_cat_final[(chunk_idx * args.sample_batch_size) : ((chunk_idx + 1) * args.sample_batch_size), :] = x_cat_batch.detach().cpu().clone()
        # final batch
        if num_samples - (chunk_idx + 1) * args.sample_batch_size > 0:
            x_num_batch, x_cat_batch = sampler.sample(num_samples - (chunk_idx + 1) * args.sample_batch_size, model, steps)
            x_num_final[((chunk_idx + 1) * args.sample_batch_size):, :] = x_num_batch.detach().cpu().clone()
            x_cat_final[((chunk_idx + 1) * args.sample_batch_size):, :] = x_cat_batch.detach().cpu().clone()
            
            
    syn_num_bf_inv, syn_num_af_inv, syn_cat_bf_inv, syn_cat_af_inv, syn_target_bf_inv, syn_target_af_inv = get_syn_target(info, 
                        x_num_final, x_cat_final, num_inverse, cat_inverse, args.input_standardize, X_train_mean, X_train_std) 

    syn_df_bf_inv = recover_data(syn_num_bf_inv, syn_cat_bf_inv, syn_target_bf_inv, info)
    syn_df_af_inv = recover_data(syn_num_af_inv, syn_cat_af_inv, syn_target_af_inv, info)
    print('Recovering syn_df...')

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df_bf_inv.rename(columns = idx_name_mapping, inplace=True)
    syn_df_af_inv.rename(columns = idx_name_mapping, inplace=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not args.imputation:
        syn_df_bf_inv.to_csv(os.path.join(save_path, 'samples_bf_inv_{}.csv'.format(ckpt_path.split('/')[-1].split("_")[1].split(".")[0])), index=False)
        syn_df_af_inv.to_csv(os.path.join(save_path, 'samples_af_inv_{}.csv'.format(ckpt_path.split('/')[-1].split("_")[1].split(".")[0])), index=False)
    if args.imputation:
        syn_df_bf_inv.to_csv(os.path.join(save_path, 'imputed_samples_bf_inv_{}.csv'.format(ckpt_path.split('/')[-1].split("_")[1].split(".")[0])), index=False)
        syn_df_af_inv.to_csv(os.path.join(save_path, 'imputed_samples_af_inv_{}.csv'.format(ckpt_path.split('/')[-1].split("_")[1].split(".")[0])), index=False)

    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')
    # dataset parameters
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--input_standardize', default=False, type=lambda x: (str(x).lower() == 'true'), help="If standardize the dataset.")

    # model parameters
    parser.add_argument('--ckpt_path', type=str, default='./exp/adult/000/model_0029999.pt', help='Checkpoint path.')
    parser.add_argument('--model_hidden_size', type=int, default=24, help='TabularDiT, hidden size.')
    parser.add_argument('--model_depth', type=int, default=4, help='TabularDiT, depth.')
    parser.add_argument('--model_num_heads', type=int, default=4, help='TabularDiT, num_heads.')
    parser.add_argument('--model_mlp_ratio', type=int, default=4, help='TabularDiT, mlp_ratio.')
    parser.add_argument('--imputation', default=False, type=lambda x: (str(x).lower() == 'true'), help="Imputation task?")

    # 
    parser.add_argument('--SDE', type=str, default="VP-LinearBeta", choices=["VP-LinearBeta", "VP-LinearBeta2"], help='SDE type')
    parser.add_argument('--save_path', type=str, default='', help='Save path for samples.')
    parser.add_argument('--sample_batch_size', type=int, default=8000, help='Number of samples in one batch.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--steps', type=int, default=1000, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)