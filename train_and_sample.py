import os
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from TabularDiT import TabularDiT
from noise import LogLinearNoise, PolynomialNoise, LinearBeta, LinearBeta2
from tqdm import tqdm
import dataset
from model import multimodal_diffusion_loss, multimodal_diffusion_loss_imp
import logging
import random
from glob import glob

import time
from sampler import EM_TauLeap_Sampler_VP, EM_TauLeap_Sampler_VE, Heun_TauLeap_Sampler_VE
import pandas as pd
from collections import OrderedDict
from copy import deepcopy

from torch.optim.lr_scheduler import _LRScheduler


# sampling func #
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

    # return syn_num, syn_cat, syn_target
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


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        return [base_lr * min(step / self.warmup_steps, 1.0) for base_lr in self.base_lrs]
#################

# Trainer #
def trainer(args):
    device = args.device
    args.two_time = args.imputation
    args.logger.info(f"args.two_time: {args.two_time}")

    with open(f'data/{args.dataname}/info.json', 'r') as f:
        info = json.load(f)

    X_num, X_cat, categories, d_numerical = dataset.preprocess(args.data_dir, task_type = info['task_type'])
    
    args.impute_cat = info['task_type'] == "binclass"
    
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # standardize numerical values >> mean=0
    if args.input_standardize:
        X_train_num = (X_train_num - np.mean(X_train_num, axis=0)) / np.std(X_train_num, axis=0)

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)
    
    train_data = dataset.TabularDataset(X_train_num.float(), X_train_cat)
    args.logger.info(f"Dataset property: categories: {categories} | d_numerical: {d_numerical} | len(train_data): {len(train_data)}")

    X_test_num = X_test_num.float().to(args.device)
    X_test_cat = X_test_cat.to(args.device)

    train_loader = DataLoader(
        train_data,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers)
    
    score_model = TabularDiT(categories, d_numerical, 
                            hidden_size = args.model_hidden_size, 
                            depth = args.model_depth, 
                            num_heads = args.model_num_heads, 
                            mlp_ratio = args.model_mlp_ratio,
                            two_time = args.two_time).to(device)
    optimizer = torch.optim.AdamW(score_model.parameters(), lr=1e-3, weight_decay=0.03, betas = (0.9, 0.9))

    if args.continue_from == "":
        args.logger.info(f">>> Model Initialized <<< \nhidden_size: {args.model_hidden_size} \ndepth: {args.model_depth} \nnum_heads: {args.model_num_heads} \nmlp_ratio: {args.model_mlp_ratio}")
        scheduler = WarmUpScheduler(optimizer, warmup_steps=args.init_warmup_steps)
    else:
        args.logger.info(f">>> Model Loaded <<< \ncontinue training from{args.continue_from} \nhidden_size: {args.model_hidden_size} \ndepth: {args.model_depth} \nnum_heads: {args.model_num_heads} \nmlp_ratio: {args.model_mlp_ratio}")
        
        score_model.load_state_dict(torch.load(args.continue_from, map_location=device)["model"])
        optimizer.load_state_dict(torch.load(args.continue_from, map_location=device)["opt"])
        scheduler = WarmUpScheduler(optimizer, warmup_steps=args.cont_warmup_steps)
    
    ema = deepcopy(score_model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False) 
    
    noise_cat = LogLinearNoise()
    if args.SDE == "VP-LinearBeta":
        noise_num = LinearBeta()
    elif args.SDE == "VP-LinearBeta2":
        noise_num = LinearBeta2()
    
    update_ema(ema, score_model, decay=0)
    score_model.train()
    ema.eval()
    args.logger.info(f"Model total number of parameters: {sum(p.numel() for p in score_model.parameters()):,}")
    args.logger.info(f"Start training...")
    
    early_eps = args.early_eps # 1e-3
    lambda_num, lambda_cat = args.lambda_num, args.lambda_cat
    args.logger.info(f">>> Training parameter <<< \n  early_eps: {early_eps} | lambda_num: {lambda_num} | lambda_cat: {lambda_cat}")
    
    total_batch_num = 0
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
        batch_loss = 0.0
        for batch_num, batch_cat in pbar:
            total_batch_num += 1

            optimizer.zero_grad()
            batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
            t = early_eps + (1 - early_eps) * torch.rand(batch_num.shape[0], 1).to(device) # No early stopping
            if args.two_time: 
                s = early_eps + (1 - early_eps) * torch.rand(batch_num.shape[0], 1).to(device)
            else:
                s = t.clone()
            dsigma_num, sigma_num = noise_num.rate_noise(t), noise_num.total_noise(t)
            dsigma_cat, sigma_cat = noise_cat.rate_noise(t), noise_cat.total_noise(t)
            
            if not args.imputation:
                move_chance_cat = 1 - (-sigma_cat).exp()
                stay_indices_cat = torch.rand(*batch_cat.shape, device=device) > move_chance_cat
                
                mask_tensor = torch.tensor(categories, dtype=torch.int, device=device)
                mask_tensor = mask_tensor.unsqueeze(0).expand_as(batch_cat)
                perturbed_batch_cat = torch.where(stay_indices_cat, batch_cat, mask_tensor)
                
                batch_num_noise = torch.randn_like(batch_num)
                
                perturbed_batch_num = torch.exp(-sigma_num).reshape(-1, 1) * batch_num + torch.sqrt(1 - torch.exp(-2 * sigma_num)).reshape(-1,1) * batch_num_noise # VP
                    
                score_num, score_cat_list = score_model(perturbed_batch_num, perturbed_batch_cat, t.reshape(-1), s.reshape(-1)) # Check t shape

                continous_loss, discrete_loss = multimodal_diffusion_loss(score_num, score_cat_list, batch_num_noise, batch_cat) 
                
                discrete_loss_final = torch.mean((dsigma_cat / torch.expm1(sigma_cat)).reshape(-1) * discrete_loss)
                loss = lambda_num * continous_loss + lambda_cat * discrete_loss_final
            else:
                if args.impute_cat:
                    dsigma_imp, sigma_imp = noise_cat.rate_noise(s), noise_cat.total_noise(s)
                    move_chance_cat = 1 - (-sigma_cat).exp()
                    stay_indices_cat = torch.rand(*batch_cat.shape, device=device) > move_chance_cat
                    
                    move_chance_imp = 1 - (-sigma_imp).exp()
                    stay_indices_cat[:, :1] = torch.rand(batch_cat.shape[0], 1, device=device) > move_chance_imp
                    mask_tensor = torch.tensor(categories, dtype=torch.int, device=device)
                    mask_tensor = mask_tensor.unsqueeze(0).expand_as(batch_cat)
                    perturbed_batch_cat = torch.where(stay_indices_cat, batch_cat, mask_tensor)
                    
                    batch_num_noise = torch.randn_like(batch_num)
                                        
                    perturbed_batch_num = torch.exp(-sigma_num).reshape(-1, 1) * batch_num + torch.sqrt(1 - torch.exp(-2 * sigma_num)).reshape(-1,1) * batch_num_noise # VP
                    
                    score_num, score_cat_list = score_model(perturbed_batch_num, perturbed_batch_cat, t.reshape(-1), s.reshape(-1)) # Check t shape
                    continous_loss, discrete_loss, discrete_loss_imp = multimodal_diffusion_loss_imp(score_num, score_cat_list, batch_num_noise, batch_cat)
                    discrete_loss_final = torch.mean((dsigma_cat / torch.expm1(sigma_cat)).reshape(-1) * discrete_loss + (dsigma_imp / torch.expm1(sigma_imp)).reshape(-1) * discrete_loss_imp)
                    loss = lambda_num * continous_loss + lambda_cat * discrete_loss_final
                    
                else:
                    dsigma_imp, sigma_imp = noise_num.rate_noise(s), noise_num.total_noise(s)
                    move_chance_cat = 1 - (-sigma_cat).exp()
                    stay_indices_cat = torch.rand(*batch_cat.shape, device=device) > move_chance_cat
                    
                    mask_tensor = torch.tensor(categories, dtype=torch.int, device=device)
                    mask_tensor = mask_tensor.unsqueeze(0).expand_as(batch_cat)
                    perturbed_batch_cat = torch.where(stay_indices_cat, batch_cat, mask_tensor)
                    
                    batch_num_noise = torch.randn_like(batch_num)

                    # VP
                    perturbed_batch_imp = torch.exp(-sigma_imp).reshape(-1, 1) * batch_num[:, :1] + torch.sqrt(1 - torch.exp(-2 * sigma_imp)).reshape(-1,1) * batch_num_noise[:, :1]
                    perturbed_batch_rest = torch.exp(-sigma_num).reshape(-1, 1) * batch_num[:, 1:] + torch.sqrt(1 - torch.exp(-2 * sigma_num)).reshape(-1,1) * batch_num_noise[:, 1:]
                    perturbed_batch_num = torch.cat((perturbed_batch_imp, perturbed_batch_rest), 1)
                        
                    score_num, score_cat_list = score_model(perturbed_batch_num, perturbed_batch_cat, t.reshape(-1), s.reshape(-1)) # Check t shape

                    continous_loss, discrete_loss = multimodal_diffusion_loss(score_num, score_cat_list, batch_num_noise, batch_cat) 
                    
                    discrete_loss_final = torch.mean((dsigma_cat / torch.expm1(sigma_cat)).reshape(-1) * discrete_loss)
                    loss = lambda_num * continous_loss + lambda_cat * discrete_loss_final
                        
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema(ema, score_model)
            pbar.set_postfix(loss=f"{loss.item():.4f}", num=f"{continous_loss.item():.4f}", cat=f"{discrete_loss_final.item():.4f}")
            batch_loss += loss.item()

            # args.logger.info(f"{total_batch_num} continous_loss {continous_loss.item()}")
            # args.logger.info(f"{total_batch_num} discrete_loss_final {discrete_loss_final.item()}")
            # args.logger.info(f"{total_batch_num} loss {loss.item()}")
        
        epoch_loss = batch_loss / len(train_loader)

        if epoch > 0 and (epoch + 1) % args.save_every == 0:
            args.logger.info("\t Epoch training loss: {:.5f}".format(epoch_loss))
            checkpoint = {
                        "model": score_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict()
                        }
            checkpoint_path = f"{args.experiment_dir}/model_{epoch:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            args.logger.info(f"{total_batch_num} continous_loss {continous_loss.item()}")
            args.logger.info(f"{total_batch_num} discrete_loss_final {discrete_loss_final.item()}")
            args.logger.info(f"{total_batch_num} loss {loss.item()}")
            args.logger.info(f"\t\t Saved checkpoint to {checkpoint_path}")
            last_epoch = epoch
        
    return last_epoch

def sampling(args):
    device = args.device

    # sample
    with open(f'data/{args.dataname}/info.json', 'r') as f:
        info = json.load(f)
    X_num, _, categories, d_numerical, num_inverse, cat_inverse = dataset.preprocess(args.data_dir, task_type = info['task_type'], inverse = True)
    X_train_num, _ = X_num
    num_samples = X_train_num.shape[0]

    if args.input_standardize:
        X_train_mean = torch.from_numpy(np.mean(X_train_num, axis=0))
        X_train_std  = torch.from_numpy(np.std(X_train_num, axis=0))
    else:
        X_train_mean = None
        X_train_std = None

    if args.sample_save_path == '':
        save_path = '/'.join(args.ckpt_path.split('/')[:-1])
    else:
        save_path = args.sample_save_path

    model = TabularDiT(categories, d_numerical, 
                       hidden_size = args.model_hidden_size, 
                       depth = args.model_depth, 
                       num_heads = args.model_num_heads, 
                       mlp_ratio = args.model_mlp_ratio,
                       two_time = args.two_time).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device)["ema"])
    model.eval()
    args.logger.info(f"Pretrained model loaded.\nGenerating samples for {args.ckpt_path}...")

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
            x_num_batch, x_cat_batch = sampler.sample(args.sample_batch_size, model, args.sampling_steps)
            x_num_final[(chunk_idx * args.sample_batch_size) : ((chunk_idx + 1) * args.sample_batch_size), :] = x_num_batch.detach().cpu().clone()
            x_cat_final[(chunk_idx * args.sample_batch_size) : ((chunk_idx + 1) * args.sample_batch_size), :] = x_cat_batch.detach().cpu().clone()
        # final batch
        if num_samples - (chunk_idx + 1) * args.sample_batch_size > 0:
            x_num_batch, x_cat_batch = sampler.sample(num_samples - (chunk_idx + 1) * args.sample_batch_size, model, args.sampling_steps)
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
    syn_df_bf_inv.to_csv(os.path.join(save_path, 'samples_bf_inv_{}.csv'.format(args.ckpt_path.split('/')[-1].split("_")[1].split(".")[0])), index=False)
    syn_df_af_inv.to_csv(os.path.join(save_path, 'samples_af_inv_{}.csv'.format(args.ckpt_path.split('/')[-1].split("_")[1].split(".")[0])), index=False)
    
    end_time = time.time()
    print('Time for sampling:', end_time - start_time)
    args.logger.info(f'Saving sampled data to {save_path}')

def main(args): 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    args.data_dir = f'data/{args.dataname}'

    os.makedirs(f"{args.result_dir}/", exist_ok=True)
    os.makedirs(f"{args.result_dir}/{args.dataname}", exist_ok=True)
    results_dir = os.path.join(args.result_dir, args.dataname)
    experiment_index = len(glob(f"{results_dir}/*"))
    args.experiment_dir = f"{results_dir}/{experiment_index:03d}"  # Create an experiment folder
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    
    # setup logging
    logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.experiment_dir}/log.txt")],
            force=True
        )
    args.logger = logging.getLogger(__name__)
    args.logger.info(f">>> Training param <<< \n  Batch size: {args.batch_size} | Epoch num: {args.num_epochs} | Dataset name: {args.dataname}")


    # start training #
    args.logger.info("Start training...")
    last_epoch = trainer(args)
    # end training #

    # start sampling
    args.ckpt_path= f"{args.experiment_dir}/model_{last_epoch:07d}.pt"
    args.logger.info("Start sampling...")
    sampling(args)
    # end sampling



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tabular DiT.')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    # dataset parameters
    parser.add_argument('--input_standardize', default=False, type=lambda x: (str(x).lower() == 'true'), help="If standardize the dataset.")
    
    # model parameters
    parser.add_argument('--model_hidden_size', type=int, default=24, help='TabularDiT, hidden size.')
    parser.add_argument('--model_depth', type=int, default=4, help='TabularDiT, depth.')
    parser.add_argument('--model_num_heads', type=int, default=4, help='TabularDiT, num_heads.')
    parser.add_argument('--model_mlp_ratio', type=int, default=4, help='TabularDiT, mlp_ratio.')
    

    # training parameters
    parser.add_argument('--early_eps', type=float, default=1e-3, help='Early stopping time.')
    parser.add_argument('--lambda_num', type=float, default=1, help='Weight before numerical loss.')
    parser.add_argument('--lambda_cat', type=float, default=0.1, help='Weight before categorical loss.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Training batch size.')
    parser.add_argument('--num_epochs', type=int, default=30000, help='Number of epochs.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--save_every', type=int, default=5000, help='Save every x epoch. Should be smaller than num_epochs!')
    parser.add_argument('--result_dir', type=str, default="./exp", help='Directory path to log and save model.')
    parser.add_argument('--SDE', type=str, default="VP-LinearBeta", choices=["VP-LinearBeta", "VP-LinearBeta2"], help='SDE type')
    parser.add_argument('--imputation', default=False, type=lambda x: (str(x).lower() == 'true'), help="Imputation task?")
    parser.add_argument('--continue_from', type=str, default="", help="Continue from ckpt?")
    parser.add_argument('--init_warmup_steps', type=int, default=200, help='Warmup steps when model is first initialized.')
    parser.add_argument('--cont_warmup_steps', type=int, default=10, help='Warmup steps when model is continued training.')
    
    # sampling parameters
    parser.add_argument('--sample_save_path', type=str, default='', help='Save path for samples.')
    parser.add_argument('--sample_batch_size', type=int, default=7000, help='Number of samples in one batch.')
    parser.add_argument('--sampling_steps', type=int, default=1000, help='Number of function evaluations.')

    # gpu ID
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    main(args)