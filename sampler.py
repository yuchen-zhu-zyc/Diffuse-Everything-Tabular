import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import abc
from tqdm import tqdm

class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self):
        pass

def tau_leaping_cat_update(h, this_cat, prob, dsigma_cat, sigma_cat, device):
    batch_num = prob.shape[0]
    N = prob.shape[-1]
    prob[this_cat != N-1]=0
    prob.scatter_(-1, this_cat[..., None], torch.zeros_like(prob[..., :1])) # make prob[b,d,x[b][d]] = 0
    # prob[b][d][n] != 0 only if x[b][d] == mask and n != mask
    rate = dsigma_cat / torch.expm1(sigma_cat)
    qmat_cat = rate[..., None, None] * prob
    diffs = torch.arange(N).view(1, 1, N).to(device) - this_cat.view(batch_num, 1, 1)
    jump_nums = torch.distributions.poisson.Poisson(h * qmat_cat).sample().to(device)
    jump_nums[jump_nums.sum(dim = -1) > 1] = 0 
    overall_jump = torch.sum(jump_nums * diffs, dim=-1).to(device)
    this_cat = torch.clamp(this_cat + overall_jump, min=0, max = N-1).to(torch.int64).reshape((batch_num, 1))
    return this_cat

def euler_num_update_ve(h, x_num, dsigma_num, sigma_num, score_num):
    drift_num = 2 * dsigma_num[..., None] * score_num
    noise_coef_num = torch.sqrt(2 * dsigma_num[..., None] * sigma_num[:, None])
    x_num_next = x_num - h * drift_num + np.sqrt(h) * noise_coef_num * torch.randn_like(x_num)
    return x_num_next

def euler_pfode_num_update_ve(h, x_num, dsigma_num, sigma_num, score_num):
    drift_num = dsigma_num[..., None] * score_num
    x_num_next = x_num - h * drift_num
    return x_num_next

def euler_num_update_vp(h, x_num, dsigma_num, sigma_num, score_num):
    drift_num = -dsigma_num[..., None] * x_num + 2 * dsigma_num[..., None] * score_num / torch.sqrt(1 - torch.exp(-2 * sigma_num)).reshape(-1, 1) ## -beta_t x_t - 2 * beta_t sigma_t * score_t
    noise_coef_num = torch.sqrt(2 * dsigma_num)[..., None]
    x_num_next = x_num - h * drift_num + np.sqrt(h) * noise_coef_num * torch.randn_like(x_num)
    return x_num_next

def euler_pfode_num_update_vp(h, x_num, dsigma_num, sigma_num, score_num):
    drift_num = -dsigma_num[..., None] * x_num + dsigma_num[..., None] * score_num / torch.sqrt(1 - torch.exp(-2 * sigma_num)).reshape(-1, 1) ## -beta_t x_t - 2 * beta_t sigma_t * score_t
    x_num_next = x_num - h * drift_num
    return x_num_next

class EM_TauLeap_Sampler_VE(BaseSampler):
    def __init__(self, noise_num, noise_cat, categories, d_numerical, device, pfode = False):
        self.noise_num = noise_num
        self.noise_cat = noise_cat
        self.device = device
        self.categories = torch.tensor(categories, dtype=torch.int64).reshape(1, -1)
        self.d_numerical = d_numerical
        self.pfode = pfode
        self.eps = 1e-3
    
    def sample(self, batch_num, score_model, steps):
        x_cat_start = self.categories.repeat(batch_num, 1).to(self.device)
        x_num_start = self.noise_num.sigmas[1] * torch.randn((batch_num, self.d_numerical)).to(self.device)
        
        x_cat = x_cat_start
        x_num = x_num_start
        batch_num = x_cat_start.shape[0]
        category_num = x_cat_start.shape[1]
        ts = np.linspace(1, self.eps, steps) 
        for idx in range(steps - 1):
            t = ts[idx]
            h = ts[idx] - ts[idx + 1]
            t_ten = torch.full((batch_num,), t).to(self.device)
            dsigma_num, sigma_num = self.noise_num.rate_noise(t_ten), self.noise_num.total_noise(t_ten)
            dsigma_cat, sigma_cat = self.noise_cat.rate_noise(t_ten), self.noise_cat.total_noise(t_ten)
            score_num, score_cat_list = score_model(x_num, x_cat, t_ten, t_ten)
            if self.pfode:
                x_num_next = euler_pfode_num_update_ve(h, x_num, dsigma_num, sigma_num, score_num)
            else:
                x_num_next = euler_num_update_ve(h, x_num, dsigma_num, sigma_num, score_num)
            x_cat_next = []
            
            for cat_idx in range(category_num):
                prob = torch.softmax(score_cat_list[cat_idx], dim=-1).reshape(batch_num, 1, -1)
                this_cat = x_cat[:, cat_idx].reshape(batch_num, 1)
                this_cat = tau_leaping_cat_update(h, this_cat, prob, dsigma_cat, sigma_cat, self.device)
                x_cat_next.append(this_cat)
            x_cat_next = torch.cat(x_cat_next, dim=-1)    
            x_num, x_cat = x_num_next, x_cat_next
            
        ## Final Step
        t = self.eps
        h = self.eps
        t_ten = torch.full((batch_num,), t).to(self.device)
        dsigma_num, sigma_num = self.noise_num.rate_noise(t_ten), self.noise_num.total_noise(t_ten)
        dsigma_cat, sigma_cat = self.noise_cat.rate_noise(t_ten), self.noise_cat.total_noise(t_ten)
        score_num, score_cat_list = score_model(x_num, x_cat, t_ten, t_ten)
        if self.pfode:
            x_num_final = euler_pfode_num_update_ve(h, x_num, dsigma_num, sigma_num, score_num)
        else:
            x_num_final = euler_num_update_ve(h, x_num, dsigma_num, sigma_num, score_num)
        
        final_cat_list = []
        for cat_idx in range(category_num):
            prob = torch.softmax(score_cat_list[cat_idx], dim=-1).reshape(batch_num, 1, -1)
            this_cat = x_cat[:, cat_idx].reshape(batch_num, 1)
            N = prob.shape[-1]
            masked = this_cat == N - 1
            this_cat[masked]=torch.argmax(prob[:,:,:-1], dim=-1)[masked] # exclude the mask token
            final_cat_list.append(this_cat)
        x_cat_final = torch.cat(final_cat_list, dim=-1)
        
        return x_num_final, x_cat_final

class Heun_TauLeap_Sampler_VE(BaseSampler):
    def __init__(self, noise_num, noise_cat, categories, d_numerical, device):
        self.noise_num = noise_num
        self.noise_cat = noise_cat
        self.device = device
        self.categories = torch.tensor(categories, dtype=torch.int64).reshape(1, -1)
        self.d_numerical = d_numerical
        self.eps = 1e-3
    
    def sample(self, batch_num, score_model, steps):
        x_cat_start = self.categories.repeat(batch_num, 1).to(self.device)
        x_num_start = self.noise_num.sigmas[1] * torch.randn((batch_num, self.d_numerical)).to(self.device)
        
        x_cat = x_cat_start
        x_num = x_num_start
        batch_num = x_cat_start.shape[0]
        category_num = x_cat_start.shape[1]
        ts = np.linspace(1, self.eps, steps) 
        for idx in range(steps - 1):
            t = ts[idx]
            h = ts[idx] - ts[idx + 1]
            t_ten = torch.full((batch_num,), t).to(self.device)
            t_ten_next = torch.full((batch_num,), ts[idx + 1]).to(self.device)
            dsigma_num, sigma_num = self.noise_num.rate_noise(t_ten), self.noise_num.total_noise(t_ten)
            dsigma_cat, sigma_cat = self.noise_cat.rate_noise(t_ten), self.noise_cat.total_noise(t_ten)
            score_num, score_cat_list = score_model(x_num, x_cat, t_ten, t_ten)
            x_num_mid = euler_pfode_num_update_ve(h, x_num, dsigma_num, sigma_num, score_num)
            
            dsigma_num_mid, sigma_num_mid = self.noise_num.rate_noise(t_ten_next), self.noise_num.total_noise(t_ten_next)
            score_num_mid, score_cat_list_mid = score_model(x_num_mid, x_cat, t_ten_next, t_ten_next)
            drift_mid = dsigma_num_mid[..., None] * score_num_mid
            x_num_next = x_num - h * 0.5 * (dsigma_num[..., None] * score_num + drift_mid)
            
            x_cat_next = []
            
            for cat_idx in range(category_num):
                prob = torch.softmax(score_cat_list[cat_idx], dim=-1).reshape(batch_num, 1, -1)
                this_cat = x_cat[:, cat_idx].reshape(batch_num, 1)
                N = prob.shape[-1]
                this_cat = tau_leaping_cat_update(h, this_cat, prob, dsigma_cat, sigma_cat, self.device)
                x_cat_next.append(this_cat)
            x_cat_next = torch.cat(x_cat_next, dim=-1)    
            x_num, x_cat = x_num_next, x_cat_next
            
        ## Final Step
        t = self.eps
        h = self.eps
        t_ten = torch.full((batch_num,), t).to(self.device)
        dsigma_num = self.noise_num.rate_noise(t_ten)
        dsigma_cat, sigma_cat = self.noise_cat.rate_noise(t_ten), self.noise_cat.total_noise(t_ten)
        score_num, score_cat_list = score_model(x_num, x_cat, t_ten, t_ten)

        x_num_final = x_num
        
        final_cat_list = []
        for cat_idx in range(category_num):
            prob = torch.softmax(score_cat_list[cat_idx], dim=-1).reshape(batch_num, 1, -1)
            this_cat = x_cat[:, cat_idx].reshape(batch_num, 1)
            N = prob.shape[-1]
            masked = this_cat == N - 1
            this_cat[masked]=torch.argmax(prob[:,:,:-1], dim=-1)[masked] # exclude the mask token
            final_cat_list.append(this_cat)
        x_cat_final = torch.cat(final_cat_list, dim=-1)
        
        return x_num_final, x_cat_final

    
class EM_TauLeap_Sampler_VP(BaseSampler):
    def __init__(self, noise_num, noise_cat, categories, d_numerical, device, pfode = False):
        self.noise_num = noise_num
        self.noise_cat = noise_cat
        self.device = device
        self.categories = torch.tensor(categories, dtype=torch.int64).reshape(1, -1)
        self.d_numerical = d_numerical
        self.pfode = pfode
        self.eps = 1e-3
        
    def sample(self, batch_num, score_model, steps):
        x_cat_start = self.categories.repeat(batch_num, 1).to(self.device)
        x_num_start = torch.randn((batch_num, self.d_numerical)).to(self.device)
        
        x_cat = x_cat_start
        x_num = x_num_start
        batch_num = x_cat_start.shape[0]
        category_num = x_cat_start.shape[1]
        ts = np.linspace(1, self.eps, steps) 
        for idx in range(steps - 1):
            t = ts[idx]
            h = ts[idx] - ts[idx + 1]
            t_ten = torch.full((batch_num,), t).to(self.device)
            dsigma_num, sigma_num = self.noise_num.rate_noise(t_ten), self.noise_num.total_noise(t_ten)
            dsigma_cat, sigma_cat = self.noise_cat.rate_noise(t_ten), self.noise_cat.total_noise(t_ten)
            score_num, score_cat_list = score_model(x_num, x_cat, t_ten, t_ten)
            if self.pfode:
                x_num_next = euler_pfode_num_update_vp(h, x_num, dsigma_num, sigma_num, score_num)
            else:   
                x_num_next = euler_num_update_vp(h, x_num, dsigma_num, sigma_num, score_num)
                
            x_cat_next = []
            for cat_idx in range(category_num):
                prob = torch.softmax(score_cat_list[cat_idx], dim=-1).reshape(batch_num, 1, -1)
                this_cat = x_cat[:, cat_idx].reshape(batch_num, 1)
                this_cat = tau_leaping_cat_update(h, this_cat, prob, dsigma_cat, sigma_cat, self.device)
                x_cat_next.append(this_cat)
                
            x_cat_next = torch.cat(x_cat_next, dim=-1)    
            x_num, x_cat = x_num_next, x_cat_next
            
        ## Final Step
        t = self.eps
        h = self.eps
        t_ten = torch.full((batch_num,), t).to(self.device)
        dsigma_num, sigma_num = self.noise_num.rate_noise(t_ten), self.noise_num.total_noise(t_ten)
        dsigma_cat, sigma_cat = self.noise_cat.rate_noise(t_ten), self.noise_cat.total_noise(t_ten)
        score_num, score_cat_list = score_model(x_num, x_cat, t_ten, t_ten)
        if self.pfode:
            x_num_final = euler_pfode_num_update_vp(h, x_num, dsigma_num, sigma_num, score_num)
        else:
            x_num_final = euler_num_update_vp(h, x_num, dsigma_num, sigma_num, score_num)
        
        final_cat_list = []
        for cat_idx in range(category_num):
            prob = torch.softmax(score_cat_list[cat_idx], dim=-1).reshape(batch_num, 1, -1)
            this_cat = x_cat[:, cat_idx].reshape(batch_num, 1)
            N = prob.shape[-1]
            masked = this_cat == N - 1
            this_cat[masked] = torch.argmax(prob[:,:,:-1], dim=-1)[masked] # exclude the mask token
            final_cat_list.append(this_cat)
        x_cat_final = torch.cat(final_cat_list, dim=-1)
        
        return x_num_final, x_cat_final

