import abc
import torch
import torch.nn as nn
import numpy as np


def get_noise(config):
    if config.noise.type == "geometric":
        return GeometricNoise(config.noise.sigma_min, config.noise.sigma_max)
    elif config.noise.type == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"{config.noise.type} is not a valid noise")


class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """
    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass


class GeometricNoise(Noise, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t

class PolynomialNoise(Noise, nn.Module):
    def __init__(self, sigma_min=0.002, sigma_max=80, rho = 7):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        self.rho = rho

    def rate_noise(self, t):
        coef = self.rho * (self.sigmas[1] ** (1/self.rho) - self.sigmas[0] ** (1/self.rho))
        t_part = (self.sigmas[0] ** (1/self.rho) + t * (self.sigmas[1] ** (1/self.rho) - self.sigmas[0] ** (1/self.rho))) ** (self.rho-1)
        return coef * t_part

    def total_noise(self, t):
         return (self.sigmas[0] ** (1/self.rho) + t * (self.sigmas[1] ** (1/self.rho) - self.sigmas[0] ** (1/self.rho))) ** self.rho

class LinearBeta(Noise, nn.Module):
    def __init__(self, beta_min = 0.1, beta_max = 20):
        super().__init__()
        self.betas = 1.0 * torch.tensor([beta_min, beta_max])

    def rate_noise(self, t):
        return self.betas[0] + t * (self.betas[1] - self.betas[0])

    def total_noise(self, t):
         return self.betas[0] * t + 0.5 * (self.betas[1] - self.betas[0]) * t ** 2
     
class LinearBeta2(Noise, nn.Module):
    def __init__(self, beta_min = 0.1, beta_max = 20):
        super().__init__()
        self.betas = 1.0 * torch.tensor([beta_min, beta_max])

    def rate_noise(self, t):
        return 1/2 * (self.betas[0] + t * (self.betas[1] - self.betas[0]))

    def total_noise(self, t):
         return 1/2 * (self.betas[0] * t + 0.5 * (self.betas[1] - self.betas[0]) * t ** 2)



class LogLinearNoise(Noise, nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

