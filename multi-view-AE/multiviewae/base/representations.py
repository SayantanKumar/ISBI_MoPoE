import torch
import torch.nn as nn

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

class alphaProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar, weights=None, eps=1e-8):
        if weights is None:
            num_components = mu.shape[0]
            weights = (1/num_components) * torch.ones(mu.shape).to(mu[0].device)
    
        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        weights = torch.broadcast_to(weights, mu.shape)
        pd_var = 1. / torch.sum(weights * T + eps, dim=0)
        pd_mu = pd_var * torch.sum(weights * mu * T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

class weightedProductOfExperts(nn.Module):
    """Return parameters for generalised product of independent experts with learnt weight.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    weight: Weight for each expert. M-1 x D for M experts 
    """
 
    def forward(self, mu, logvar, weight, eps=1e-8):
        var = torch.exp(logvar) + eps     
        weight_M = torch.ones(1, mu.shape[-1]).to(mu.device) - torch.sum(weight, dim=0)
        weight = torch.cat([weight, weight_M], dim=0)
        weight = weight[:, None, :].repeat(1, mu.shape[1],1)

        T = 1.0 / (var + eps)
        pd_mu = torch.sum(mu * weight * T, dim=0) / torch.sum(weight*T, dim=0)
        pd_var = 1.0 / torch.sum(weight*T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mus (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvars (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, logvars):

        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * torch.ones(num_components).to(mus[0].device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, logvar_sel

class MeanRepresentation(nn.Module):
    """Return mean of separate VAE representations.
    
    Args:
    mu (torch.Tensor): Mean of distributions. M x D for M views.
    logvar (torch.Tensor): Log of Variance of distributions. M x D for M views.
    """

    def forward(self, mu, logvar, eps=1e-8):
        mean_mu = torch.mean(mu, axis=0)
        mean_logvar = torch.mean(logvar, axis=0)
        
        return mean_mu, mean_logvar
