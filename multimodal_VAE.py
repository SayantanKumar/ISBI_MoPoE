import numpy as np
import pandas as pd
import os
import warnings
import ggseg_python
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import neuroHarmonize
from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel, saveHarmonizationModel

import seaborn as sns
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, StratifiedKFold

import scipy
import statsmodels
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import fdrcorrection_twostage

from data_utils import *
from ADNI_KARI_merge_compare import *
from harmonize_combat import *

from dataloaders import *

#--------------------------------------------------------------
#--------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

def one_hot_encoding(table, col): # returns one hot encoding matrix
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(table[col].values)
    #print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    
    return onehot_encoded
    #return (to_categorical(table.Age_group.to_numpy()))


def min_max_scaling(train_df, val_df):
    
    train_df_scaled = MinMaxScaler().fit(train_df).transform(train_df)
    val_df_scaled = MinMaxScaler().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled


def standard_scaling(train_df, val_df):
    
    train_df_scaled = StandardScaler().fit(train_df).transform(train_df)
    val_df_scaled = StandardScaler().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled


def robust_scaling(train_df, val_df):
    
    train_df_scaled = RobustScaler().fit(train_df).transform(train_df)
    val_df_scaled = RobustScaler().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled


def quantile_transformer_scaling(train_df, val_df):
    
    train_df_scaled = QuantileTransformer().fit(train_df).transform(train_df)
    val_df_scaled = QuantileTransformer().fit(train_df).transform(val_df)

    return train_df_scaled, val_df_scaled

#--------------------------------------------------------------
#--------------------------------------------------------------


# Multi-modal Variational Autoencoder ------ 3 modalities
#https://arxiv.org/pdf/1802.05335.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
import random

class MVAE(nn.Module): # @param n_latents : number of latent dimensions
    
    def __init__(self, n_latents, m1_data_shape, m2_data_shape, m3_data_shape, cond_shape):
        
        super(MVAE, self).__init__()
        
        self.cort_encoder = cort_encoder(n_latents, m1_data_shape, cond_shape)
        self.subcort_encoder = subcort_encoder(n_latents, m2_data_shape, cond_shape)
        self.hcm_encoder = hcm_encoder(n_latents, m3_data_shape, cond_shape)
        
        self.cort_decoder = cort_decoder(n_latents, m1_data_shape, cond_shape)
        self.subcort_decoder = subcort_decoder(n_latents, m2_data_shape, cond_shape)
        self.hcm_decoder = hcm_decoder(n_latents, m3_data_shape, cond_shape)
        
        self.experts = ProductOfExperts()
        self.n_latents = n_latents
        self.m1_data_shape = m1_data_shape
        self.m2_data_shape = m2_data_shape
        self.m3_data_shape = m3_data_shape
        self.cond_shape = cond_shape
    
    def reparametrize(self, mu, logvar):
            
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
            
        else:
            return mu
            
        
        
    def forward(self, cort, subcort, hcm, age_cond):
            
        mu, logvar = self.infer(cort, subcort, hcm, age_cond)
        z = self.reparametrize(mu, logvar)
        #z_cond = torch.cat((z, age_cond), dim= -1)
        cort_recon = self.cort_decoder(z, age_cond)
        subcort_recon = self.subcort_decoder(z, age_cond)
        hcm_recon = self.hcm_decoder(z, age_cond)
            
        return cort_recon, subcort_recon, hcm_recon, mu, logvar
        
        
    def infer(self, cort, subcort, hcm, age_cond): 
            
        batch_size = cort.size(0) if cort is not None else subcort.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents), 
                                      use_cuda=use_cuda)
        
        if cort is not None:
            cort_mu, cort_logvar = self.cort_encoder(cort, age_cond)
            mu     = torch.cat((mu, cort_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, cort_logvar.unsqueeze(0)), dim=0)
            
            
        if subcort is not None:
            subcort_mu, subcort_logvar = self.subcort_encoder(subcort, age_cond)
            mu     = torch.cat((mu, subcort_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, subcort_logvar.unsqueeze(0)), dim=0)
        
        if hcm is not None:
            hcm_mu, hcm_logvar = self.hcm_encoder(hcm, age_cond)
            mu     = torch.cat((mu, hcm_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, hcm_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
            
        return mu, logvar
            
            
    
class cort_encoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m1_data_shape, cond_shape):
        super(cort_encoder, self).__init__()
        input_shape = m1_data_shape + cond_shape
        self.fc1   = nn.Linear(input_shape, 64)
        self.fc2   = nn.Linear(64, 32)
#         self.fc3   = nn.Linear(128, 256)
#         self.fc4   = nn.Linear(256, 512)
        self.fc51  = nn.Linear(32, n_latents)
        self.fc52  = nn.Linear(32, n_latents)
        self.relu = Relu()

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        #h = self.swish(self.fc1(x.view(-1, 99)))
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
#         h = self.relu(self.fc3(h))
#         h = self.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)




    
class cort_decoder(nn.Module):
    
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m1_data_shape, cond_shape):
        super(cort_decoder, self).__init__()
        decoder_shape = n_latents + cond_shape
        self.fc1   = nn.Linear(decoder_shape, 32)
        self.fc2   = nn.Linear(32, 64)
#         self.fc3   = nn.Linear(256, 128)
#         self.fc4   = nn.Linear(128, 64)
        self.fc5   = nn.Linear(64, m1_data_shape)
        self.relu = Relu()
        self.sigmoid = Sigmoid()

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
#         h = self.relu(self.fc3(h))
#         h = self.relu(self.fc4(h))
        return self.sigmoid(self.fc5(h))  # NOTE: no sigmoid here. See train.py


    

class subcort_encoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m2_data_shape, cond_shape):
        super(subcort_encoder, self).__init__()
        input_shape = m2_data_shape + cond_shape
        self.fc1   = nn.Linear(input_shape, 64)
        self.fc2   = nn.Linear(64, 32)
#         self.fc3   = nn.Linear(128, 256)
#         self.fc4   = nn.Linear(256, 512)
        self.fc51  = nn.Linear(32, n_latents)
        self.fc52  = nn.Linear(32, n_latents)
        self.relu = Relu()

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        #h = self.swish(self.fc1(x.view(-1, 99)))
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
#         h = self.relu(self.fc3(h))
#         h = self.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)



    
class subcort_decoder(nn.Module):
    
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m2_data_shape, cond_shape):
        super(subcort_decoder, self).__init__()
        decoder_shape = n_latents + cond_shape
        self.fc1   = nn.Linear(decoder_shape, 32)
        self.fc2   = nn.Linear(32, 64)
#         self.fc3   = nn.Linear(256, 128)
#         self.fc4   = nn.Linear(128, 64)
        self.fc5   = nn.Linear(64, m2_data_shape)
        self.relu = Relu()
        self.sigmoid = Sigmoid()

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
#         h = self.relu(self.fc3(h))
#         h = self.relu(self.fc4(h))
        return self.sigmoid(self.fc5(h))  # NOTE: no sigmoid here. See train.py

    
    
    
    
class hcm_encoder(nn.Module):
    """Parametrizes q(z|y).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m3_data_shape, cond_shape):
        super(hcm_encoder, self).__init__()
        input_shape = m3_data_shape + cond_shape
        self.fc1   = nn.Linear(input_shape, 64)
        self.fc2   = nn.Linear(64, 32)
#         self.fc3   = nn.Linear(128, 256)
#         self.fc4   = nn.Linear(256, 512)
        self.fc51  = nn.Linear(32, n_latents)
        self.fc52  = nn.Linear(32, n_latents)
        self.relu = Relu()

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
#         h = self.relu(self.fc3(h))
#         h = self.relu(self.fc4(h))
        return self.fc51(h), self.fc52(h)

    
    

class hcm_decoder(nn.Module):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, m3_data_shape, cond_shape):
        super(hcm_decoder, self).__init__()
        decoder_shape = n_latents + cond_shape
        self.fc1   = nn.Linear(decoder_shape, 32)
        self.fc2   = nn.Linear(32, 64)
#         self.fc3   = nn.Linear(256, 128)
#         self.fc4   = nn.Linear(128, 64)
        self.fc5   = nn.Linear(64, m3_data_shape)
        self.relu = Relu()
        self.sigmoid = Sigmoid()

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
#         h = self.relu(self.fc3(h))
#         h = self.relu(self.fc4(h))
        return self.sigmoid(self.fc5(h))  # NOTE: no softmax here. See train.py


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)
      
class Sigmoid(nn.Module):
    def forward(self, x):
        return F.sigmoid(x)
    
class Relu(nn.Module):
    def forward(self, x):
        return F.relu(x)
    
class Tanh(nn.Module):
    def forward(self, x):
        return F.tanh(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar  



#--------------------------------------------------------------
#--------------------------------------------------------------


def elbo_loss(cort_recon, cort, subcort_recon, subcort, hcm_recon, hcm, mu, logvar, lambda_m1 = 1.0, lambda_m2 = 1.0, lambda_m3 = 1.0, beta = 1):
    
    cort_recon_mse, subcort_recon_mse, hcm_mse = 0,0,0
    mse_loss = torch.nn.MSELoss(reduction = 'mean')
    
    if cort_recon is not None and cort is not None:
        cort_recon_mse = mse_loss(cort, cort_recon)
        cort_recon_mse *= cort.shape[1]
        
        
    if subcort_recon is not None and subcort is not None:
        subcort_recon_mse = mse_loss(subcort, subcort_recon)
        subcort_recon_mse *= subcort.shape[1]
        
        
    if hcm_recon is not None and hcm is not None:
        hcm_mse = mse_loss(hcm, hcm_recon)
        hcm_mse *= hcm.shape[1]
        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_m1 * cort_recon_mse + lambda_m2 * subcort_recon_mse + lambda_m3 * hcm_mse + beta * KLD)
    
    return ELBO


#--------------------------------------------------------------
#--------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
#--------------------------------------------------------------
#--------------------------------------------------------------


def train_val(model, train_loader, val_loader, epoch, optimizer, params):
    
    model.train()
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    for batch_idx, (X_train_m1, X_train_m2, X_train_m3, age_cond_train) in enumerate(train_loader):

        m1_train = Variable(X_train_m1)
        m2_train = Variable(X_train_m2)
        m3_train = Variable(X_train_m3)
        age_cond_train = Variable(age_cond_train)
        #y_train = Variable(y_train)
        
        batch_size = len(m1_train)

        optimizer.zero_grad()

        recon_m1_train, recon_m2_train, recon_m3_train, mu_train, logvar_train = model(m1_train, m2_train, m3_train, age_cond_train)

        joint_loss_train = elbo_loss(recon_m1_train, m1_train, recon_m2_train, m2_train, recon_m3_train, m3_train, mu_train, logvar_train, lambda_m1 = params['alpha_1'], lambda_m2 = params['alpha_2'], lambda_m3 = params['alpha_3'], beta = params['beta'])
    
        train_loss = joint_loss_train
        train_loss_meter.update(train_loss.data, batch_size)

        train_loss.backward()
        optimizer.step()
        
    model.eval()
        
    for batch_idx, (X_val_m1, X_val_m2, X_val_m3, age_cond_val) in enumerate(val_loader):
        
        m1_val = Variable(X_val_m1, volatile = True)
        m2_val = Variable(X_val_m2, volatile = True)
        m3_val = Variable(X_val_m3, volatile = True)
        
        age_cond_val = Variable(age_cond_val, volatile = True)
        batch_size = len(m1_val)
        
        recon_m1_val, recon_m2_val, recon_m3_val, mu_val, logvar_val = model(m1_val, m2_val, m3_val, age_cond_val)
        
        joint_loss_val = elbo_loss(recon_m1_val, m1_val, recon_m2_val, m2_val, recon_m3_val, m3_val, mu_val, logvar_val, lambda_m1 = params['alpha_1'], lambda_m2 = params['alpha_2'], beta = params['beta'])
    
        val_loss = joint_loss_val
        val_loss_meter.update(val_loss.data, batch_size)

    #print('====> Epoch: {}\t Train Loss: {:.4f} \t Val Loss: {:.4f}'.format(epoch, train_loss_meter.avg, val_loss_meter.avg))
    
    return train_loss_meter.avg, val_loss_meter.avg, recon_m1_val, recon_m2_val, recon_m3_val



#--------------------------------------------------------------
#--------------------------------------------------------------
  

def test_mvae(test_loader, model):
    
    model.eval()
        
    for batch_idx, (X_test_m1, X_test_m2, X_test_m3, age_cond_test) in enumerate(test_loader):
        
        m1_test = Variable(X_test_m1, volatile = True)
        m2_test = Variable(X_test_m2, volatile = True)
        m3_test = Variable(X_test_m3, volatile = True)
        
        age_cond_test = Variable(age_cond_test, volatile = True)
        batch_size = len(m1_test)
        
        recon_m1_test, recon_m2_test, recon_m3_test, mu_val_test, logvar_val_test = model(m1_test, m2_test, m3_test, age_cond_test)
        
    return recon_m1_test, recon_m2_test, recon_m3_test


#--------------------------------------------------------------
#--------------------------------------------------------------


def latent_space_mmvae(test_loader, model):
    
    model.eval()
        
    for batch_idx, (X_test_m1, X_test_m2, X_test_m3, age_cond_test) in enumerate(test_loader):
        
        m1_test = Variable(X_test_m1, volatile = True)
        m2_test = Variable(X_test_m2, volatile = True)
        m3_test = Variable(X_test_m3, volatile = True)
        
        age_cond_test = Variable(age_cond_test, volatile = True)
        batch_size = len(m1_test)
        
        mu, logvar = model.infer(m1_test, m2_test, m3_test, age_cond_test)
        z = model.reparametrize(mu, logvar)
        
        return z, mu, logvar
    
#--------------------------------------------------------------
#--------------------------------------------------------------


def feature_space_mmvae(test_loader, model):
    
    model.eval()
        
    for batch_idx, (X_test_m1, X_test_m2, X_test_m3, age_cond_test) in enumerate(test_loader):
        
        m1_test = Variable(X_test_m1, volatile = True)
        m2_test = Variable(X_test_m2, volatile = True)
        m3_test = Variable(X_test_m3, volatile = True)
        
        age_cond_test = Variable(age_cond_test, volatile = True)
        batch_size = len(m1_test)
        
        mu, logvar = model.infer(m1_test, m2_test, m3_test, age_cond_test)
        z = model.reparametrize(mu, logvar)
        recon_m1 = model.cort_decoder(z, age_cond_test)
        recon_m2 = model.subcort_decoder(z, age_cond_test)
        recon_m3 = model.hcm_decoder(z, age_cond_test)
        
        return recon_m1, recon_m2, recon_m3
        

#--------------------------------------------------------------
#--------------------------------------------------------------


def calculate_deviations(X_org_val, X_pred_val, fs_cols, X_test_org, X_pred_test):
    
    mean_CN = []
    std_CN = []

#     X_org_val_m1 = pd.DataFrame(X_val_m1, columns = m1_cols)
#     X_org_val_m2 = pd.DataFrame(X_val_m2, columns = m2_cols)
#     X_org_val = pd.concat([X_org_val_m1[m1_cols], X_org_val_m2[m2_cols]], axis = 1)

#     X_pred_val_m1 = pd.DataFrame(recon_m1_val.detach().numpy(), columns = m1_cols)
#     X_pred_val_m2 = pd.DataFrame(recon_m2_val.detach().numpy(), columns = m2_cols)
#     X_pred_val = pd.concat([X_pred_val_m1[m1_cols], X_pred_val_m2[m2_cols]], axis = 1)

    X_cn_diff = (X_org_val.reset_index(drop = True)[fs_cols] - X_pred_val.reset_index(drop = True)[fs_cols])

    for col in fs_cols:
        mean_CN.append(X_cn_diff[col].mean())
        std_CN.append(X_cn_diff[col].std())


    mean_df = pd.DataFrame(mean_CN)
    mean_df.index = fs_cols

    std_df = pd.DataFrame(std_CN)
    std_df.index = fs_cols


    ##--------------------------------------------------------------------

    dev_bvae = X_test_org.copy()
    diff = X_test_org.copy()

    diff[fs_cols] = ((X_test_org.reset_index(drop = True)[fs_cols] - X_pred_test.reset_index(drop = True)[fs_cols]))

    for idx in diff.index.values:
        for col in fs_cols:
            dev_bvae.loc[idx, col] = (diff.loc[idx][col] - mean_df.loc[col]).at[0]/(std_df.loc[col].at[0])

            
    return diff, dev_bvae, mean_df, std_df


#--------------------------------------------------------------
#--------------------------------------------------------------

def train_model_MVAE(X_train_m1, X_train_m2, X_train_m3, X_val_m1, X_val_m2, X_val_m3, train_age_group, val_age_group, m1_cols, m2_cols, m3_cols, params, retrain = False):
    
    train_data = train_dataloader(X_train_m1, X_train_m2, X_train_m3, train_age_group.values)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = params['batch_size'] ,shuffle=False)

    val_data = val_dataloader(X_val_m1, X_val_m2, X_val_m3, val_age_group.values)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = params['batch_size'],shuffle=False)

    # for batch_idx, (X_train_m1, X_train_m2) in enumerate(train_loader):
    #     print(batch_idx, X_train_m1.shape, X_train_m2.shape)

    m1_data_shape = X_train_m1.shape[1] 
    m2_data_shape = X_train_m2.shape[1] 
    m3_data_shape = X_train_m3.shape[1]
    cond_shape = train_age_group.shape[1]

    # X = Input(shape=(X_train.shape[1],))
    # age_cond = Input(shape=(train_age_group.shape[1],))
    # encoder_inputs = concat([X, age_cond])

    
    if retrain == True:
        model = torch.load('./saved_models/trained_MVAE_UKB')
        optimizer = optim.Adam(model.parameters(), lr = params['lr'])
    else:
        model = MVAE(params['latent_dim'], m1_data_shape, m2_data_shape, m3_data_shape, cond_shape)
        optimizer = optim.Adam(model.parameters(), lr = params['lr'])
    
    train_loss = []
    val_loss = []
    #best_loss_value = 1000
    for epoch in range(1, params['epochs'] + 1):
        train_loss_value, val_loss_value, recon_m1_val, recon_m2_val, recon_m3_val = train_val(model, train_loader, val_loader, epoch, optimizer, params)
        train_loss.append(float(train_loss_value.numpy()))
        val_loss.append(float(val_loss_value.numpy()))
        
        #print('Epoch = {}, training loss = {}, validation loss = {}'.format(epoch, train_loss_value, val_loss_value))

    X_org_val_m1 = pd.DataFrame(X_val_m1, columns = m1_cols)
    X_org_val_m2 = pd.DataFrame(X_val_m2, columns = m2_cols)
    X_org_val_m3 = pd.DataFrame(X_val_m3, columns = m3_cols)
    
    X_org_val = pd.concat([X_org_val_m1[m1_cols], X_org_val_m2[m2_cols], X_org_val_m3[m3_cols]], axis = 1)

    X_pred_val_m1 = pd.DataFrame(recon_m1_val.detach().numpy(), columns = m1_cols)
    X_pred_val_m2 = pd.DataFrame(recon_m2_val.detach().numpy(), columns = m2_cols)
    X_pred_val_m3 = pd.DataFrame(recon_m3_val.detach().numpy(), columns = m3_cols)
    
    X_pred_val = pd.concat([X_pred_val_m1[m1_cols], X_pred_val_m2[m2_cols], X_pred_val_m3[m3_cols]], axis = 1)
    
    return model, train_loss, val_loss, X_org_val, X_pred_val


#--------------------------------------------------------------
#--------------------------------------------------------------


def plot_mean_deviation(Z_score_all, fs_cols):
    
    Z_score_all['Mean deviation'] = (abs((Z_score_all[fs_cols])).sum(axis = 1).to_numpy()/Z_score_all[fs_cols].shape[1])

    #plt.figure(figsize = (10,7))

    sns.boxplot(x = 'DX_bl', y = 'Mean deviation', data = Z_score_all, order = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('Disease category', fontsize = 18)
    plt.ylabel('Mean deviation across \n all brain regions', fontsize = 18)
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)

    
#--------------------------------------------------------------
#--------------------------------------------------------------

    
def complete_training_prediction(CN_model, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, common_cols, age_sex_site_df, params):
    
    model, train_loss, val_loss, X_org_val, X_pred_val, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3 = model_training(CN_model, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, common_cols, age_sex_site_df, params)
    
    dev_bvae, X_valho_org, X_pred_valho, X_pred_test, X_test_total = model_inference(model, X_test_org, CN_held_val, common_cols, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, age_sex_site_df, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3)
    #dev_mvae, recon = mean_deviation_plots(X_test_total, X_pred_test, dev_bvae, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, train_loss, val_loss)
    
    return model, train_loss, val_loss, X_org_val, X_pred_val, dev_bvae, X_valho_org, X_pred_valho, X_pred_test, X_test_total
    
    
#-------------------------------------------------------------
#-------------------------------------------------------------

def model_training(CN_model, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, common_cols, age_sex_site_df, params):
    
    # Prepare training data (healthy controls)
    X_train_m1, X_train_m2, X_train_m3, X_val_m1, X_val_m2, X_val_m3, train_age_group, val_age_group, m1_cols, m2_cols, m3_cols, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3 = create_training_data_MVAE(CN_model.reset_index(drop = True), MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, age_sex_site_df, common_cols)

    # Train mmVAE model on healthy controls of ADNI + KARI 
    model, train_loss, val_loss, X_org_val, X_pred_val = train_model_MVAE(X_train_m1, X_train_m2, X_train_m3, X_val_m1, X_val_m2, X_val_m3, train_age_group, val_age_group, m1_cols, m2_cols, m3_cols, params, retrain = False)
    
#     plt.figure(figsize = (7,5))
#     num_epochs = range(len(train_loss))
#     plt.plot(num_epochs, train_loss, label='Training loss')
#     plt.plot(epochs, val_loss, label='Validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Training and validation loss')
#     plt.legend()
#     plt.title('params = {}'.format(params))

    return model, train_loss, val_loss, X_org_val, X_pred_val, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3

#---------------------------------------------------------------
#---------------------------------------------------------------

def model_inference(model, X_test_org, CN_held_val, common_cols, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, age_sex_site_df, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3):
    
    # # Prepare test data (all patients other than healthy controls)
    test_loader, X_test_total, test_age_group, X_test_m1, X_test_m2, X_test_m3 = create_test_data_MVAE(X_test_org, common_cols, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, age_sex_site_df, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3)

    # Apply mmVAE on test data
    recon_m1_test, recon_m2_test, recon_m3_test = test_mvae(test_loader, model)
    X_pred_test = concat_pred_modality(X_test_m1, X_test_m2, X_test_m3, X_test_total, recon_m1_test, recon_m2_test, recon_m3_test, common_cols, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)

    # Held out validation cohort of healthy controls for normalizing deviations
    X_org_ho_val =  CN_held_val.copy()
    val_ho_loader, X_valho_org, val_ho_age_group, X_valho_m1, X_valho_m2, X_valho_m3 = create_valho_data_MVAE(X_org_ho_val, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, age_sex_site_df, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3)
    recon_m1_valho, recon_m2_valho, recon_m3_valho = test_mvae(val_ho_loader, model)
    X_pred_valho = concat_pred_modality_valho(X_valho_m1, X_valho_m2, X_valho_m3, X_valho_org, recon_m1_valho, recon_m2_valho, recon_m3_valho, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)

    # Calculating normalized deviations (Z-scores)
    fs_cols = MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols
    diff, dev_bvae, mean_df, std_df = calculate_deviations(X_valho_org, X_pred_valho, fs_cols, X_test_total, X_pred_test)

    return dev_bvae, X_valho_org, X_pred_valho, X_pred_test, X_test_total


#--------------------------------------------------------------
#------------------------------------------------------------- 


def mean_deviation_plots(X_test_total, X_pred_test, dev_bvae, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, train_loss, val_loss):
    
    recon = X_test_total.copy()
    recon[MRI_vol_cols] = (X_test_total[MRI_vol_cols] - X_pred_test[MRI_vol_cols])**2
    recon[amyloid_SUVR_cols] = (X_test_total[amyloid_SUVR_cols] - X_pred_test[amyloid_SUVR_cols])**2
    recon[tau_SUVR_cols] = (X_test_total[tau_SUVR_cols] - X_pred_test[tau_SUVR_cols])**2

    recon['mean_recon_mri'] = (abs((recon[MRI_vol_cols])).sum(axis = 1).to_numpy()/recon[MRI_vol_cols].shape[1])
    recon['mean_recon_amyloid'] = (abs((recon[amyloid_SUVR_cols])).sum(axis = 1).to_numpy()/recon[amyloid_SUVR_cols].shape[1])
    recon['mean_recon_tau'] = (abs((recon[tau_SUVR_cols])).sum(axis = 1).to_numpy()/recon[tau_SUVR_cols].shape[1])

    #----------------- Training/validation losses, mean reconstruction (concatenated) and mean deviation (concatenated) -----------
    
    plt.subplots(figsize = (40, 10))
    
    epochs = range(len(train_loss))
    plt.subplot(1,3,1)
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training and validation loss')
    plt.legend()
    
    fs_cols = MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols
    
    recon['mean_recon_all'] = (abs((recon[fs_cols])).sum(axis = 1).to_numpy()/recon[fs_cols].shape[1])
    plt.subplot(1,3,2)
    sns.boxplot(x = 'stage', y = 'mean_recon_all', data = recon, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('Disease category', fontsize = 24)
    plt.ylabel('Mean reconstruction loss \n for all modalities', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)
    plt.title('Mean reconstruction loss \n (all modalities)', fontsize = 28)
    
    dev_bvae['mean_dev_all'] = (abs((dev_bvae[fs_cols])).sum(axis = 1).to_numpy()/dev_bvae[fs_cols].shape[1])
    plt.subplot(1,3,3)
    sns.boxplot(x = 'stage', y = 'mean_dev_all', data = dev_bvae, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('Disease category', fontsize = 24)
    plt.ylabel('Mean deviations \n for all modalities', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)
    plt.title('Mean deviations \n (all modalities)', fontsize = 28)


    #----------------- Mean reconstruction losses for each modality -----------
    
    plt.subplots(figsize = (40, 10))

    #x_axis = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1']
    plt.subplot(1,3,1)
    sns.boxplot(x = 'stage', y = 'mean_recon_mri', data = recon, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 24)
    plt.ylabel('Mean reconstruction loss \n for MRI volume', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)
    plt.title('Mean reconstruction loss \n (MRI volume)', fontsize = 28)

    plt.subplot(1,3,2)
    sns.boxplot(x = 'stage', y = 'mean_recon_mri', data = recon, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 24)
    plt.ylabel('Mean reconstruction loss \n for amyloid SUVR', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)
    plt.title('Mean reconstruction loss \n (Amyloid SUVR)', fontsize = 28) 

    plt.subplot(1,3,3)
    sns.boxplot(x = 'stage', y = 'mean_recon_tau', data = recon, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 24)
    plt.ylabel('Mean reconstruction loss \n for tau SUVR', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)
    plt.title('Mean reconstruction loss \n tau SUVR', fontsize = 28) 

    plt.tight_layout()
    plt.suptitle('MMVAE mean reconstruction across CDR categories (a) MRI volume (b) Amyloid SUVR (c) Tau SUVR', fontsize = 36)
    plt.subplots_adjust(top = 0.75, hspace= 0.2, wspace = 0.2)

    #----------------- Mean deviations for each modality across CDR categories ---------

    dev_bvae['mean_dev_mri'] = (abs((dev_bvae[MRI_vol_cols])).sum(axis = 1).to_numpy()/dev_bvae[MRI_vol_cols].shape[1])
    dev_bvae['mean_dev_amyloid'] = (abs((dev_bvae[amyloid_SUVR_cols])).sum(axis = 1).to_numpy()/dev_bvae[amyloid_SUVR_cols].shape[1])
    dev_bvae['mean_dev_tau'] = (abs((dev_bvae[tau_SUVR_cols])).sum(axis = 1).to_numpy()/dev_bvae[tau_SUVR_cols].shape[1])

    plt.subplots(figsize = (40, 10))

    plt.subplot(1,3,1)
    plt.title('Mean deviation after \n normalization (MRI volume Z-scores)', fontsize = 28)  
    sns.boxplot(x = 'stage', y = 'mean_dev_mri', data = dev_bvae, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 24)
    plt.ylabel('Mean deviation (Z scores) \n MRI volume', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)

    plt.subplot(1,3,2)
    plt.title('Mean deviation after \n normalization (Amyloid SUVR Z-scores)', fontsize = 28)  
    sns.boxplot(x = 'stage', y = 'mean_dev_amyloid', data = dev_bvae, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 24)
    plt.ylabel('Mean deviation (Z scores) \n Amyloid SUVR', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)

    plt.subplot(1,3,3)
    plt.title('Mean deviation after \n normalization (Tau SUVR Z-scores)', fontsize = 28)  
    sns.boxplot(x = 'stage', y = 'mean_dev_tau', data = dev_bvae, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 24)
    plt.ylabel('Mean deviation (Z scores) \n Tau SUVR', fontsize = 24)
    plt.yticks(fontsize = 22)
    plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 22)

    plt.tight_layout()
    plt.suptitle('MMVAE mean deviations across CDR categories (a) MRI volume (b) Amyloid SUVR (c) Tau SUVR', fontsize = 36)
    plt.subplots_adjust(top = 0.75, hspace= 0.2, wspace = 0.2)

    return dev_bvae, recon

#model, train_loss, val_loss, X_org_val, X_pred_val, dev_bvae, X_pred_valho, X_pred_test, X_test_total = complete_training_prediction(CN_model, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, age_sex_site_df, params)


'''
Calculating slope of deviations across different disease categories (for both proposed and unimodal) 
'''

def calculate_slope_model(cat, dev_bvae):
    
    from scipy.optimize import curve_fit

    #cat = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1']
    all_mean = []

    for i in cat:
        all_mean.append(dev_bvae['mean_dev_all'][dev_bvae.stage == i].mean())

    all_mean = dict(zip(cat, all_mean))

    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress([1, 2, 3, 4], list(all_mean.values()))
    slope = round(slope,2)
    
    print('slope = {}, intercept = {}\n'.format(slope, intercept))
    
    return slope
