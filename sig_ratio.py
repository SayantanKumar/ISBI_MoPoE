import numpy as np
import pandas as pd
import os
import warnings
import ggseg_python
import re
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from multiviewae import mVAE, weighted_mVAE, mmVAE, MoPoEVAE, mmJSD, mcVAE, JMVAE, DVCCA, AAE

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
from multimodal_VAE import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer


###########################################################
#-------------- Relevant functions ------------------------
###########################################################


def latent_deviations_mahalanobis_across_sig(cohort, train):
    latent_dim = cohort[0].shape[1]
    dists = calc_robust_mahalanobis_distance(cohort[0], train[0])
    pvals = 1 - chi2.cdf(dists, latent_dim - 1)
    return pvals


def calc_mahalanobis_distance(values, train_values):
    covariance  = np.cov(train_values, rowvar=False)
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    centerpoint = np.mean(train_values, axis=0)
    dists = np.zeros((values.shape[0],1))
    for i in range(0, values.shape[0]):
        p0 = values[i,:]
        dist = (p0-centerpoint).T.dot(covariance_pm1).dot(p0-centerpoint)
        dists[i,:] = dist
    return dists


def calc_robust_mahalanobis_distance(values, train_values):
    # fit a MCD robust estimator to data
    robust_cov = MinCovDet(random_state=42).fit(train_values)
    mahal_robust_cov = robust_cov.mahalanobis(values)
    return mahal_robust_cov


def latent_count_ratio(pvals_cohort, pvals_holdout, model_type, cols):
    thresh = 0.001
    count_cohort = (pvals_cohort <= thresh).sum()
    count_holdout = (pvals_holdout <= thresh).sum()
    ratio_cohort = count_cohort/pvals_cohort.shape[0]
    ratio_holdout = count_holdout/pvals_holdout.shape[0]
    ratio = ratio_cohort/ratio_holdout
    df = pd.DataFrame(np.array([key, count_cohort, ratio_cohort, count_holdout, ratio_holdout, ratio]).reshape(1,-1),
    columns=cols)
    return df


def feature_sig_ratio(cohort_recon, holdout_recon, train_recon, key, cols):
    
    thresh = 0.001
    dists = calc_robust_mahalanobis_distance(cohort_recon, train_recon)
    pvals_cohort = 1 - chi2.cdf(dists, cohort_recon.shape[1] - 1)
        
    dists = calc_robust_mahalanobis_distance(holdout_recon, train_recon)
    pvals_holdout = 1 - chi2.cdf(dists, cohort_recon.shape[1] - 1)
        

    count_cohort = (pvals_cohort <= thresh).sum()
    count_holdout = (pvals_holdout <= thresh).sum()
    ratio_cohort = count_cohort/pvals_cohort.shape[0]
    ratio_holdout = count_holdout/pvals_holdout.shape[0]
    ratio = ratio_cohort/ratio_holdout
        
    df = pd.DataFrame(np.array([key, count_cohort, ratio_cohort, count_holdout, ratio_holdout, ratio]).reshape(1,-1),
        columns=cols)
    
    return df

###########################################################
#-------------- Main function ------------------------
###########################################################


a_n_merged_harm = pd.read_csv('./saved_dataframes/a_n_merged_harm.csv')
a_n_merged = pd.read_csv('./saved_dataframes/a_n_merged.csv')

## Harmonized ADNI and KARI --> a_n_merged_harm
## Concatenated non-harmonized ADNI and KARI --> a_n_merged

a_n_merged_adni = a_n_merged.loc[a_n_merged.dataset == 'ADNI'].reset_index(drop = True)
a_n_merged_kari = a_n_merged.loc[a_n_merged.dataset == 'KARI'].reset_index(drop = True)

CN_model_adni, CN_held_val_adni, only_CN_test_adni, X_test_org_adni = split_train_test(a_n_merged_adni, 0.15, 'ADNI')

CN_model_kari, CN_held_val_kari, only_CN_test_kari, X_test_org_kari = split_train_test(a_n_merged_kari, 0.15, 'KARI')

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)

    
############# Results only for a single dataset (e.g. KARI) ############# 

m1_train = CN_model_kari[MRI_vol_cols]
m2_train = CN_model_kari[amyloid_SUVR_cols]

m1_holdout = CN_held_val_kari[MRI_vol_cols]
m2_holdout = CN_held_val_kari[amyloid_SUVR_cols]

m1_test = X_test_org_kari[MRI_vol_cols]
m2_test = X_test_org_kari[amyloid_SUVR_cols]

#-------- Scaling train, test and holdout based on parameters from train-----
mean_controls_m1 = np.mean(m1_train, axis=0)
sd_controls_m1 = np.std(m1_train, axis=0)
mean_controls_m2 = np.mean(m2_train, axis=0)
sd_controls_m2 = np.std(m2_train, axis=0)


m1_train_scaled = (m1_train - mean_controls_m1)/sd_controls_m1
m2_train_scaled = (m2_train - mean_controls_m2)/sd_controls_m2

m1_holdout_scaled = (m1_holdout - mean_controls_m1)/sd_controls_m1
m2_holdout_scaled = (m2_holdout - mean_controls_m2)/sd_controls_m2

m1_test_scaled = (m1_test - mean_controls_m1)/sd_controls_m1
m2_test_scaled = (m2_test - mean_controls_m2)/sd_controls_m2

train_concat = np.concatenate((m1_train_scaled, m2_train_scaled), axis=1)
holdout_concat = np.concatenate((m1_holdout_scaled, m2_holdout_scaled), axis=1)
test_concat = np.concatenate((m1_test_scaled, m2_test_scaled), axis=1)


model_dict = {'MoPoEVAE': './saved_models/MoPoEVAE',
              'mmVAE': './saved_models/mmVAE',
              'weighted_mVAE': './saved_models/weighted_mVAE',
              'mVAE': './saved_models/mVAE',
              'mmJSD': './saved_models/mmJSD'
              'JMVAE': './saved_models/JMVAE',
              'mcVAE': './saved_models/mcVAE',
              'mri_only': './saved_models/mri_only',
              'amyloid_only': './saved_models/amyloid_only',
              'mri_amyloid_concat': './saved_models/mri_amyloid_concat',}

cols = ['model', 'count cohort', 'ratio cohort', 'count holdout', 'ratio holdout', 'significance ratio']
results_df = pd.DataFrame(columns=cols)

'''
model_path = './saved_models/MoPoEVAE'
model = torch.load(join(model_path, 'model.pkl'))
train_latents = model.predict_latents(m1_train_scaled, m2_train_scaled)
holdout_latents = model.predict_latents(m1_holdout_scaled, m2_holdout_scaled)
test_latents = model.predict_latents(m1_test_scaled, m2_test_scaled)

pvals_holdout = latent_deviations_mahalanobis_across_sig(holdout_latents, train_latents)
pvals_test = latent_deviations_mahalanobis_across_sig(test_latents, train_latents)
results = latent_count_ratio(pvals_test, pvals_holdout, key, cols)
results_df = pd.concat([results_df, results], axis=0)
'''

##########################################################
#-------signifcance ratio (latent mahalnobis distance------------------
##########################################################


cols = ['model', 'count cohort', 'ratio cohort', 'count holdout', 'ratio holdout', 'significance ratio']
latent_sig_results = pd.DataFrame(columns=cols)

for key, val in model_dict.items():
    
    model = torch.load(join(val, 'model.pkl'))
    print(key)
    
    if key == 'mri_amyloid_concat':
        train_latents = model.predict_latents(train_concat)
        holdout_latents = model.predict_latents(holdout_concat)
        test_latents = model.predict_latents(test_concat)
        
    elif key == 'mri_only':
        train_latents = model.predict_latents(m1_train_scaled)
        holdout_latents = model.predict_latents(m1_holdout_scaled)
        test_latents = model.predict_latents(m1_test_scaled)

    elif key == 'amyloid_only':
        train_latents = model.predict_latents(m2_train_scaled)
        holdout_latents = model.predict_latents(m2_holdout_scaled)
        test_latents = model.predict_latents(m2_test_scaled)
    
    elif key == 'mmVAE':
        
        train_latents = model.predict_latents(m1_train_scaled, m2_train_scaled)
        train_latents = [np.mean([train_latents[0], train_latents[1]], axis=0)]
        
        holdout_latents = model.predict_latents(m1_holdout_scaled, m2_holdout_scaled)
        holdout_latents = [np.mean([holdout_latents[0], holdout_latents[1]], axis=0)]
        
        test_latents = model.predict_latents(m1_test_scaled, m2_test_scaled)
        test_latents = [np.mean([test_latents[0], test_latents[1]], axis=0)]
        
        
    else:
        train_latents = model.predict_latents(m1_train_scaled, m2_train_scaled)
        holdout_latents = model.predict_latents(m1_holdout_scaled, m2_holdout_scaled)
        test_latents = model.predict_latents(m1_test_scaled, m2_test_scaled)

    pvals_holdout = latent_deviations_mahalanobis_across_sig(holdout_latents, train_latents)
    pvals_test = latent_deviations_mahalanobis_across_sig(test_latents, train_latents)
    results = latent_count_ratio(pvals_test, pvals_holdout, key, cols)
    latent_sig_results = pd.concat([results_df, results], axis=0)

latent_sig_results.to_csv('./saved_models/latent_sig_results.csv'.format(date), header=True, index=False)


##########################################################
#-------signifcance ratio (Feature mahalnobis distance)------------------
##########################################################

def deviation(orig, recon, recon_type='abs'):
    return np.sqrt((orig - recon)**2)

cols = ['model', 'count cohort', 'ratio cohort', 'count holdout', 'ratio holdout', 'significance ratio']
feature_sig_results = pd.DataFrame(columns=cols)

for key, val in model_dict.items():
    
    model = torch.load(join(val, 'model.pkl'))
    
    if key == 'mri_amyloid_concat':
        train_recon = model.predict_reconstruction(train_concat)
        holdout_recon = model.predict_reconstruction(holdout_concat)
        test_recon = model.predict_reconstruction(test_concat)#
        
        dev_holdout = deviation(holdout_concat, holdout_recon[0][0])
        dev_test = deviation(test_concat, test_recon[0][0])
        dev_train = deviation(train_concat, train_recon[0][0])
        
#         results = deviation_sig_count(dev_test, dev_holdout, dev_train, key, cols, zscore=zscore)
#         results_df = pd.concat([results_df, results], axis=0)

    elif key == 'mri_only':
        
        train_recon = model.predict_reconstruction(m1_train_scaled)
        holdout_recon = model.predict_reconstruction(m1_holdout_scaled)
        test_recon = model.predict_reconstruction(m1_test_scaled)
        
        dev_holdout = deviation(m1_holdout_scaled, holdout_recon[0][0])
        dev_test = deviation(m1_test_scaled, test_recon[0][0])
        dev_train = deviation(m1_train_scaled, train_recon[0][0])

    elif key == 'amyloid_only':
        
        train_recon = model.predict_reconstruction(m2_train_scaled)
        holdout_recon = model.predict_reconstruction(m2_holdout_scaled)
        test_recon = model.predict_reconstruction(m2_test_scaled)
        
        dev_holdout = deviation(m2_holdout_scaled, holdout_recon[0][0])
        dev_test = deviation(m2_test_scaled, test_recon[0][0])
        dev_train = deviation(m2_train_scaled, train_recon[0][0])
        
       
    elif key == 'mmVAE':
        
        train_recon = model.predict_reconstruction(m1_train_scaled, m2_train_scaled)
        holdout_recon = model.predict_reconstruction(m1_holdout_scaled, m2_holdout_scaled)
        test_recon = model.predict_reconstruction(m1_test_scaled, m2_test_scaled)
        
        m1_dev_holdout = deviation(m1_holdout_scaled, holdout_recon[0][0])
        m2_dev_holdout = deviation(m2_holdout_scaled, holdout_recon[1][1])
        
        m1_dev_train = deviation(m1_train_scaled, train_recon[0][0])
        m2_dev_train = deviation(m2_train_scaled, train_recon[1][1])
        
        m1_dev_test = deviation(m1_test_scaled, test_recon[0][0])
        m2_dev_test = deviation(m2_test_scaled, test_recon[1][1])
        
        dev_test = np.concatenate((m1_dev_test, m2_dev_test), axis=1)
        dev_holdout = np.concatenate((m1_dev_holdout, m2_dev_holdout), axis=1)
        dev_train = np.concatenate((m1_dev_train, m2_dev_train), axis=1)

        
    else:
        train_recon = model.predict_reconstruction(m1_train_scaled, m2_train_scaled)
        holdout_recon = model.predict_reconstruction(m1_holdout_scaled, m2_holdout_scaled)
        test_recon = model.predict_reconstruction(m1_test_scaled, m2_test_scaled)
        
        m1_dev_holdout = deviation(m1_holdout_scaled, holdout_recon[0][0])
        m2_dev_holdout = deviation(m2_holdout_scaled, holdout_recon[0][1])
        
        m1_dev_test = deviation(m1_test_scaled, test_recon[0][0])
        m2_dev_test = deviation(m2_test_scaled, test_recon[0][1])

        m1_dev_train = deviation(m1_train_scaled, train_recon[0][0])
        m2_dev_train = deviation(m2_train_scaled, train_recon[0][1])

        dev_test = np.concatenate((m1_dev_test, m2_dev_test), axis=1)
        dev_holdout = np.concatenate((m1_dev_holdout, m2_dev_holdout), axis=1)
        dev_train = np.concatenate((m1_dev_train, m2_dev_train), axis=1)
        
    results = feature_sig_ratio(dev_test, dev_holdout, dev_train, key, cols) 
    feature_sig_results = pd.concat([results_df, results], axis=0)

feature_sig_results.to_csv('./saved_models/feature_sig_results.csv'.format(date), header=True, index=False)

