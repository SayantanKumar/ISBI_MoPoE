import numpy as np
import pandas as pd
import os
import warnings
import ggseg_python
import re
import pickle
import json
import matplotlib.pyplot as plt
from scipy.stats import chi2, linregress
from sklearn.model_selection import train_test_split

#import neuroHarmonize
from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel, saveHarmonizationModel

from multiviewae import mVAE, weighted_mVAE, mmVAE, MoPoEVAE, mmJSD, mcVAE, JMVAE, DVCCA, AAE

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

def deviations_mahalanobis(cohort, train):
    dists = calc_mahalanobis_distance(cohort[0], train[0])
    return dists

def calc_mahalanobis_distance(values, train_values):
    covariance  = np.cov(train_values, rowvar=False)
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    centerpoint = np.mean(train_values, axis=0)
    dists = np.zeros((values.shape[0]))
    for i in range(0, values.shape[0]):
        p0 = values[i,:]
        dist = (p0-centerpoint).T.dot(covariance_pm1).dot(p0-centerpoint)
        dists[i] = dist
    return dists


def calculate_deviations_across_stages(model, X_test_org_adni, m1_train_scaled, m2_train_scaled, m1_test_scaled, m2_test_scaled, dataset):
    
    print(model)
    
    #-------------
    train_latents = model.predict_latents(m1_train_scaled, m2_train_scaled)
    test_latents = model.predict_latents(m1_test_scaled, m2_test_scaled)
    latent_dev = deviations_mahalanobis(test_latents, train_latents)
    latent_dev = X_test_org_adni['latent_mahal']
    
    #--------------
    train_recon = model.predict_reconstruction(m1_train_scaled, m2_train_scaled)
    test_recon = model.predict_reconstruction(m1_test_scaled, m2_test_scaled)
        
    m1_dev_test = deviation(m1_test_scaled, test_recon[0][0])
    m2_dev_test = deviation(m2_test_scaled, test_recon[0][1])

    m1_dev_train = deviation(m1_train_scaled, train_recon[0][0])
    m2_dev_train = deviation(m2_train_scaled, train_recon[0][1])

    dev_test = np.concatenate((m1_dev_test, m2_dev_test), axis=1)
    dev_train = np.concatenate((m1_dev_train, m2_dev_train), axis=1)
    
    feature_dev = deviations_mahalanobis(dev_test, dev_train)
    feature_dev = X_test_org_adni['feature_mahal']
    
    plot_deviations_across_stages(X_test_org_adni, 'stage', 'latent_mahal', model)
    plot_deviations_across_stages(X_test_org_adni, 'stage', 'feature_mahal', model)
    
    return feature_dev, latent_dev
    
    
def plot_deviations_across_stages(df, x_col, y_col, model, dataset): 
    
    slope, intercept, r, p, sterr = linregress(x=df[x_col], y=df[y_col])
    
    plt.figure(figsize = (6,4))
    sns.boxplot(x = x_col, y = y_col, data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize': 10})
    
    add_stat_annotation(ax=plt.gca(), data=df, x= x_col, y= y_col, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'],
                    box_pairs=[("HC", "preclinical"), ("preclinical", "cdr = 0.5"), ("cdr = 0.5", "cdr >= 1")],
                    test='Kruskal', text_format='star', loc='inside', verbose=1)

    plt.xlabel('Disease stages', fontsize = 18)
    plt.ylabel('severity (Z-scores)', fontsize = 18)
    plt.yticks(fontsize = 16)
    plt.xticks(ticks = [0,1,2,3], labels=['CU', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    #plt.legend(fontsize = 20)
    plt.title('{} latent deviations across stages ({}) \n slope = {}'.format(model, dataset, slope), fontsize = 20)


    
def plot_cognition_correlation(df, x_axis = '', y_axis = ''):
    
    #plt.figure(figsize = (6,4))
    
    pearson = []
    corr_p, _ = pearsonr(df[x_axis], df[y_axis])
    pearson.append(corr_p)
    R2_pearson = [i**2 for i in pearson]
    
    m, b = np.polyfit(df[x_axis], df[y_axis], 1)
    plt.plot(df[x_axis], m*df[x_axis] + b, color = 'red')

    plt.scatter(df[x_axis], df[y_axis])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.xlabel(x_axis, fontsize = 18)
    #plt.ylabel('Disease severity \n {}'.format(modality), fontsize = 18)
    #plt.title('$r$ = {}'.format(round(corr_p, 3)), fontsize = 18)
    
    return corr_p


def calculate_regression_coefficients(df, cog_col, severity_col):
    
    import statsmodels.api as sm

    #df = dev_bvae_adni_cog.copy()
    df['intercept'] = 1

    X = df[['intercept', cog_col]]
    y = df[severity_col]

    model = sm.OLS(y, X)
    results = model.fit()

    beta = results.params[1]
    p_value = results.pvalues[1]

    print("Beta coefficient:", beta)
    print("P-value:", p_value)
    
    return beta, p_value



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

    
############# ADNI results ########################### 
# Note that sig ratio results were shown for KARI only.

m1_train = CN_model_adni[MRI_vol_cols]
m2_train = CN_model_adni[amyloid_SUVR_cols]

m1_test = X_test_org_adni[MRI_vol_cols]
m2_test = X_test_org_adni[amyloid_SUVR_cols]

#-------- Scaling train, test and holdout based on parameters from train-----

mean_controls_m1 = np.mean(m1_train, axis=0)
sd_controls_m1 = np.std(m1_train, axis=0)
mean_controls_m2 = np.mean(m2_train, axis=0)
sd_controls_m2 = np.std(m2_train, axis=0)


m1_train_scaled = (m1_train - mean_controls_m1)/sd_controls_m1
m2_train_scaled = (m2_train - mean_controls_m2)/sd_controls_m2

m1_test_scaled = (m1_test - mean_controls_m1)/sd_controls_m1
m2_test_scaled = (m2_test - mean_controls_m2)/sd_controls_m2

train_concat = np.concatenate((m1_train_scaled, m2_train_scaled), axis=1)
test_concat = np.concatenate((m1_test_scaled, m2_test_scaled), axis=1)

model = torch.load(join('./saved_models/ADNI_finetuned/MoPoEVAE', 'model.pkl'))

#------------ Sensitivity w.r.t disease staging --------
feature_dev, latent_dev = calculate_deviations_across_stages(model, X_test_org_adni, m1_train_scaled, m2_train_scaled, m1_test_scaled, m2_test_scaled, 'ADNI')


#------------ Association with cognition ----------------

cog_adni = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/UWNPSYCHSUM_01_23_23_13May2023.csv')
cog_dev = pd.merge(X_test_org_adni, cog_adni[['RID', 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS']], left_on = 'ID', right_on = 'RID', how = 'inner')

#regress out effect of age from cognitive scores
x = cog_dev['AGE'].to_numpy().reshape(-1,1)
for col in ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS']:
    lin_reg = LinearRegression()
    y = cog_dev[col].to_numpy().reshape(-1,1)
    lin_reg.fit(x,y)
    y_pred = lin_reg.predict(x)
    resid = y - y_pred
    cog_dev['{0}_reg'.format(col)] = np.squeeze(resid)



for cog_col in ['ADNI_MEM_reg', 'ADNI_EF_reg', 'ADNI_LAN_reg', 'ADNI_VS_reg']:
    
    beta_l, pval_l = calculate_regression_coefficients(cog_dev, cog_col, 'latent_mahal')
    beta_f, pval_f = calculate_regression_coefficients(cog_dev, cog_col, 'feature_mahal')
    
    plt.subplots(figsize = (12,6))
    
    plt.subplot(1,2,1)
    corr_l = plot_cognition_correlation(cog_dev, x_axis = cog_col, y_axis = 'latent_mahal')
    plt.ylabel('{} latent deviations'.format(model), fontsize = 18)
    plt.title('$r$ = {}, p = {}'.format(round(corr_l, 3), pval_l), fontsize = 18)
    
    plt.subplot(1,2,2)
    plot_cognition_correlation(cog_dev, x_axis = cog_col, y_axis = 'latent_feature')
    plt.ylabel('{} feature deviations'.format(model), fontsize = 18)
    plt.title('$r$ = {}, p = {}'.format(round(corr_f, 3), pval_f), fontsize = 18)
