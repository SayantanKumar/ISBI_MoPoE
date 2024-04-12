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



###########################################################
#-------------- Main function ------------------------
###########################################################

a_n_merged_harm = pd.read_csv('./saved_dataframes/a_n_merged_harm.csv')
a_n_merged = pd.read_csv('./saved_dataframes/a_n_merged.csv')

## Harmonized ADNI and KARI --> a_n_merged_harm
## Concatenated non-harmonized ADNI and KARI --> a_n_merged

a_n_merged_adni = a_n_merged.loc[a_n_merged.dataset == 'ADNI'].reset_index(drop = True)
a_n_merged_kari = a_n_merged.loc[a_n_merged.dataset == 'KARI'].reset_index(drop = True)

CN_model_adni, CN_held_val_adni, only_CN_test_adni, X_test_org_adni = split_train_test(a_n_merged_adni, 0, 'ADNI')

CN_model_kari, CN_held_val_kari, only_CN_test_kari, X_test_org_kari = split_train_test(a_n_merged_kari, 0.15, 'KARI')

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)
    
non_roi_cols = ['ID', 'Age', 'Sex', 'cdr', 'MR_Date', 'CDR_Date', 'amyloid_positive', 'amyloid_centiloid', 'cdr', 'stage', 'dataset']
    
############# ADNI results ############# 

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

fs_cols = list(MRI_vol_cols) + list(amyloid_SUVR_cols)
model = torch.load(join('./saved_models/ADNI_finetuned/MoPoEVAE', 'model.pkl'))

#----------------------------------------------------
#-------- Calculate latent and feature Z-scores ----
#----------------------------------------------------

feature_dev = X_test_org_adni.copy()
feature_dev[non_roi_cols] = X_test_org_adni[non_roi_cols]
feature_dev[fs_cols] = feature_zscores(model, m1_train_scaled, m2_train_scaled, m1_test_scaled, m2_test_scaled)

latent_dev = X_test_org_adni.copy()
latent_dev[non_roi_cols] = X_test_org_adni[non_roi_cols]
latent_dev[fs_cols] = latent_zscores(model, m1_train_scaled, m2_train_scaled, m1_test_scaled, m2_test_scaled)


def data_deviations_zscores(cohort_recon, train_recon):
    mean_train = np.mean(train_recon, axis=0)
    sd_train = np.std(train_recon, axis=0)
    z_scores = (cohort_recon - mean_train)/sd_train
    return z_scores

def deviation(orig, recon):
    return np.sqrt((orig - recon)**2)


def feature_zscores(model, m1_train_scaled, m2_train_scaled, m1_test_scaled, m2_test_scaled):
    
    train_concat = np.concatenate((m1_train_scaled, m2_train_scaled), axis=1)
    test_concat = np.concatenate((m1_test_scaled, m2_test_scaled), axis=1)

    train_recon = model.predict_reconstruction(m1_train_scaled, m2_train_scaled)
    train_recon = np.concatenate((train_recon[0][0], train_recon[0][1]), axis=1)
    train_dev = deviation(train_concat, train_recon)

    test_recon = model.predict_reconstruction(m1_test_scaled, m2_test_scaled)
    test_recon = np.concatenate((test_recon[0][0], test_recon[0][1]), axis=1)
    test_dev = deviation(test_concat, test_recon)
    
    feature_zscores = data_deviations_zscores(test_dev, train_dev)
    
    return feature_zscores


def latent_zscores(model, m1_train_scaled, m2_train_scaled, m1_test_scaled, m2_test_scaled):

    train_latents = model.predict_latents(m1_train_scaled, m2_train_scaled)
    test_latents = model.predict_latents(m1_test_scaled, m2_test_scaled)
    
    latent_zscores = data_deviations_zscores(test_latents, train_latents)
    
    return latent_zscores



#---------------------------------------------------------------------------
#-------- Effect size brain maps based on feature-based Z-score deviations ----
#---------------------------------------------------------------------------

def plot_effect_size(temp_dev_mmvae, modality_cols, modality, strip_suffix_col, dataset, value_range):
    
    from scipy.stats import f_oneway

    temp_dev_mmvae = temp_dev_mmvae.loc[temp_dev_mmvae.dataset == dataset].reset_index(drop = True)

    sig_cohen_hc_precl, sig_cohen_hc_cdr05, sig_cohen_hc_cdr1 = calculate_effect_size(temp_dev_mmvae, modality_cols)

    sig_cohen_hc_precl = [abs(x) for x in sig_cohen_hc_precl]
    sig_cohen_hc_cdr05 = [abs(x) for x in sig_cohen_hc_cdr05]
    sig_cohen_hc_cdr1 = [abs(x) for x in sig_cohen_hc_cdr1]
    
    ggseg_cols = [element.rstrip(strip_suffix_col) for element in modality_cols]
    
    effect_hc_precl = {key: value for key, value in dict(zip(ggseg_cols, sig_cohen_hc_precl)).items() if value != 0}
    effect_hc_cdr05 = {key: value for key, value in dict(zip(ggseg_cols, sig_cohen_hc_cdr05)).items() if value != 0}
    effect_hc_cdr1 = {key: value for key, value in dict(zip(ggseg_cols, sig_cohen_hc_cdr1)).items() if value != 0}
    
    def plot_maps(effect_list, sig_cohen_list, modality, fig_title, value_range):
        
        ggseg_python.plot_dk(effect_list, cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 18, vminmax = value_range, 
                          ylabel='Effect size', title=fig_title)
        print('Max = {}, Min = {}'.format(max(sig_cohen_list), min(sig_cohen_list)))


        ggseg_python.plot_aseg(effect_list, cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 18, vminmax = value_range, 
                          ylabel='Effect size', title=fig_title)
        print('Max = {}, Min = {}'.format(max(sig_cohen_list), min(sig_cohen_list)))

    
    plot_maps(effect_hc_precl, sig_cohen_hc_precl, modality, 'HC vs precl ({} {} deviations)'.format(dataset, modality), value_range)
    plot_maps(effect_hc_cdr05, sig_cohen_hc_cdr05, modality, 'HC vs cdr=0.5 ({} {} deviations)'.format(dataset, modality), value_range)
    plot_maps(effect_hc_cdr1, sig_cohen_hc_cdr1, modality, 'HC vs cdr>=1 ({} {} deviations)'.format(dataset, modality), value_range)
    
    #return sig_cohen_hc_precl, sig_cohen_hc_cdr05, sig_cohen_hc_cdr1, effect_hc_precl, effect_hc_cdr05, effect_hc_cdr1 

    
    
#--------------------------------------
#--------------------------------------

def cohen_d(group1, group2):
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    n1 = len(group1)
    n2 = len(group2)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d


def calculate_effect_size(temp_dev_mmvae, MRI_vol_cols):
    
#     dev_hc = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0 amyloid negative']
#     dev_precl = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0 amyloid positive']
#     dev_cdr05 = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0.5']
#     dev_cdr1 = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr >= 1']
    
    dev_hc = (temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'HC'][MRI_vol_cols])
    dev_precl = (temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'preclinical'][MRI_vol_cols])
    dev_cdr05 = (temp_dev_mmvae.loc[(temp_dev_mmvae.stage == 'cdr = 0.5')][MRI_vol_cols])
    dev_cdr1 = (temp_dev_mmvae.loc[(temp_dev_mmvae.stage == 'cdr >= 1')][MRI_vol_cols])

    p_list_hc_precl = []
    p_list_hc_cdr05 = []
    p_list_hc_cdr1 = []

    cohen_list_hc_precl = []
    cohen_list_hc_cdr05 = []
    cohen_list_hc_cdr1 = []

    for mri_col in MRI_vol_cols:
        F, p = f_oneway(dev_hc[mri_col], dev_precl[mri_col])
        p_list_hc_precl.append(p)
        cohen_list_hc_precl.append(cohen_d(dev_hc[mri_col], dev_precl[mri_col]))

    for mri_col in MRI_vol_cols:
        F, p = f_oneway(dev_hc[mri_col], dev_cdr05[mri_col])
        p_list_hc_cdr05.append(p)
        cohen_list_hc_cdr05.append(cohen_d(dev_hc[mri_col], dev_cdr05[mri_col]))

    for mri_col in MRI_vol_cols:
        F, p = f_oneway(dev_hc[mri_col], dev_cdr1[mri_col])
        p_list_hc_cdr1.append(p)
        cohen_list_hc_cdr1.append(cohen_d(dev_hc[mri_col], dev_cdr1[mri_col]))

    fdr_p_list_hc_precl = fdrcorrection(p_list_hc_precl, method='indep')[1]
    fdr_p_list_hc_cdr05 = fdrcorrection(p_list_hc_cdr05, method='indep')[1]
    fdr_p_list_hc_cdr1 = fdrcorrection(p_list_hc_cdr1, method='indep')[1]

    sig_cohen_hc_precl = cohen_list_hc_precl.copy()
    for i in range(len(cohen_list_hc_precl)):
        if fdr_p_list_hc_precl[i] < 0.1:
            sig_cohen_hc_precl[i] = cohen_list_hc_precl[i]
        else:
            sig_cohen_hc_precl[i] = 0


    sig_cohen_hc_cdr05 = cohen_list_hc_cdr05.copy()
    for i in range(len(cohen_list_hc_cdr05)):
        if fdr_p_list_hc_cdr05[i] < 0.1:
            sig_cohen_hc_cdr05[i] = cohen_list_hc_cdr05[i]
        else:
            sig_cohen_hc_cdr05[i] = 0


    sig_cohen_hc_cdr1 = cohen_list_hc_cdr1.copy()
    for i in range(len(cohen_list_hc_cdr1)):
        if fdr_p_list_hc_cdr1[i] < 0.1:
            sig_cohen_hc_cdr1[i] = cohen_list_hc_cdr1[i]
        else:
            sig_cohen_hc_cdr1[i] = 0
            
#     if MRI == True:
#         #ggseg_mri_cols = [element.rstrip('_Nvol') for element in MRI_vol_cols]
#         effect_hc_precl = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_precl)).items() if value != 0}
#         effect_hc_cdr05 = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr05)).items() if value != 0}
#         effect_hc_cdr1 = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr1)).items() if value != 0}

#     if SUVR == True:
#         #ggseg_mri_cols = [element.rstrip('_Nvol') for element in MRI_vol_cols]
#         effect_hc_precl = {key: abs(value) for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_precl)).items() if value != 0}
#         effect_hc_cdr05 = {key: abs(value) for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr05)).items() if value != 0}
#         effect_hc_cdr1 = {key: abs(value) for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr1)).items() if value != 0}

    return sig_cohen_hc_precl, sig_cohen_hc_cdr05, sig_cohen_hc_cdr1
    

#---------------------------------
#---------------------------------
    
def effect_size_maps(temp_dev_mmvae, MRI_vol_cols, ggseg_mri_cols):
    
    sig_cohen_hc_precl_mri, sig_cohen_hc_cdr05_mri, sig_cohen_hc_cdr1_mri = calculate_effect_size(temp_dev_mmvae, MRI_vol_cols)
    
    sig_cohen_hc_precl_mri = [abs(x) for x in sig_cohen_hc_precl_mri]
    sig_cohen_hc_cdr05_mri = [abs(x) for x in sig_cohen_hc_cdr05_mri]
    sig_cohen_hc_cdr1_mri = [abs(x) for x in sig_cohen_hc_cdr1_mri]

    #ggseg_mri_cols = [element.rstrip('_Nvol') for element in MRI_vol_cols]
    effect_hc_precl_mri = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_precl_mri)).items() if value != 0}
    effect_hc_cdr05_mri = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr05_mri)).items() if value != 0}
    effect_hc_cdr1_mri = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr1_mri)).items() if value != 0}

    return effect_hc_precl_mri, effect_hc_cdr05_mri, effect_hc_cdr1_mri


plot_effect_size(feature_dev, MRI_vol_cols, 'MRI', '_Nvol', 'ADNI', [0, 3.5])
plot_effect_size(feature_dev, amyloid_SUVR_cols, 'Amyloid', '_Asuvr', 'ADNI', [0, 3.5])

