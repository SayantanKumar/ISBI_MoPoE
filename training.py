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

#------------------------------------
def add_covariates(roi_demo_both):

    age_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'Age'), index = roi_demo_both.index.values)
    age_mat = one_hot_encoding(roi_demo_both, 'Age')

    sex_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'Sex'), index = roi_demo_both.index.values)
    sex_mat = one_hot_encoding(roi_demo_both, 'Sex')

    site_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'dataset'), index = roi_demo_both.index.values)
    site_mat = one_hot_encoding(roi_demo_both, 'dataset')

    age_sex_df = pd.DataFrame(np.concatenate((age_mat, sex_mat), axis=1), index = roi_demo_both.index.values)

    return age_sex_df
#-----------------------------------

def split_train_test(roi_demo_both, ho_frac, dataset):
    
    roi_demo_dataset = roi_demo_both.loc[roi_demo_both.dataset == dataset].reset_index(drop = True)
    
    roi_demo_dataset.loc[roi_demo_dataset.Sex == 0, 'Sex'] = 'Female'
    roi_demo_dataset.loc[roi_demo_dataset.Sex == 1, 'Sex'] = 'Male'
    roi_demo_dataset['Age'] = round(roi_demo_dataset['Age'])

    only_CN = roi_demo_dataset.loc[roi_demo_dataset.stage == 'cdr = 0 amyloid negative']
    rest = roi_demo_dataset.loc[roi_demo_dataset['stage'] != 'cdr = 0 amyloid negative']

    y_CN = only_CN['stage']
    y_rest = rest['stage']

    only_CN_test = only_CN.sample(n=round(0*len(only_CN)), random_state=1)
    CN_model, CN_held_val = train_test_split(only_CN.loc[~only_CN.ID.isin(only_CN_test.ID.values)], test_size=ho_frac, shuffle = False, random_state = 1000)

    X_test_org = pd.concat([only_CN_test, rest]).copy().reset_index(drop = True)

    print('Number of {} CN used for model training/val: {}'.format(dataset, len(CN_model)))
    print('Number of {} CN used for normalization: {}'.format(dataset, len(CN_held_val)))
    print('Number of {} CN used in test set: {}'.format(dataset, len(only_CN_test)))
    print('Number of {} disease patients used in test set: {}'.format(dataset, len(X_test_org)))

    return CN_model, CN_held_val, only_CN_test, X_test_org

#--------------------------------------------------------------
#--------------------------------------------------------------

###########################################################
#-------------- Main function ------------------------
###########################################################

# Number of KARI CN used for model training/val: 257
# Number of KARI CN used for normalization: 65
# Number of KARI CN used in test set: 0
# Number of KARI disease patients used in test set: 214

# Number of ADNI CN used for model training/val: 147
# Number of ADNI CN used for normalization: 37
# Number of ADNI CN used in test set: 0
# Number of ADNI disease patients used in test set: 320


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

############# Training multimodal and unimodal models ############# 

m1_train = CN_model_kari[MRI_vol_cols]
m2_train = CN_model_kari[amyloid_SUVR_cols]

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# m1_train_scaled = scaler.fit_transform(m1_train)
# m2_train_scaled = scaler.fit_transform(m2_train)

mean_controls_m1 = np.mean(m1_train, axis=0)
sd_controls_m1 = np.std(m1_train, axis=0)
m1_train_scaled = (m1_train - mean_controls_m1)/sd_controls_m1

mean_controls_m2 = np.mean(m2_train, axis=0)
sd_controls_m2 = np.std(m2_train, axis=0)
m2_train_scaled = (m2_train - mean_controls_m2)/sd_controls_m2

input_dims=[m1_train_scaled.shape[1],m2_train_scaled.shape[1]]

train = np.concatenate((m1_train_scaled, m2_train_scaled), axis=1)

#------------------------------------------------
#------------------------------------------------

from multiviewae import mVAE, weighted_mVAE, mmVAE, MoPoEVAE, mmJSD, mcVAE, JMVAE, DVCCA, AAE

## mvae --> Wu and Goodman PoE
## mmVAE --> Shi et al. MoE
## weighted_mVAE --> MICCAI paper
## MoPoEVAE --> MoPoE
## mmJSD --> Sutter et al., Multimodal Jensen-Shannon divergence (mmJSD) model with Product-of-Experts dynamic prior. 
## mcVAE --> Multi-Channel Variational Autoencoder and Sparse Multi-Channel Variational Autoencoder.
## JMVAE --> Suzuki, Masahiro & Nakayama, Kotaro & Matsuo, Yutaka. (2016). Joint Multimodal Learning with Deep Generative Models.
## DVCCA --> Wang, Weiran & Lee, Honglak & Livescu, Karen. (2016). Deep Variational Canonical Correlation Analysis.
## AAE --> Multi-view Adversarial Autoencoder model with a separate latent representation for each view.


max_epochs = 2000
batch_size = 64

#---- Proposed MoPoE

mvae_MoPoE = MoPoEVAE(cfg=".configs/multimodal.yaml", input_dim=input_dims,)
mvae_MoPoE.cfg.out_dir = './saved_models/MoPoEVAE'
print('Mixture-of-Product-of-Experts')
mvae_MoPoE.fit(m1_train_scaled, m2_train_scaled,  max_epochs=max_epochs, batch_size=batch_size)


#--- Aggregation strategies (MoE, PoE, gPoE)

mvae_MoE = mmVAE(cfg=".configs/multimodal.yaml", input_dim=input_dims,)
mvae_MoE.cfg.out_dir = './saved_models/mmVAE'
print('Mixture-of-Products')
mvae_MoE.fit(m1_train_scaled, m2_train_scaled,  max_epochs=max_epochs, batch_size=batch_size)


mvae_gPoE = weighted_mVAE(cfg=".configs/multimodal.yaml", input_dim=input_dims,)
mvae_gPoE.cfg.out_dir = './saved_models/weighted_mVAE'
print('Weighted Product-of-Experts')
mvae_gPoE.fit(m1_train_scaled, m2_train_scaled,  max_epochs=max_epochs, batch_size=batch_size)


mvae_PoE = mVAE(cfg=".configs/multimodal.yaml", input_dim=input_dims,)
mvae_PoE.cfg.out_dir = './saved_models/mVAE'
print('Product-of-Experts')
mvae_PoE.fit(m1_train_scaled, m2_train_scaled,  max_epochs=max_epochs, batch_size=batch_size)

#---- SOTA multimodal VAE/AE---

mvae_JSD = mmJSD(cfg=".configs/multimodal.yaml", input_dim=input_dims,)
mvae_JSD.cfg.out_dir = './saved_models/mmJSD'
print('Multimodal Jensen-Shannon divergence (mmJSD) model with Product-of-Experts dynamic prior. ')
mvae_JSD.fit(m1_train_scaled, m2_train_scaled,  max_epochs=max_epochs, batch_size=batch_size)


mvae_JMVAE = JMVAE(cfg=".configs/multimodal.yaml", input_dim=input_dims,)
mvae_JMVAE.cfg.out_dir = './saved_models/JMVAE'
print('Joint Multimodal Learning with Deep Generative Models.')
mvae_JMVAE.fit(m1_train_scaled, m2_train_scaled,  max_epochs=max_epochs, batch_size=batch_size)


mvae_mcVAE = mcVAE(cfg=".configs/multimodal.yaml", input_dim=input_dims,)
mvae_mcVAE.cfg.out_dir = './saved_models/mcVAE'
print('Multi-Channel Variational Autoencoder and Sparse Multi-Channel Variational Autoencoder.')
mvae_mcVAE.fit(m1_train_scaled, m2_train_scaled,  max_epochs=max_epochs, batch_size=batch_size)


#--- Unimodal baselines

mvae_m1 = mVAE(cfg=".configs/unimodal.yaml", input_dim=[input_dims[0]])
print('MRI only' )
mvae_m1.cfg.out_dir = './saved_models/mri_only'
mvae_m1.fit(m1_train_scaled, max_epochs=max_epochs, batch_size=batch_size)


mvae_m2 = mVAE(cfg=".congfigs/unimodal.yaml", input_dim=[input_dims[1]])
print('Amyloid only' )
mvae_m2.cfg.out_dir = './saved_models/amyloid_only'
mvae_m2.fit(m2_train_scaled, max_epochs=max_epochs, batch_size=batch_size)


mvae_m1_m2 = mVAE(cfg=".configs/unimodal.yaml", input_dim=[input_dims[0] + input_dims[1]])
print('Concatenated MRI and amyloid')
mvae_m1_m2.cfg.out_dir = './saved_models/mri_amyloid_concat'
concat_both = np.concatenate((m1_train_scaled, m2_train_scaled), axis=1)
mvae_m1_m2.fit(concat_both, max_epochs=max_epochs, batch_size=batch_size)


