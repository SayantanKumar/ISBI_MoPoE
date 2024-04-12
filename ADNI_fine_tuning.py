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
#--------------------------------------
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
    
with open("./saved_dataframes/tau_SUVR_cols", "rb") as fp: 
    tau_SUVR_cols = pickle.load(fp)
    
############# Results only for a single dataset (e.g. KARI) ############# 

m1_transfer = CN_model_adni[MRI_vol_cols]
m2_transfer = CN_model_adni[amyloid_SUVR_cols]

m1_test = X_test_org_adni[MRI_vol_cols]
m2_test = X_test_org_adni[amyloid_SUVR_cols]

#-------- Scaling train, test and holdout based on parameters from train-----

mean_controls_m1 = np.mean(m1_transfer, axis=0)
sd_controls_m1 = np.std(m1_transfer, axis=0)

mean_controls_m2 = np.mean(m2_transfer, axis=0)
sd_controls_m2 = np.std(m2_transfer, axis=0)


m1_transfer_scaled = (m1_transfer - mean_controls_m1)/sd_controls_m1
m2_transfer_scaled = (m2_transfer - mean_controls_m2)/sd_controls_m2

m1_test_scaled = (m1_test - mean_controls_m1)/sd_controls_m1
m2_test_scaled = (m2_test - mean_controls_m2)/sd_controls_m2

transfer_concat = np.concatenate((m1_transfer_scaled, m2_transfer_scaled), axis=1)
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


max_epochs = 100
batch_size = 20

for key, val in model_dict.items():
    
    model = torch.load(join(val, "model.pkl"))
    new_path = join(val, "ADNI_finetuned")
    model.cfg.out_dir = new_path
    
    if key  == 'mri_amyloid_concat':
        model.fit(transfer_concat, max_epochs=max_epochs, batch_size=batch_size)
        
    elif key == 'mri_only':
        model.fit(m1_transfer_scaled, max_epochs=max_epochs, batch_size=batch_size)
        
    elif key == 'amyloid_only':
        model.fit(m2_transfer_scaled, max_epochs=max_epochs, batch_size=batch_size)
        
    else:
        model.fit(m1_transfer_scaled, m2_transfer_scaled, max_epochs=max_epochs, batch_size=batch_size)
        
#------------------------------------