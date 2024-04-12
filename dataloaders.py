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

#--------------------------------------------------------------
#--------------------------------------------------------------


class train_dataloader(torch.utils.data.Dataset):
 
  def __init__(self, X_train_m1, X_train_m2, X_train_m3, age_cond):

    self.X_train_m1=torch.tensor(X_train_m1,dtype=torch.float32)
    self.X_train_m2=torch.tensor(X_train_m2,dtype=torch.float32)
    self.X_train_m3=torch.tensor(X_train_m3,dtype=torch.float32)
    
    self.age_cond = torch.tensor(age_cond,dtype=torch.float32)
    
    #self.y_train = torch.tensor(y_train,dtype=torch.float32)
    
    #self.X_val_m1=torch.tensor(X_val_m1,dtype=torch.float32)
    #self.X_val_m2=torch.tensor(X_val_m2,dtype=torch.float32)
 

  def __len__(self):
    return len(self.X_train_m1)
   
  def __getitem__(self,idx):
    return self.X_train_m1[idx], self.X_train_m2[idx], self.X_train_m3[idx], self.age_cond[idx]

#--------------------------------------------------------------
#--------------------------------------------------------------


class val_dataloader(torch.utils.data.Dataset):
 
  def __init__(self, X_val_m1, X_val_m2, X_val_m3, age_cond):
    
    self.X_val_m1=torch.tensor(X_val_m1,dtype=torch.float32)
    self.X_val_m2=torch.tensor(X_val_m2,dtype=torch.float32)
    self.X_val_m3=torch.tensor(X_val_m3,dtype=torch.float32)
    
    self.age_cond = torch.tensor(age_cond,dtype=torch.float32)
    #self.y_val = torch.tensor(y_val,dtype=torch.float32)

  def __len__(self):
    return len(self.X_val_m1)
   
  def __getitem__(self,idx):
    return self.X_val_m1[idx], self.X_val_m2[idx], self.X_val_m3[idx], self.age_cond[idx] 


#--------------------------------------------------------------
#--------------------------------------------------------------


class test_dataloader(torch.utils.data.Dataset):
 
  def __init__(self, X_test_m1, X_test_m2, X_test_m3, age_cond):
    
    self.X_test_m1=torch.tensor(X_test_m1,dtype=torch.float32)
    self.X_test_m2=torch.tensor(X_test_m2,dtype=torch.float32)
    self.X_test_m3=torch.tensor(X_test_m3,dtype=torch.float32)
    
    self.age_cond = torch.tensor(age_cond,dtype=torch.float32)

  def __len__(self):
    return len(self.X_test_m1)
   
  def __getitem__(self,idx):
    return self.X_test_m1[idx], self.X_test_m2[idx], self.X_test_m3[idx], self.age_cond[idx]


#--------------------------------------------------------------
#--------------------------------------------------------------


class val_ho_dataloader(torch.utils.data.Dataset):
 
  def __init__(self, X_val_ho_m1, X_val_ho_m2, X_val_ho_m3, age_cond):
    
    self.X_val_ho_m1=torch.tensor(X_val_ho_m1,dtype=torch.float32)
    self.X_val_ho_m2=torch.tensor(X_val_ho_m2,dtype=torch.float32)
    self.X_val_ho_m3=torch.tensor(X_val_ho_m3,dtype=torch.float32)
    
    self.age_cond = torch.tensor(age_cond,dtype=torch.float32)

  def __len__(self):
    return len(self.X_val_ho_m1)
   
  def __getitem__(self,idx):
    return self.X_val_ho_m1[idx], self.X_val_ho_m2[idx], self.X_val_ho_m3[idx], self.age_cond[idx]


#--------------------------------------------------------------
#--------------------------------------------------------------


def create_training_data_MVAE(X_train_total, cortical_cols, subcortical_cols, hcm_cols, age_sex_site_df, common_cols):
    
    X_train_total_m1 = pd.concat([X_train_total[common_cols], X_train_total[cortical_cols]], axis = 1)
    X_train_total_m2 = pd.concat([X_train_total[common_cols], X_train_total[subcortical_cols]], axis = 1)
    X_train_total_m3 = pd.concat([X_train_total[common_cols], X_train_total[hcm_cols]], axis = 1)

    
    m1_cols = list(cortical_cols) 
    m2_cols = list(subcortical_cols)
    m3_cols = hcm_cols

    X_train_allfold_m1, X_val_allfold_m1, y_train_allfold_m1, y_val_allfold_m1, scale_allfold_m1, age_group_trainfolds, age_group_valfolds = prepare_input_for_training(X_train_total_m1, age_sex_site_df, m1_cols)
    X_train_allfold_m2, X_val_allfold_m2, y_train_allfold_m2, y_val_allfold_m2, scale_allfold_m2, age_group_trainfolds, age_group_valfolds = prepare_input_for_training(X_train_total_m2, age_sex_site_df, m2_cols)
    X_train_allfold_m3, X_val_allfold_m3, y_train_allfold_m3, y_val_allfold_m3, scale_allfold_m3, age_group_trainfolds, age_group_valfolds = prepare_input_for_training(X_train_total_m3, age_sex_site_df, m3_cols)

    k = 1
    
    #assert(y_train_allfold_m1[k].values.all() == y_train_allfold_m2[k].values.all())
    #assert(y_val_allfold_m1[k].values.all() == y_val_allfold_m2[k].values.all())

    X_train_m1 = X_train_allfold_m1[k]
    X_val_m1 = X_val_allfold_m1[k]

    print('Number of training samples: {}'.format(len(X_train_m1)))
    print('Number of validation samples: {}'.format(len(X_val_m1)))

    X_train_m2 = X_train_allfold_m2[k]
    X_val_m2 = X_val_allfold_m2[k]
    
    X_train_m3 = X_train_allfold_m3[k]
    X_val_m3 = X_val_allfold_m3[k]

    train_age_group = age_group_trainfolds[k]
    val_age_group = age_group_valfolds[k]

    # y_train = y_train_allfold_m1[k]
    # y_val = y_val_allfold_m1[k]

    return X_train_m1, X_train_m2, X_train_m3, X_val_m1, X_val_m2, X_val_m3, train_age_group, val_age_group, m1_cols, m2_cols, m3_cols, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3


#--------------------------------------------------------------
#--------------------------------------------------------------


def concat_pred_modality(X_test_m1, X_test_m2, X_test_m3, X_test, recon_m1_test, recon_m2_test, recon_m3_test, non_roi_cols_test, m1_cols, m2_cols, m3_cols):
    
    X_pred_test_m1 = X_test_m1.copy()
    X_pred_test_m2 = X_test_m2.copy()
    X_pred_test_m3 = X_test_m3.copy()

    X_pred_test_m1[m1_cols] = pd.DataFrame(recon_m1_test.detach().numpy(), columns = m1_cols, index = X_test.index.values)
    X_pred_test_m2[m2_cols] = pd.DataFrame(recon_m2_test.detach().numpy(), columns = m2_cols, index = X_test.index.values)
    X_pred_test_m3[m3_cols] = pd.DataFrame(recon_m3_test.detach().numpy(), columns = m3_cols, index = X_test.index.values)

    X_pred_test = pd.concat([X_pred_test_m1[non_roi_cols_test], X_pred_test_m1[m1_cols], X_pred_test_m2[m2_cols], X_pred_test_m3[m3_cols]], axis = 1)

    return X_pred_test


#--------------------------------------------------------------
#--------------------------------------------------------------


def create_valho_data_MVAE(X_test, cortical_cols, subcortical_cols, hcm_cols, age_sex_site_df, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3):
    
    X_test_m1 = pd.concat([X_test[cortical_cols]], axis = 1)
    X_test_m2 = pd.concat([ X_test[subcortical_cols]], axis = 1)
    X_test_m3 = pd.concat([X_test[hcm_cols]], axis = 1)

    m1_cols = list(cortical_cols) 
    m2_cols = list(subcortical_cols)
    m3_cols = hcm_cols

    X_test_scaled_m1 = scale_allfold_m1[1].transform(X_test_m1[m1_cols])
    X_test_m1[m1_cols] = pd.DataFrame(X_test_scaled_m1, columns = m1_cols, index = X_test.index.values)

    X_test_scaled_m2 = scale_allfold_m2[1].transform(X_test_m2[m2_cols])
    X_test_m2[m2_cols] = pd.DataFrame(X_test_scaled_m2, columns = m2_cols, index = X_test.index.values)

    X_test_scaled_m3 = scale_allfold_m3[1].transform(X_test_m3[m3_cols])
    X_test_m3[m3_cols] = pd.DataFrame(X_test_scaled_m3, columns = m3_cols, index = X_test.index.values)

    X_test_org = pd.concat([X_test_m1[m1_cols], X_test_m2[m2_cols], X_test_m3[m3_cols]], axis = 1)


    test_index = X_test.index.values
    test_age_group = age_sex_site_df.loc[test_index]

    #X_pred_test = X_test.copy()
    #X_pred_test[fs_cols] = vae.predict([X_test[fs_cols], test_age_group])

    test_data = test_dataloader(X_test_m1[m1_cols].to_numpy(), X_test_m2[m2_cols].to_numpy(), X_test_m3[m3_cols].to_numpy(), test_age_group.values)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(X_test), shuffle=False)

    return test_loader, X_test_org, test_age_group, X_test_m1, X_test_m2, X_test_m3 


#--------------------------------------------------------------
#--------------------------------------------------------------


def concat_pred_modality_valho(X_test_m1, X_test_m2, X_test_m3, X_test, recon_m1_test, recon_m2_test, recon_m3_test, m1_cols, m2_cols, m3_cols):
    
    X_pred_test_m1 = X_test_m1.copy()
    X_pred_test_m2 = X_test_m2.copy()
    X_pred_test_m3 = X_test_m3.copy()

    X_pred_test_m1[m1_cols] = pd.DataFrame(recon_m1_test.detach().numpy(), columns = m1_cols, index = X_test.index.values)
    X_pred_test_m2[m2_cols] = pd.DataFrame(recon_m2_test.detach().numpy(), columns = m2_cols, index = X_test.index.values)
    X_pred_test_m3[m3_cols] = pd.DataFrame(recon_m3_test.detach().numpy(), columns = m3_cols, index = X_test.index.values)

    X_pred_test = pd.concat([X_pred_test_m1[m1_cols], X_pred_test_m2[m2_cols], X_pred_test_m3[m3_cols]], axis = 1)

    return X_pred_test

    
def create_test_data_MVAE(X_test, non_roi_cols_test, cortical_cols, subcortical_cols, hcm_cols, age_sex_site_df, scale_allfold_m1, scale_allfold_m2, scale_allfold_m3):
    
    X_test_m1 = pd.concat([X_test[non_roi_cols_test], X_test[cortical_cols]], axis = 1)
    X_test_m2 = pd.concat([X_test[non_roi_cols_test], X_test[subcortical_cols]], axis = 1)
    X_test_m3 = pd.concat([X_test[non_roi_cols_test], X_test[hcm_cols]], axis = 1)

    m1_cols = list(cortical_cols) 
    m2_cols = list(subcortical_cols)
    m3_cols = hcm_cols

    X_test_scaled_m1 = scale_allfold_m1[1].transform(X_test_m1[m1_cols])
    X_test_m1[m1_cols] = pd.DataFrame(X_test_scaled_m1, columns = m1_cols, index = X_test.index.values)

    X_test_scaled_m2 = scale_allfold_m2[1].transform(X_test_m2[m2_cols])
    X_test_m2[m2_cols] = pd.DataFrame(X_test_scaled_m2, columns = m2_cols, index = X_test.index.values)

    X_test_scaled_m3 = scale_allfold_m3[1].transform(X_test_m3[m3_cols])
    X_test_m3[m3_cols] = pd.DataFrame(X_test_scaled_m3, columns = m3_cols, index = X_test.index.values)

    X_test_org = pd.concat([X_test[non_roi_cols_test], X_test_m1[m1_cols], X_test_m2[m2_cols], X_test_m3[m3_cols]], axis = 1)

    test_index = X_test.index.values
    test_age_group = age_sex_site_df.loc[test_index]

    #X_pred_test = X_test.copy()
    #X_pred_test[fs_cols] = vae.predict([X_test[fs_cols], test_age_group])

    test_data = test_dataloader(X_test_m1[m1_cols].to_numpy(), X_test_m2[m2_cols].to_numpy(), X_test_m3[m3_cols].to_numpy(), test_age_group.values)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(X_test), shuffle=False)

    return test_loader, X_test_org, test_age_group, X_test_m1, X_test_m2, X_test_m3 



#-------------------------------------------------------------
#-------------------------------------------------------------

def prepare_input_for_training(X_train_total, age_sex_df, fs_cols):
    
    skf = StratifiedKFold(n_splits= 10, shuffle = False)
    #X_train_total = X_train_org.reset_index(drop = True)

    X_train_allfold = {}
    X_val_allfold = {}
    age_group_trainfolds = {}
    age_group_valfolds = {}
    scale_allfold = {}

    train_index_allk = {}
    val_index_allk = {}
    
    y_train_allfold = {}
    y_val_allfold = {}


    for k in range(1,11):
        for train_index, val_index in skf.split(X_train_total, X_train_total['dataset'].values):

            train_index_allk[k] = train_index
            val_index_allk[k] = val_index

            X_train, X_val = X_train_total.loc[train_index], X_train_total.loc[val_index]
            y_train, y_val = X_train_total[['dataset']].loc[train_index], X_train_total[['dataset']].loc[val_index]

            y_train_allfold[k] = y_train
            y_val_allfold[k] = y_val
            
            age_group_trainfolds[k] = age_sex_df.loc[train_index]
            age_group_valfolds[k] = age_sex_df.loc[val_index]

            scale = MinMaxScaler().fit(X_train[fs_cols])
            X_train_scaled = scale.transform(X_train[fs_cols])
            X_val_scaled = scale.transform(X_val[fs_cols])

            X_train_allfold[k] = X_train_scaled
            X_val_allfold[k] = X_val_scaled
            scale_allfold[k] = scale
        
    return X_train_allfold, X_val_allfold, y_train_allfold, y_val_allfold, scale_allfold, age_group_trainfolds, age_group_valfolds

