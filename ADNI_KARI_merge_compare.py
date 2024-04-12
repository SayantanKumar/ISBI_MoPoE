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

#########################################################################
#------------------- Relevant funtions ---------------------------------
#########################################################################

def ADNI_vs_KARI_MRI_comparison(ADNI_MRI_vol, KARI_MRI_vol, cortical_cols_MRI, subcortical_cols_MRI):
    
    from scipy.stats import f_oneway
       
    ADNI_MRI_vol_cdr0 = ADNI_MRI_vol.loc[ADNI_MRI_vol.CDGLOBAL == 0]
    ADNI_MRI_vol_cdr0_5 = ADNI_MRI_vol.loc[ADNI_MRI_vol.CDGLOBAL == 0.5]
    ADNI_MRI_vol_cdr1 = ADNI_MRI_vol.loc[ADNI_MRI_vol.CDGLOBAL >= 1]

    KARI_MRI_vol_cdr0 = KARI_MRI_vol.loc[KARI_MRI_vol.cdr == 0]
    KARI_MRI_vol_cdr0_5 = KARI_MRI_vol.loc[KARI_MRI_vol.cdr == 0.5]
    KARI_MRI_vol_cdr1 = KARI_MRI_vol.loc[KARI_MRI_vol.cdr >= 1]
     
    p_list_cdr0_cort = []
    p_list_cdr0_5_cort = []
    p_list_cdr1_cort = []
    p_list_cdr0_subcort = []
    p_list_cdr0_5_subcort = []
    p_list_cdr1_subcort = []
    
    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_MRI_vol_cdr0[mri_col], KARI_MRI_vol_cdr0[mri_col])
        p_list_cdr0_cort.append(p)

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_MRI_vol_cdr0_5[mri_col], KARI_MRI_vol_cdr0_5[mri_col])
        p_list_cdr0_5_cort.append(p)

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_MRI_vol_cdr1[mri_col], KARI_MRI_vol_cdr1[mri_col])
        p_list_cdr1_cort.append(p)
      
    #---------------------------
    
    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_MRI_vol_cdr0[mri_col], KARI_MRI_vol_cdr0[mri_col])
        p_list_cdr0_subcort.append(p)

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_MRI_vol_cdr0_5[mri_col], KARI_MRI_vol_cdr0_5[mri_col])
        p_list_cdr0_5_subcort.append(p)

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_MRI_vol_cdr1[mri_col], KARI_MRI_vol_cdr1[mri_col])
        p_list_cdr1_subcort.append(p)
        
    return p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort



def plot_cortical_maps(p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, cortical_cols_MRI):
    
    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr0_cort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical volume) \n for CDR = 0')
    print('cort_cdr0 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_cort), min(p_list_cdr0_cort)))

    
    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr0_5_cort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical volume) \n for CDR = 0.5')
    print('cort_cdr0_5 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_5_cort), min(p_list_cdr0_5_cort)))

    
    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr1_cort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical volume) \n for CDR >= 1')
    print('cort_cdr1 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr1_cort), min(p_list_cdr1_cort)))

    

def plot_subcortical_maps(p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort, subcortical_cols_MRI):
    
    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr0_subcort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical volume) \n for CDR = 0')
    print('subcort_cdr0 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_subcort), min(p_list_cdr0_subcort)))

    
    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr0_5_subcort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical volume) \n for CDR = 0.5')
    print('subcort_cdr0_5 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_5_subcort), min(p_list_cdr0_5_subcort)))

    
    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr1_subcort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical volume) \n for CDR >= 1')
    print('cort_cdr1 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr1_subcort), min(p_list_cdr1_subcort)))

    

def plot_p_maps_MRI(ADNI_MRI_vol, KARI_MRI_vol, cortical_cols_MRI, subcortical_cols_MRI):
        
    p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort = ADNI_vs_KARI_MRI_comparison(ADNI_MRI_vol, KARI_MRI_vol, cortical_cols_MRI, subcortical_cols_MRI)
    plot_cortical_maps(p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, cortical_cols_MRI)
    plot_subcortical_maps(p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort, subcortical_cols_MRI)
    
#-------------------------------------------------------
#-------------------------------------------------------
  
def ADNI_vs_KARI_amyloid_comparison(ADNI_amyloid, KARI_amyloid, cortical_cols_MRI, subcortical_cols_MRI):
    
    from scipy.stats import f_oneway

    ADNI_amyloid_present = ADNI_amyloid.loc[ADNI_amyloid['amyloid_present'] == 'yes']
    KARI_amyloid_present = KARI_amyloid.loc[KARI_amyloid['amyloid_present'] == 'yes']

    ADNI_amyloid_cdr0 = ADNI_amyloid_present.loc[ADNI_amyloid_present.CDGLOBAL == 0]
    ADNI_amyloid_cdr0_5 = ADNI_amyloid_present.loc[ADNI_amyloid_present.CDGLOBAL == 0.5]
    ADNI_amyloid_cdr1 = ADNI_amyloid_present.loc[ADNI_amyloid_present.CDGLOBAL >= 1]

    KARI_amyloid_cdr0 = KARI_amyloid_present.loc[KARI_amyloid_present.cdr == 0]
    KARI_amyloid_cdr0_5 = KARI_amyloid_present.loc[KARI_amyloid_present.cdr == 0.5]
    KARI_amyloid_cdr1 = KARI_amyloid_present.loc[KARI_amyloid_present.cdr >= 1]

    p_list_cdr0_cort = []
    p_list_cdr0_5_cort = []
    p_list_cdr1_cort = []
    p_list_cdr0_subcort = []
    p_list_cdr0_5_subcort = []
    p_list_cdr1_subcort = []

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_amyloid_cdr0[mri_col], KARI_amyloid_cdr0[mri_col])
        p_list_cdr0_cort.append(p)

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_amyloid_cdr0_5[mri_col], KARI_amyloid_cdr0_5[mri_col])
        p_list_cdr0_5_cort.append(p)

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_amyloid_cdr1[mri_col], KARI_amyloid_cdr1[mri_col])
        p_list_cdr1_cort.append(p)

        #---------------------------

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_amyloid_cdr0[mri_col], KARI_amyloid_cdr0[mri_col])
        p_list_cdr0_subcort.append(p)

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_amyloid_cdr0_5[mri_col], KARI_amyloid_cdr0_5[mri_col])
        p_list_cdr0_5_subcort.append(p)

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_amyloid_cdr1[mri_col], KARI_amyloid_cdr1[mri_col])
        p_list_cdr1_subcort.append(p)

    return p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort



def plot_p_maps_amyloid(ADNI_amyloid, KARI_amyloid, cortical_cols_MRI, subcortical_cols_MRI):
    
    p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort = ADNI_vs_KARI_amyloid_comparison(ADNI_amyloid, KARI_amyloid, cortical_cols_MRI, subcortical_cols_MRI)
    
    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr0_cort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical amyloid SUVR) \n for CDR = 0')
    print('cort_cdr0 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_cort), min(p_list_cdr0_cort)))


    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr0_5_cort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical amyloid SUVR) \n for CDR = 0.5')
    print('cort_cdr0_5 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_5_cort), min(p_list_cdr0_5_cort)))


    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr1_cort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical amyloid SUVR) \n for CDR >= 1')
    print('cort_cdr1 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr1_cort), min(p_list_cdr1_cort)))

    #-------------------------------------------------------

    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr0_subcort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (subcortical amyloid SUVR) \n for CDR = 0')
    print('subcort_cdr0 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_subcort), min(p_list_cdr0_subcort)))


    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr0_5_subcort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (subcortical amyloid SUVR) \n for CDR = 0.5')
    print('subcort_cdr0_5 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_5_subcort), min(p_list_cdr0_5_subcort)))


    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr1_subcort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (subcortical amyloid SUVR) \n for CDR >= 1')
    print('cort_cdr1 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr1_subcort), min(p_list_cdr1_subcort)))


#plot_p_maps_amyloid(ADNI_amyloid, KARI_amyloid, cortical_cols_MRI, subcortical_cols_MRI)

#-----------------------------------------------------
#-----------------------------------------------------

def ADNI_vs_KARI_tau_comparison(ADNI_tau, KARI_tau, cortical_cols_MRI, subcortical_cols_MRI):
    
    from scipy.stats import f_oneway

    ADNI_tau_present = ADNI_tau.loc[ADNI_tau['tau_present'] == 'yes']
    KARI_tau_present = KARI_tau.loc[KARI_tau['tau_present'] == 'yes']

    ADNI_tau_cdr0 = ADNI_tau_present.loc[ADNI_tau_present.CDGLOBAL == 0]
    ADNI_tau_cdr0_5 = ADNI_tau_present.loc[ADNI_tau_present.CDGLOBAL == 0.5]
    ADNI_tau_cdr1 = ADNI_tau_present.loc[ADNI_tau_present.CDGLOBAL >= 1]

    KARI_tau_cdr0 = KARI_tau_present.loc[KARI_tau_present.cdr == 0]
    KARI_tau_cdr0_5 = KARI_tau_present.loc[KARI_tau_present.cdr == 0.5]
    KARI_tau_cdr1 = KARI_tau_present.loc[KARI_tau_present.cdr >= 1]

    p_list_cdr0_cort = []
    p_list_cdr0_5_cort = []
    p_list_cdr1_cort = []
    p_list_cdr0_subcort = []
    p_list_cdr0_5_subcort = []
    p_list_cdr1_subcort = []

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_tau_cdr0[mri_col], KARI_tau_cdr0[mri_col])
        p_list_cdr0_cort.append(p)

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_tau_cdr0_5[mri_col], KARI_tau_cdr0_5[mri_col])
        p_list_cdr0_5_cort.append(p)

    for mri_col in cortical_cols_MRI:
        F, p = f_oneway(ADNI_tau_cdr1[mri_col], KARI_tau_cdr1[mri_col])
        p_list_cdr1_cort.append(p)

        #---------------------------

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_tau_cdr0[mri_col], KARI_tau_cdr0[mri_col])
        p_list_cdr0_subcort.append(p)

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_tau_cdr0_5[mri_col], KARI_tau_cdr0_5[mri_col])
        p_list_cdr0_5_subcort.append(p)

    for mri_col in subcortical_cols_MRI:
        F, p = f_oneway(ADNI_tau_cdr1[mri_col], KARI_tau_cdr1[mri_col])
        p_list_cdr1_subcort.append(p)

    return p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort



def plot_p_maps_tau(ADNI_tau, KARI_tau, cortical_cols_MRI, subcortical_cols_MRI):
    
    p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort = ADNI_vs_KARI_tau_comparison(ADNI_tau, KARI_tau, cortical_cols_MRI, subcortical_cols_MRI)
    
    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr0_cort)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical tau SUVR) \n for CDR = 0')
    print('cort_cdr0 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_cort), min(p_list_cdr0_cort)))


    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr0_5_cort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical tau SUVR) \n for CDR = 0.5')
    print('cort_cdr0_5 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_5_cort), min(p_list_cdr0_5_cort)))


    ggseg_python.plot_dk(dict(zip(cortical_cols_MRI, p_list_cdr1_cort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical tau SUVR) \n for CDR >= 1')
    print('cort_cdr1 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr1_cort), min(p_list_cdr1_cort)))

    #-------------------------------------------------------

    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr0_subcort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (subcortical tau SUVR) \n for CDR = 0')
    print('subcort_cdr0 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_subcort), min(p_list_cdr0_subcort)))


    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr0_5_subcort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (subcortical tau SUVR) \n for CDR = 0.5')
    print('subcort_cdr0_5 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr0_5_subcort), min(p_list_cdr0_5_subcort)))


    ggseg_python.plot_aseg(dict(zip(subcortical_cols_MRI, p_list_cdr1_subcort)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (subcortical tau SUVR) \n for CDR >= 1')
    print('cort_cdr1 (subcortical) : Max = {}, Min = {}'.format(max(p_list_cdr1_subcort), min(p_list_cdr1_subcort)))



#plot_p_maps_tau(ADNI_tau, KARI_tau, cortical_cols_MRI, subcortical_cols_MRI)

#---------------------------------------------------------------
#---------------------------------------------------------------

def merge_MRI_both(ADNI_MRI, KARI_MRI, mri_vol_cols):
    
    ADNI_MRI_final= ADNI_MRI.rename(columns = {'RID':'ID', 'AGE':'Age', 'PTGENDER':'Sex', 'CDGLOBAL':'cdr'}).drop(columns = ['DX_bl', 'Intracranial_vol']).reindex(columns=mri_vol_cols)
    KARI_MRI_final = KARI_MRI.rename(columns = {'sex':'Sex'}).reindex(columns=mri_vol_cols)
    ADNI_MRI_final['dataset'] = 'ADNI'
    KARI_MRI_final['dataset'] = 'KARI'

    MRI_both = pd.concat([ADNI_MRI_final, KARI_MRI_final]).reset_index(drop = True)

    MRI_both = MRI_both.rename(columns={c: c+'_Nvol' for c in MRI_both.columns if c not in ['ID', 'Age', 'Sex', 'MR_Date', 'CDR_Date', 'cdr', 'dataset']})

    return MRI_both


def merge_amyloid_both(ADNI_amyloid, KARI_amyloid, amyloid_SUVR_cols):
    
    ADNI_amyloid_final= ADNI_amyloid.rename(columns = {'RID':'ID', 'AGE':'Age', 'PTGENDER':'Sex', 'CDGLOBAL':'cdr'}).drop(columns = ['DX_bl']).reindex(columns=amyloid_SUVR_cols)
    KARI_amyloid_final = KARI_amyloid.rename(columns = {'sex':'Sex'}).reindex(columns=amyloid_SUVR_cols)
    ADNI_amyloid_final['dataset'] = 'ADNI'
    KARI_amyloid_final['dataset'] = 'KARI'

    amyloid_both = pd.concat([ADNI_amyloid_final, KARI_amyloid_final]).reset_index(drop = True)

    amyloid_both = amyloid_both.rename(columns={c: c+'_Asuvr' for c in amyloid_both.columns if c not in ['ID', 'Age', 'Sex', 'MR_Date', 'amyloid_present', 'amyloid_date',  'CDR_Date', 'cdr', 'dataset']})

    return amyloid_both


def merge_tau_both(ADNI_tau, KARI_tau, tau_SUVR_cols):
    
    ADNI_tau_final= ADNI_tau.rename(columns = {'RID':'ID', 'AGE':'Age', 'PTGENDER':'Sex', 'CDGLOBAL':'cdr'}).drop(columns = ['DX_bl']).reindex(columns=tau_SUVR_cols)
    KARI_tau_final = KARI_tau.rename(columns = {'sex':'Sex'}).reindex(columns=tau_SUVR_cols)
    ADNI_tau_final['dataset'] = 'ADNI'
    KARI_tau_final['dataset'] = 'KARI'

    tau_both = pd.concat([ADNI_tau_final, KARI_tau_final]).reset_index(drop = True)

    tau_both = tau_both.rename(columns={c: c+'_Tsuvr' for c in tau_both.columns if c not in ['ID', 'Age', 'Sex', 'MR_Date', 'tau_present', 'tau_date',  'CDR_Date', 'cdr', 'dataset']})

    return tau_both


def merge_allmod(MRI_both, amyloid_both, tau_both):
    
#     MRI_both = merge_MRI_both(ADNI_MRI, KARI_MRI, mri_vol_cols)
#     amyloid_both = merge_amyloid_both(ADNI_amyloid, KARI_amyloid, amyloid_SUVR_cols)
#     tau_both = merge_tau_both(ADNI_tau, KARI_tau, tau_SUVR_cols)
    
    common_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'CDR_Date', 'cdr']
    data_frames = [MRI_both, amyloid_both, tau_both]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on = common_cols, how='outer'), data_frames)
    #df_merged.loc[(df_merged.dataset == 'ADNI') & (df_merged.amyloid_present == 'yes') & (df_merged.tau_present == 'yes')].cdr.value_counts()

    # Select patients with amyloid always present and tau (present/absent)
    a_n_merged = df_merged.loc[df_merged['amyloid_present'] == 'yes']

    MRI_vol_cols = [col for col in a_n_merged.columns if col.endswith('_Nvol')]
    amyloid_SUVR_cols = [col for col in a_n_merged.columns if col.endswith('_Asuvr')]
    tau_SUVR_cols = [col for col in a_n_merged.columns if col.endswith('_Tsuvr')]

    print('Patients having both MRI and amyloid = {}'.format(len(a_n_merged)))
    print('Patients having only MRI and amyloid but no tau = {}'.format(len(a_n_merged.loc[a_n_merged.tau_present == 'no'])))
    print('Patients having all MRI and amyloid = {}'.format(len(a_n_merged.loc[a_n_merged.tau_present == 'yes'])))
    
    return a_n_merged, df_merged, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols
    

#---------------------------------------------------------------
#---------------------------------------------------------------

def adni_amyloid_pos(adni_amyloid, a_n_merged):
    
    adni_amyloid_cutoff = adni_amyloid[['RID', 'EXAMDATE', 'SUMMARYSUVR_WHOLECEREBNORM', 'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF']].rename(columns = {'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF':'amyloid_positive'})
    adni_amyloid_cutoff = adni_amyloid_cutoff.sort_values(by = ['RID', 'EXAMDATE', 'amyloid_positive'])#.drop_duplicates(subset = 'RID', keep = 'first').reset_index(drop = True)
    adni_amyloid_cutoff['amyloid_centiloid'] = 196.9 * adni_amyloid_cutoff['SUMMARYSUVR_WHOLECEREBNORM'] - 196.03
    
    apos_adni = pd.merge(adni_amyloid_cutoff, a_n_merged.loc[a_n_merged.dataset == 'ADNI'][['ID', 'amyloid_date', 'cdr']], left_on = ['RID', 'EXAMDATE'], right_on = ['ID', 'amyloid_date'], how = 'right')
    apos_adni['stage'] = np.nan
    
    apos_adni.loc[(apos_adni['amyloid_positive'] == 0) & (apos_adni['cdr'] == 0), 'stage'] = 'cdr = 0 amyloid negative'
    apos_adni.loc[(apos_adni['amyloid_positive'] == 1) & (apos_adni['cdr'] == 0), 'stage'] = 'cdr = 0 amyloid positive'
    apos_adni.loc[(apos_adni['cdr'] == 0.5), 'stage'] = 'cdr = 0.5'
    apos_adni.loc[(apos_adni['cdr'] >= 1), 'stage'] = 'cdr >= 1'
    apos_adni['dataset'] = 'ADNI'
    
    return apos_adni



def kari_amyloid_pos(av45, a_n_merged):
    
    kari_amyloid_cutoff = av45[['ID', 'PET_Date', 'av45_fsuvr_tot_cortmean']]
    kari_amyloid_cutoff['amyloid_positive'] = np.nan
    kari_amyloid_cutoff['amyloid_centiloid'] = 164.6 * kari_amyloid_cutoff['av45_fsuvr_tot_cortmean'] - 181
    kari_amyloid_cutoff.loc[kari_amyloid_cutoff.av45_fsuvr_tot_cortmean > 1.24, 'amyloid_positive'] = 1
    kari_amyloid_cutoff.loc[kari_amyloid_cutoff.av45_fsuvr_tot_cortmean <= 1.24, 'amyloid_positive'] = 0

    kari_amyloid_cutoff = kari_amyloid_cutoff.sort_values(by = ['ID', 'PET_Date', 'amyloid_positive'])
    temp = a_n_merged.loc[a_n_merged.dataset == 'KARI'][['ID', 'amyloid_date', 'cdr']]
    temp['amyloid_date']=temp['amyloid_date'].astype('datetime64[ns]')

    apos_kari = pd.merge(kari_amyloid_cutoff, temp, left_on = ['ID', 'PET_Date'], right_on = ['ID', 'amyloid_date'], how = 'right')
    apos_kari.loc[(apos_kari['amyloid_positive'] == 0) & (apos_kari['cdr'] == 0), 'stage'] = 'cdr = 0 amyloid negative'
    apos_kari.loc[(apos_kari['amyloid_positive'] == 1) & (apos_kari['cdr'] == 0), 'stage'] = 'cdr = 0 amyloid positive'
    apos_kari.loc[(apos_kari['cdr'] == 0.5), 'stage'] = 'cdr = 0.5'
    apos_kari.loc[(apos_kari['cdr'] >= 1), 'stage'] = 'cdr >= 1'
    apos_kari['dataset'] = 'KARI'
    
    return apos_kari


