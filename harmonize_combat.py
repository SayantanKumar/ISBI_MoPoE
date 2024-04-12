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

tadpole_challenge_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/tadpole_challenge'
roi_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis'


#########################################################################
#------------------- Relevant funtions ---------------------------------
#########################################################################

def harmonization_step_stratify(a_n_merged, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols):
    
    if len(os.listdir('./saved_models/harmonization_models')) > 1:
            
        os.remove('./saved_models/harmonization_models/model_mri')
        os.remove('./saved_models/harmonization_models/model_amyloid')
        os.remove('./saved_models/harmonization_models/model_tau')

    
    # Prepare training data for harmonization learn (healthy controls)
    
    cn_mri = np.array(a_n_merged.loc[a_n_merged['stage'] == 'cdr = 0 amyloid negative'][MRI_vol_cols].reset_index(drop = True))
    cn_amyloid = np.array(a_n_merged.loc[a_n_merged['stage'] == 'cdr = 0 amyloid negative'][amyloid_SUVR_cols].reset_index(drop = True))
    cn_tau = np.array(a_n_merged.loc[(a_n_merged['stage'] == 'cdr = 0 amyloid negative') & (a_n_merged['tau_present'] == 'yes')][tau_SUVR_cols].reset_index(drop = True))

    rest_mri = np.array(a_n_merged.loc[a_n_merged['stage'] != 'cdr = 0 amyloid negative'][MRI_vol_cols].reset_index(drop = True))
    rest_amyloid = np.array(a_n_merged.loc[a_n_merged['stage'] != 'cdr = 0 amyloid negative'][amyloid_SUVR_cols].reset_index(drop = True))
    rest_tau = np.array(a_n_merged.loc[(a_n_merged['stage'] != 'cdr = 0 amyloid negative') & (a_n_merged['tau_present'] == 'yes')][tau_SUVR_cols].reset_index(drop = True))

    # Prepare covariate data 
    covars_CN = a_n_merged.loc[a_n_merged['stage'] == 'cdr = 0 amyloid negative'].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M']]
    covars_tau_CN = a_n_merged.loc[(a_n_merged['stage'] == 'cdr = 0 amyloid negative') & (a_n_merged['tau_present'] == 'yes')].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M']]

    # Learning harmoniztion individually on the 3 modalities 
    model_mri, mri_harm = harmonizationLearn(cn_mri, covars_CN, smooth_terms=['AGE'])
    model_amyloid, amyloid_harm = harmonizationLearn(cn_amyloid, covars_CN, smooth_terms=['AGE'])
    model_tau, tau_harm = harmonizationLearn(cn_tau, covars_tau_CN, smooth_terms=['AGE'])
    

    # Saving the models learnt on CNs to be applied on disease patients
    if len(os.listdir('./saved_models/harmonization_models/')) == 1:
        print('Saving harmonization models')
        saveHarmonizationModel(model_mri, './saved_models/harmonization_models/model_mri')
        saveHarmonizationModel(model_amyloid, './saved_models/harmonization_models/model_amyloid')
        saveHarmonizationModel(model_tau, './saved_models/harmonization_models/model_tau')


    cn_idx = a_n_merged.loc[a_n_merged['stage'] == 'cdr = 0 amyloid negative'].index.values
    cn_tau_idx = a_n_merged.loc[(a_n_merged['stage'] != 'cdr = 0 amyloid negative') & (a_n_merged['tau_present'] == 'yes')].index.values
    
    mri_harm_CN = pd.DataFrame(mri_harm, columns = MRI_vol_cols)
    amyloid_harm_CN = pd.DataFrame(amyloid_harm, columns = amyloid_SUVR_cols)
    tau_harm_CN = pd.DataFrame(tau_harm, columns = tau_SUVR_cols)

    # Loading saved harmonization models
    print('Loading pre-trained harmonization models')
    load_model_mri = loadHarmonizationModel('./saved_models/harmonization_models/model_mri')
    load_model_amyloid = loadHarmonizationModel('./saved_models/harmonization_models/model_amyloid')
    load_model_tau = loadHarmonizationModel('./saved_models/harmonization_models/model_tau')

    # Applying pretrained harmonization models on disease patients (for each modality)

    covars_rest = a_n_merged.loc[a_n_merged['stage'] != 'cdr = 0 amyloid negative'].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M']]
    covars_tau_rest = a_n_merged.loc[(a_n_merged['stage'] != 'cdr = 0 amyloid negative') & (a_n_merged['tau_present'] == 'yes')].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M']]

    apply_model_mri = harmonizationApply(rest_mri, covars_rest, load_model_mri)
    apply_model_amyloid = harmonizationApply(rest_amyloid, covars_rest, load_model_amyloid)
    apply_model_tau = harmonizationApply(rest_tau, covars_tau_rest, load_model_tau)
    
    rest_idx = a_n_merged.loc[a_n_merged['stage'] != 'cdr = 0 amyloid negative'].index.values
    rest_tau_idx = a_n_merged.loc[(a_n_merged['stage'] != 'cdr = 0 amyloid negative') & (a_n_merged['tau_present'] == 'yes')].index.values
    
    mri_harm_rest = pd.DataFrame(apply_model_mri, columns = MRI_vol_cols)
    amyloid_harm_rest = pd.DataFrame(apply_model_amyloid, columns = amyloid_SUVR_cols)
    tau_harm_rest = pd.DataFrame(apply_model_tau, columns = tau_SUVR_cols)
    
    return mri_harm_CN, amyloid_harm_CN, tau_harm_CN, mri_harm_rest, amyloid_harm_rest, tau_harm_rest, covars_CN, covars_rest


#--------------------------------------------------------

def harmonization_step_whole(a_n_merged, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, extra_cov = None):
    
    # Prepare training data for harmonization learn (healthy controls + disease)
    
    a_n_merged_temp = a_n_merged.copy()
    a_n_merged_temp['disease_status'] = 'empty'
    a_n_merged_temp.loc[a_n_merged_temp.stage == 'cdr = 0 amyloid negative', 'disease_status'] = 0
    a_n_merged_temp.loc[a_n_merged_temp.stage == 'cdr = 0 amyloid positive', 'disease_status'] = 1
    a_n_merged_temp.loc[a_n_merged_temp.stage == 'cdr = 0.5', 'disease_status'] = 2
    a_n_merged_temp.loc[a_n_merged_temp.stage == 'cdr >= 1', 'disease_status'] = 3
    
    stage_cat = pd.get_dummies(a_n_merged_temp['stage'])
    a_n_merged_temp = pd.concat([a_n_merged_temp, stage_cat], axis = 1)

    data_mri = np.array(a_n_merged_temp[MRI_vol_cols].reset_index(drop = True))
    data_amyloid = np.array(a_n_merged_temp[amyloid_SUVR_cols].reset_index(drop = True))
    data_tau = np.array(a_n_merged_temp.loc[(a_n_merged['tau_present'] == 'yes')][tau_SUVR_cols].reset_index(drop = True))

    # Prepare covariate data
    #--------------------- 
    if extra_cov == None:
        covars = a_n_merged_temp.rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M']]
        covars_tau = a_n_merged_temp.loc[(a_n_merged['tau_present'] == 'yes')].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M']]

        # Learning harmoniztion individually on the 3 modalities 
        model_mri, mri_harm = harmonizationLearn(data_mri, covars, smooth_terms=['AGE'])
        model_amyloid, amyloid_harm = harmonizationLearn(data_amyloid, covars, smooth_terms=['AGE'])
        model_tau, tau_harm = harmonizationLearn(data_tau, covars_tau, smooth_terms=['AGE'])
    
    #--------------------- 
    if extra_cov == 'amyloid_centiloid':
        covars = a_n_merged_temp.rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'amyloid_centiloid']]
        covars_tau = a_n_merged_temp.loc[(a_n_merged['tau_present'] == 'yes')].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'amyloid_centiloid']]

        # Learning harmoniztion individually on the 3 modalities 
        model_mri, mri_harm = harmonizationLearn(data_mri, covars, smooth_terms=['AGE'])
        model_amyloid, amyloid_harm = harmonizationLearn(data_amyloid, covars, smooth_terms=['AGE'])
        model_tau, tau_harm = harmonizationLearn(data_tau, covars_tau, smooth_terms=['AGE'])   
    
    #---------------------
    if extra_cov == 'amyloid_positive':
        covars = a_n_merged_temp.rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'amyloid_positive']]
        covars_tau = a_n_merged_temp.loc[(a_n_merged_temp['tau_present'] == 'yes')].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'amyloid_positive', 'disease_status']]

        # Learning harmoniztion individually on the 3 modalities 
        model_mri, mri_harm = harmonizationLearn(data_mri, covars, smooth_terms=['AGE'])
        model_amyloid, amyloid_harm = harmonizationLearn(data_amyloid, covars, smooth_terms=['AGE'])
        model_tau, tau_harm = harmonizationLearn(data_tau, covars_tau, smooth_terms=['AGE'])
        
    #---------------------    
    if extra_cov == 'disease_status':
        covars = a_n_merged_temp.rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'disease_status']]
        covars_tau = a_n_merged_temp.loc[(a_n_merged_temp['tau_present'] == 'yes')].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'disease_status']]

        # Learning harmoniztion individually on the 3 modalities 
        model_mri, mri_harm = harmonizationLearn(data_mri, covars, smooth_terms=['AGE'])
        model_amyloid, amyloid_harm = harmonizationLearn(data_amyloid, covars, smooth_terms=['AGE'])
        model_tau, tau_harm = harmonizationLearn(data_tau, covars_tau, smooth_terms=['AGE'])
        
    #----------------------------  
    
    if extra_cov == 'stage_cat':
        covars = a_n_merged_temp.rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1']]
        covars_tau = a_n_merged_temp.loc[(a_n_merged_temp['tau_present'] == 'yes')].rename(columns = {'dataset':'SITE', 'Age':'AGE', 'Sex':'SEX_M'})[['SITE', 'AGE', 'SEX_M', 'cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1']]
        
        # Learning harmoniztion individually on the 3 modalities 
        model_mri, mri_harm = harmonizationLearn(data_mri, covars, smooth_terms=['AGE'])
        model_amyloid, amyloid_harm = harmonizationLearn(data_amyloid, covars, smooth_terms=['AGE'])
        model_tau, tau_harm = harmonizationLearn(data_tau, covars_tau, smooth_terms=['AGE'])
        
    
    mri_harm = pd.DataFrame(mri_harm, columns = MRI_vol_cols)
    amyloid_harm = pd.DataFrame(amyloid_harm, columns = amyloid_SUVR_cols)
    tau_harm = pd.DataFrame(tau_harm, columns = tau_SUVR_cols)

    return mri_harm, amyloid_harm, tau_harm, covars, covars_tau, a_n_merged_temp 


#-----------------------------------------------------------------
#-----------------------------------------------------------------

def plot_harmonization_effect(a_n_merged_CN, a_n_merged_harm_CN, a_n_merged_rest, a_n_merged_harm_rest, mri_col, amyloid_col, tau_col):
    
    #-------------------- Effect of harmonization for healthy controls -------
    plt.subplots(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.title('Before harmonization')
    sns.boxplot(y=a_n_merged_CN[mri_col], x=a_n_merged_CN["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.subplot(1,2,2)
    plt.title('After harmonization')
    sns.boxplot(y=a_n_merged_harm_CN[mri_col], x=a_n_merged_harm_CN["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.suptitle('Harmonization of {} (CN only)'.format(mri_col))


    plt.subplots(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.title('Before harmonization')
    sns.boxplot(y=a_n_merged_CN[amyloid_col], x=a_n_merged_CN["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.subplot(1,2,2)
    plt.title('After harmonization')
    sns.boxplot(y=a_n_merged_harm_CN[amyloid_col], x=a_n_merged_harm_CN["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.suptitle('Harmonization of {} (CN only)'.format(amyloid_col))


    plt.subplots(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.title('Before harmonization')
    sns.boxplot(y=a_n_merged_CN[tau_col], x=a_n_merged_CN["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.subplot(1,2,2)
    plt.title('After harmonization')
    sns.boxplot(y=a_n_merged_harm_CN[tau_col], x=a_n_merged_harm_CN["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.suptitle('Harmonization of {} (CN only)'.format(tau_col))

    #-------------------- Effect of harmonization for disease patients -------

    plt.subplots(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.title('Before harmonization')
    sns.boxplot(y=a_n_merged_rest[mri_col], x=a_n_merged_rest["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.subplot(1,2,2)
    plt.title('After harmonization')
    sns.boxplot(y=a_n_merged_harm_rest[mri_col], x=a_n_merged_harm_rest["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.suptitle('Harmonization of {} (disease)'.format(mri_col))


    plt.subplots(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.title('Before harmonization')
    sns.boxplot(y=a_n_merged_rest[amyloid_col], x=a_n_merged_rest["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.subplot(1,2,2)
    plt.title('After harmonization')
    sns.boxplot(y=a_n_merged_harm_rest[amyloid_col], x=a_n_merged_harm_rest["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.suptitle('Harmonization of {} (disease)'.format(amyloid_col))


    plt.subplots(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.title('Before harmonization')
    sns.boxplot(y=a_n_merged_rest[tau_col], x=a_n_merged_rest["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.subplot(1,2,2)
    plt.title('After harmonization')
    sns.boxplot(y=a_n_merged_harm_rest[tau_col], x=a_n_merged_harm_rest["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.suptitle('Harmonization of {} (disease)'.format(tau_col))



#plot_harmonization_effect(a_n_merged_CN, a_n_merged_harm_CN, a_n_merged_rest, a_n_merged_harm_rest, 'precuneus_left_Nvol', 'precuneus_left_Asuvr', 'precuneus_left_Tsuvr')
    
    
#-----------------------------------------------

#---------------------------------------------
### Plot p-values (MRI volume) before and after harmonization ########
#-------------------------------------------------

def plot_p_maps_MRI_harm(a_n_merged_harm_CN, a_n_merged_harm_rest, common_cols, MRI_vol_cols):
    
    post_harm_mri = pd.concat([a_n_merged_harm_CN, a_n_merged_harm_rest], axis = 0)[common_cols + MRI_vol_cols]

    post_harm_mri.columns = post_harm_mri.columns.str.replace(r'_Nvol', '')
    temp_mri_cols = [col for col in post_harm_mri.columns if col not in common_cols]

    from scipy.stats import f_oneway

    ADNI_Asuvr_CN = post_harm_mri.loc[(post_harm_mri.dataset == 'ADNI') & (post_harm_mri.stage == 'cdr = 0 amyloid negative')]
    ADNI_Asuvr_precl = post_harm_mri.loc[(post_harm_mri.dataset == 'ADNI') & (post_harm_mri.stage == 'cdr = 0 amyloid positive')]
    ADNI_Asuvr_cdr05 = post_harm_mri.loc[(post_harm_mri.dataset == 'ADNI') & (post_harm_mri.stage == 'cdr = 0.5')]
    ADNI_Asuvr_cdr1 = post_harm_mri.loc[(post_harm_mri.dataset == 'ADNI') & (post_harm_mri.stage == 'cdr >= 1')]


    KARI_Asuvr_CN = post_harm_mri.loc[(post_harm_mri.dataset == 'KARI') & (post_harm_mri.stage == 'cdr = 0 amyloid negative')]
    KARI_Asuvr_precl = post_harm_mri.loc[(post_harm_mri.dataset == 'KARI') & (post_harm_mri.stage == 'cdr = 0 amyloid positive')]
    KARI_Asuvr_cdr05 = post_harm_mri.loc[(post_harm_mri.dataset == 'KARI') & (post_harm_mri.stage == 'cdr = 0.5')]
    KARI_Asuvr_cdr1 = post_harm_mri.loc[(post_harm_mri.dataset == 'KARI') & (post_harm_mri.stage == 'cdr >= 1')]

    p_list_CN = []
    p_list_precl = []
    p_list_cdr05 = []
    p_list_cdr1 = []

    for mri_col in temp_mri_cols:
        F, p = f_oneway(ADNI_Asuvr_CN[mri_col], KARI_Asuvr_CN[mri_col])
        p_list_CN.append(p)

    for mri_col in temp_mri_cols:
        F, p = f_oneway(ADNI_Asuvr_precl[mri_col], KARI_Asuvr_precl[mri_col])
        p_list_precl.append(p)

    for mri_col in temp_mri_cols:
        F, p = f_oneway(ADNI_Asuvr_cdr05[mri_col], KARI_Asuvr_cdr05[mri_col])
        p_list_cdr05.append(p)

    for mri_col in temp_mri_cols:
        F, p = f_oneway(ADNI_Asuvr_cdr1[mri_col], KARI_Asuvr_cdr1[mri_col])
        p_list_cdr1.append(p)
    #return p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort

    plot_cortical_maps_MRI(p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1, temp_mri_cols)
    #plot_subcortical_maps_MRI(p_list_CN, p_list_rest, temp_amyloid_cols)
    
    return p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1
    

def plot_cortical_maps_MRI(p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1, temp_mri_cols):
    
    ggseg_python.plot_dk(dict(zip(temp_mri_cols, p_list_CN)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical volume) \n for healthy controls')
    print('cort_CN (cortical) : Max = {}, Min = {}'.format(max(p_list_CN), min(p_list_CN)))

    
#     ggseg_python.plot_dk(dict(zip(temp_mri_cols, p_list_precl)), cmap='hot',
#                           background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 24, vminmax = [0, 0.05],
#                           ylabel='p-value', title='ADNI vs KARI (cortical volume) \n for preclinical patients')
#     print('cort_precl (cortical) : Max = {}, Min = {}'.format(max(p_list_precl), min(p_list_precl)))


#     ggseg_python.plot_dk(dict(zip(temp_mri_cols, p_list_cdr05)), cmap='hot',
#                           background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 24, vminmax = [0, 0.05],
#                           ylabel='p-value', title='ADNI vs KARI (cortical volume) \n for CDR = 0.5 patients')
#     print('cort_cdr05 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr05), min(p_list_cdr05)))

#     ggseg_python.plot_dk(dict(zip(temp_mri_cols, p_list_precl)), cmap='hot',
#                           background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 24, vminmax = [0, 0.05],
#                           ylabel='p-value', title='ADNI vs KARI (cortical volume) \n for CDR >= 1 patients')
#     print('cort_cdr1 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr1), min(p_list_cdr1)))

    

def plot_subcortical_maps_MRI(p_list_CN, p_list_rest, temp_mri_cols):
    
    ggseg_python.plot_aseg(dict(zip(temp_mri_cols, p_list_CN)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical volume) \n for healthy controls')
    print('subcort_CN (subcortical) : Max = {}, Min = {}'.format(max(p_list_CN), min(p_list_CN)))

    
    ggseg_python.plot_aseg(dict(zip(temp_mri_cols, p_list_rest)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical volume) \n for disease patients')
    print('subcort_rest (subcortical) : Max = {}, Min = {}'.format(max(p_list_rest), min(p_list_rest)))
    
    
    
    
#-----------------------------------------------------------------
#-----------------------------------------------------------------

#-------------------------------------------------------    
### Plot p-values (amyloid SUVR) before and after harmonization ########
#--------------------------------------------------------

def plot_p_maps_amyloid_harm(a_n_merged_harm_CN, a_n_merged_harm_rest, common_cols, amyloid_SUVR_cols):
    
    post_harm_amyloid = pd.concat([a_n_merged_harm_CN, a_n_merged_harm_rest], axis = 0)[common_cols + amyloid_SUVR_cols]

    post_harm_amyloid.columns = post_harm_amyloid.columns.str.replace(r'_Asuvr', '')
    temp_amyloid_cols = [col for col in post_harm_amyloid.columns if col not in common_cols]

    from scipy.stats import f_oneway

    ADNI_Asuvr_CN = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'ADNI') & (post_harm_amyloid.stage == 'cdr = 0 amyloid negative')]
    ADNI_Asuvr_precl = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'ADNI') & (post_harm_amyloid.stage == 'cdr = 0 amyloid positive')]
    ADNI_Asuvr_cdr05 = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'ADNI') & (post_harm_amyloid.stage == 'cdr = 0.5')]
    ADNI_Asuvr_cdr1 = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'ADNI') & (post_harm_amyloid.stage == 'cdr >= 1')]


    KARI_Asuvr_CN = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'KARI') & (post_harm_amyloid.stage == 'cdr = 0 amyloid negative')]
    KARI_Asuvr_precl = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'KARI') & (post_harm_amyloid.stage == 'cdr = 0 amyloid positive')]
    KARI_Asuvr_cdr05 = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'KARI') & (post_harm_amyloid.stage == 'cdr = 0.5')]
    KARI_Asuvr_cdr1 = post_harm_amyloid.loc[(post_harm_amyloid.dataset == 'KARI') & (post_harm_amyloid.stage == 'cdr >= 1')]

    p_list_CN = []
    p_list_precl = []
    p_list_cdr05 = []
    p_list_cdr1 = []

    for amyloid_col in temp_amyloid_cols:
        F, p = f_oneway(ADNI_Asuvr_CN[amyloid_col], KARI_Asuvr_CN[amyloid_col])
        p_list_CN.append(p)

    for amyloid_col in temp_amyloid_cols:
        F, p = f_oneway(ADNI_Asuvr_precl[amyloid_col], KARI_Asuvr_precl[amyloid_col])
        p_list_precl.append(p)

    for amyloid_col in temp_amyloid_cols:
        F, p = f_oneway(ADNI_Asuvr_cdr05[amyloid_col], KARI_Asuvr_cdr05[amyloid_col])
        p_list_cdr05.append(p)

    for amyloid_col in temp_amyloid_cols:
        F, p = f_oneway(ADNI_Asuvr_cdr1[amyloid_col], KARI_Asuvr_cdr1[amyloid_col])
        p_list_cdr1.append(p)
    #return p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort

    plot_cortical_maps_amyloid(p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1, temp_amyloid_cols)
    #plot_subcortical_maps_amyloid(p_list_CN, p_list_rest, temp_amyloid_cols)
    
    return p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1
    

def plot_cortical_maps_amyloid(p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1, temp_amyloid_cols):
    
    ggseg_python.plot_dk(dict(zip(temp_amyloid_cols, p_list_CN)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical Amyloid SUVR) \n for healthy controls')
    print('cort_CN (cortical) : Max = {}, Min = {}'.format(max(p_list_CN), min(p_list_CN)))


    ggseg_python.plot_dk(dict(zip(temp_amyloid_cols, p_list_precl)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical Amyloid SUVR) \n for preclinical patients')
    print('cort_precl (cortical) : Max = {}, Min = {}'.format(max(p_list_precl), min(p_list_precl)))


    ggseg_python.plot_dk(dict(zip(temp_amyloid_cols, p_list_cdr05)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical Amyloid SUVR) \n for CDR = 0.5 patients')
    print('cort_cdr05 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr05), min(p_list_cdr05)))

    ggseg_python.plot_dk(dict(zip(temp_amyloid_cols, p_list_precl)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical Amyloid SUVR) \n for CDR >= 1 patients')
    print('cort_cdr1 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr1), min(p_list_cdr1)))

    

def plot_subcortical_maps_amyloid(p_list_CN, p_list_rest, temp_amyloid_cols):
    
    ggseg_python.plot_aseg(dict(zip(temp_amyloid_cols, p_list_CN)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical Amyloid SUVR) \n for healthy controls')
    print('subcort_CN (subcortical) : Max = {}, Min = {}'.format(max(p_list_CN), min(p_list_CN)))

    
    ggseg_python.plot_aseg(dict(zip(temp_amyloid_cols, p_list_rest)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical Amyloid SUVR) \n for disease patients')
    print('subcort_rest (subcortical) : Max = {}, Min = {}'.format(max(p_list_rest), min(p_list_rest)))



#---------------------------------------------------------------
#---------------------------------------------------------------

#-------------------------------------------------------    
### Plot p-values (tau SUVR) before and after harmonization ########
#--------------------------------------------------------

def plot_p_maps_tau_harm(a_n_merged_harm_CN, a_n_merged_harm_rest, common_cols, tau_SUVR_cols):
    
    #a_n_merged_harm_CN = a_n_merged_harm_CN.loc[a_n_merged_harm_CN.tau_present == 'yes']
    #a_n_merged_harm_rest = a_n_merged_harm_rest.loc[a_n_merged_harm_rest.tau_present == 'yes']
    
    post_harm_tau = pd.concat([a_n_merged_harm_CN, a_n_merged_harm_rest], axis = 0)[common_cols + tau_SUVR_cols]
    post_harm_tau = post_harm_tau.loc[post_harm_tau.tau_present == 'yes']
    
    post_harm_tau.columns = post_harm_tau.columns.str.replace(r'_Tsuvr', '')
    temp_tau_cols = [col for col in post_harm_tau.columns if col not in common_cols]

    from scipy.stats import f_oneway

    ADNI_Asuvr_CN = post_harm_tau.loc[(post_harm_tau.dataset == 'ADNI') & (post_harm_tau.stage == 'cdr = 0 amyloid negative')]
    ADNI_Asuvr_precl = post_harm_tau.loc[(post_harm_tau.dataset == 'ADNI') & (post_harm_tau.stage == 'cdr = 0 amyloid positive')]
    ADNI_Asuvr_cdr05 = post_harm_tau.loc[(post_harm_tau.dataset == 'ADNI') & (post_harm_tau.stage == 'cdr = 0.5')]
    ADNI_Asuvr_cdr1 = post_harm_tau.loc[(post_harm_tau.dataset == 'ADNI') & (post_harm_tau.stage == 'cdr >= 1')]


    KARI_Asuvr_CN = post_harm_tau.loc[(post_harm_tau.dataset == 'KARI') & (post_harm_tau.stage == 'cdr = 0 amyloid negative')]
    KARI_Asuvr_precl = post_harm_tau.loc[(post_harm_tau.dataset == 'KARI') & (post_harm_tau.stage == 'cdr = 0 amyloid positive')]
    KARI_Asuvr_cdr05 = post_harm_tau.loc[(post_harm_tau.dataset == 'KARI') & (post_harm_tau.stage == 'cdr = 0.5')]
    KARI_Asuvr_cdr1 = post_harm_tau.loc[(post_harm_tau.dataset == 'KARI') & (post_harm_tau.stage == 'cdr >= 1')]

    p_list_CN = []
    p_list_precl = []
    p_list_cdr05 = []
    p_list_cdr1 = []

    for tau_col in temp_tau_cols:
        F, p = f_oneway(ADNI_Asuvr_CN[tau_col], KARI_Asuvr_CN[tau_col])
        p_list_CN.append(p)

    for tau_col in temp_tau_cols:
        F, p = f_oneway(ADNI_Asuvr_precl[tau_col], KARI_Asuvr_precl[tau_col])
        p_list_precl.append(p)

    for tau_col in temp_tau_cols:
        F, p = f_oneway(ADNI_Asuvr_cdr05[tau_col], KARI_Asuvr_cdr05[tau_col])
        p_list_cdr05.append(p)

    for tau_col in temp_tau_cols:
        F, p = f_oneway(ADNI_Asuvr_cdr1[tau_col], KARI_Asuvr_cdr1[tau_col])
        p_list_cdr1.append(p)
    #return p_list_cdr0_cort, p_list_cdr0_5_cort, p_list_cdr1_cort, p_list_cdr0_subcort, p_list_cdr0_5_subcort, p_list_cdr1_subcort

    plot_cortical_maps_tau(p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1, temp_tau_cols)
    #plot_subcortical_maps_amyloid(p_list_CN, p_list_rest, temp_amyloid_cols)
    
    return p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1
    




def plot_cortical_maps_tau(p_list_CN, p_list_precl, p_list_cdr05, p_list_cdr1, temp_tau_cols):
    
    ggseg_python.plot_dk(dict(zip(temp_tau_cols, p_list_CN)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (cortical Tau SUVR) \n for healthy controls')
    print('cort_CN (cortical) : Max = {}, Min = {}'.format(max(p_list_CN), min(p_list_CN)))

    
    ggseg_python.plot_dk(dict(zip(temp_tau_cols, p_list_precl)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical Tau SUVR) \n for preclinical patients')
    print('cort_precl (cortical) : Max = {}, Min = {}'.format(max(p_list_precl), min(p_list_precl)))


    ggseg_python.plot_dk(dict(zip(temp_tau_cols, p_list_cdr05)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical Tau SUVR) \n for CDR = 0.5 patients')
    print('cort_cdr05 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr05), min(p_list_cdr05)))

    ggseg_python.plot_dk(dict(zip(temp_tau_cols, p_list_precl)), cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                          ylabel='p-value', title='ADNI vs KARI (cortical Tau SUVR) \n for CDR >= 1 patients')
    print('cort_cdr1 (cortical) : Max = {}, Min = {}'.format(max(p_list_cdr1), min(p_list_cdr1)))

    

def plot_subcortical_maps_tau(p_list_CN, p_list_rest, temp_tau_cols):
    
    ggseg_python.plot_aseg(dict(zip(temp_tau_cols, p_list_CN)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical tau SUVR) \n for healthy controls')
    print('subcort_CN (subcortical) : Max = {}, Min = {}'.format(max(p_list_CN), min(p_list_CN)))

    
    ggseg_python.plot_aseg(dict(zip(temp_tau_cols, p_list_rest)), cmap='hot',
                      background='w', edgecolor='k', bordercolor='gray', figsize = (8,5), fontsize = 24, vminmax = [0, 0.05],
                      ylabel='p-value', title='ADNI vs KARI (subcortical tau SUVR) \n for disease patients')
    print('subcort_rest (subcortical) : Max = {}, Min = {}'.format(max(p_list_rest), min(p_list_rest)))



#-------------------------------------------------------
#-------------------------------------------------------

