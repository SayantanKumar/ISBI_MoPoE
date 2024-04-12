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

tadpole_challenge_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/tadpole_challenge'
roi_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis'

#----------------------------------------------------------
#---------------- MRI volumes --------------------------
#----------------------------------------------------------


#######################################################
##---------------- ADNI -----------------------------
#######################################################

# 'UCSFFSL51Y1_08_01_16.csv' -- Longitudinal Freesurfer 5.1 (ADNI 2/GO 3T 2010-2016)
# 'UCSFFSX51FINAL_11_08_19.csv' -- Cross-sectional Freesurfer 5.1 (ADNI 2/GO 3T 2010-2016)
# 'UCSFFSL_02_01_16.csv' -- Longitudinal Freesurfer 4.4 (ADNI 1 maybe?)
# 'UCSFFSX_11_02_15' -- Cross-sectional Freesurfer 4.3 (ADNI 1 maybe)
# 'UCSFFSX6_03_02_21.csv' -- Cross-sectional Freesurfer 6 (ADNI 3 3T)

tadpole_challenge_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/tadpole_challenge'
roi_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis'
pet_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/PET_Image_Analysis'


all_atn_adni = merge_atn_dates_adni(roi_path, pet_path, missing = True)

adnimerge_cols = ['RID', 'EXAMDATE', 'DX_bl', 'AGE', 'PTGENDER', 'ICV_bl']
adnimerge = pd.read_csv(os.path.join(tadpole_challenge_path, 'ADNIMERGE.csv'), low_memory = False)[adnimerge_cols].rename(columns = {'ICV_bl':'Intracranial_vol'})
  
#adnimerge_bl = get_demo_adni(tadpole_challenge_path)
#temp_adni = get_roi_adni(roi_path, adnimerge_bl)

ucsf_data = pd.read_csv(os.path.join(roi_path, 'UCSFFSX6_03_02_21.csv'))

ADNI_MRI_vol, cortical_cols_MRI, subcortical_cols_MRI = get_adni_mri_vol(adnimerge, ucsf_data, all_atn_adni)
ADNI_MRI_vol = ADNI_MRI_vol.drop_duplicates().reset_index(drop = True)

cdr = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/CDR.csv')
ADNI_MRI = add_cdr_mri_adni(cdr, ADNI_MRI_vol)
ADNI_MRI = ADNI_MRI.rename(columns = {'EXAMDATE':'MR_Date'}).dropna(subset = ['Intracranial_vol'])

ADNI_MRI['AGE'] = ADNI_MRI['AGE'].fillna(ADNI_MRI['AGE'].mean())
ADNI_MRI['CDGLOBAL'] = ADNI_MRI['CDGLOBAL'].fillna(ADNI_MRI['CDGLOBAL'].mode()[0])
ADNI_MRI['PTGENDER'] = ADNI_MRI['PTGENDER'].fillna(ADNI_MRI['PTGENDER'].mode()[0])

print('CDR distribution for ADNI MRI vol = \n{}'.format(ADNI_MRI['CDGLOBAL'].value_counts()))
print('Number of patients having ADNI MRI vol = {}'.format(ADNI_MRI.RID.nunique()))


#######################################################
##---------------- KARI -----------------------------
#######################################################


adni_kari_MRI_cort_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/ADNI_KARI_MRI_cort_dict.csv'))
adni_kari_MRI_subcort_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/ADNI_KARI_MRI_subcort_dict.csv'))

kari_imaging_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/imaging'

mri = pd.read_excel(os.path.join(kari_imaging_path, 'mri_t3.xlsx'))

all_atn_kari= merge_atn_dates_kari(mri, adni_kari_MRI_cort_dict, adni_kari_MRI_subcort_dict, kari_imaging_path, missing = True)

kari_demo = pd.read_excel('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/clinical_core/mod_demographics.xlsx')[['ID', 'sex', 'BIRTH']]

kari_demo = kari_demo.loc[kari_demo.ID.isin(all_atn_kari.ID.unique())].reset_index(drop = True)

#----------------------------------------------


KARI_MRI_vol = get_kari_mri_vol(mri, all_atn_kari, adni_kari_MRI_cort_dict, adni_kari_MRI_subcort_dict, cortical_cols_MRI, subcortical_cols_MRI)
                 
demo_kari_temp = kari_demo.merge(KARI_MRI_vol, on = 'ID', how = 'right')
demo_kari_temp['Age'] = (abs(pd.to_datetime(demo_kari_temp['MR_Date']) - pd.to_datetime(demo_kari_temp['BIRTH'])).dt.days)/365

cdr = pd.read_excel(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/clinical_core/mod_b4_cdr.xlsx'))[['ID', 'TESTDATE', 'cdr']]
KARI_MRI = add_cdr_mri_kari(cdr, KARI_MRI_vol)

demo_kari_temp = kari_demo.merge(KARI_MRI_vol, on = 'ID', how = 'right')
demo_kari_temp['Age'] = (abs(pd.to_datetime(demo_kari_temp['MR_Date']) - pd.to_datetime(demo_kari_temp['BIRTH'])).dt.days)/365
demo_kari_temp = demo_kari_temp.drop(columns = 'BIRTH').rename(columns = {'TESTDATE':'CDR_Date'})

cdr = pd.read_excel(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/clinical_core/mod_b4_cdr.xlsx'))[['ID', 'TESTDATE', 'cdr']]
KARI_MRI = add_cdr_mri_kari(cdr, demo_kari_temp).rename(columns = {'TESTDATE':'CDR_Date'})
KARI_MRI['cdr'] = KARI_MRI['cdr'].fillna(KARI_MRI['cdr'].mode()[0])

print('CDR distribution for KARI MRI vol = \n{}'.format(KARI_MRI['cdr'].value_counts()))
print('Number of patients having KARI MRI vol = {}'.format(KARI_MRI.ID.nunique()))


##----------------Combine both dataframes ADNI and KARI ---


#plot_p_maps_MRI(ADNI_MRI, KARI_MRI, cortical_cols_MRI, subcortical_cols_MRI)


mri_vol_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'CDR_Date', 'cdr'] + cortical_cols_MRI + subcortical_cols_MRI

ADNI_MRI_final= ADNI_MRI.rename(columns = {'RID':'ID', 'AGE':'Age', 'PTGENDER':'Sex', 'CDGLOBAL':'cdr'}).drop(columns = ['DX_bl', 'Intracranial_vol']).reindex(columns=mri_vol_cols)
KARI_MRI_final = KARI_MRI.rename(columns = {'sex':'Sex'}).reindex(columns=mri_vol_cols)
ADNI_MRI_final['dataset'] = 'ADNI'
KARI_MRI_final['dataset'] = 'KARI'

MRI_both = pd.concat([ADNI_MRI_final, KARI_MRI_final])

MRI_both = MRI_both.rename(columns={c: c+'_Nvol' for c in MRI_both.columns if c not in ['ID', 'Age', 'Sex', 'MR_Date', 'CDR_Date', 'cdr', 'dataset']})



#----------------------------------------------------------
#---------------- Amyloid SUVR --------------------------
#----------------------------------------------------------

#######################################################
##---------------- ADNI -----------------------------
#######################################################

pet_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/PET_Image_Analysis'
adni_amyloid = pd.read_csv(os.path.join(pet_path, 'UCBERKELEYAV45_04_26_22.csv'))

ADNI_amyloid_suvr, cortical_cols_adni_amyloid, subcortical_cols_adni_amyloid = get_adni_amyloid_SUVR(adni_amyloid)

ADNI_amyloid_suvr = pd.merge(ADNI_amyloid_suvr, all_atn_adni[['RID', 'amyloid_date', 'amyloid_present']], left_on = ['RID', 'EXAMDATE'], right_on = ['RID', 'amyloid_date'], how = 'right').drop(columns = 'EXAMDATE').reset_index(drop = True)

cols_add = ['RID', 'MR_Date', 'DX_bl', 'AGE', 'PTGENDER', 'CDR_Date', 'CDGLOBAL']
ADNI_amyloid = pd.merge(ADNI_amyloid_suvr, ADNI_MRI[cols_add], on = 'RID', how = 'inner')
ADNI_amyloid['amyloid_present'] = 'yes'
ADNI_amyloid.loc[(pd.isnull(ADNI_amyloid['amyloid_date']) == True), 'amyloid_present'] = 'no'

print('CDR distribution for ADNI amyloid SUVR = \n{}'.format(ADNI_amyloid['CDGLOBAL'].value_counts()))
print('Number of patients having ADNI MRI vol = {}'.format(ADNI_MRI.RID.nunique()))
print('Number of patients having MRI vol and Amyloid SUVR = {}'.format(len(ADNI_amyloid.loc[ADNI_amyloid['amyloid_present'] == 'yes'])))

ADNI_amyloid = ADNI_amyloid.rename(columns = {'Left-Thalamus-Proper':'Left-Thalamus', 'Right-Thalamus-Proper':'Right-Thalamus', 'Left-Ventraldc':'Left-VentralDC', 'Right-Ventraldc':'Right-VentralDC', 'Brainstem':'Brain-Stem'})


#######################################################
##---------------- KARI -----------------------------
#######################################################

adni_kari_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/adni_kari_ROI_names_dict.csv'))
adni_kari_subcortical_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/adni_kari_subcortical_SUVR_dict.csv'))

kari_imaging_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/imaging'

av45 = pd.read_excel(os.path.join(kari_imaging_path, 'av45.xlsx'))
av45 = av45.sort_values(by = ['ID', 'PET_Date'])#.drop_duplicates(subset = 'ID', keep = 'first').reset_index(drop = True)

av45_kari, cortical_cols_kari_amyloid, subcortical_cols_kari_amyloid = kari_brain_amyloid_SUVR(av45, adni_kari_dict, adni_kari_subcortical_dict)

# kari_amyloid_suvr = pd.merge(av45_kari, all_atn_kari[['ID', 'amyloid_date']], left_on = ['ID', 'PET_Date'], right_on = ['ID', 'amyloid_date'], how = 'left')#.dropna(subset = ['amyloid_date']).drop(columns = 'PET_Date').reset_index(drop = True)

KARI_amyloid_suvr = pd.merge(av45_kari, all_atn_kari[['ID', 'amyloid_date', 'amyloid_present']], left_on = ['ID', 'PET_Date'], right_on = ['ID', 'amyloid_date'], how = 'right').drop(columns = 'PET_Date').reset_index(drop = True)

cols_add = ['ID', 'MR_Date', 'sex', 'Age', 'CDR_Date', 'cdr']
KARI_amyloid = pd.merge(KARI_amyloid_suvr, KARI_MRI[cols_add], on = 'ID', how = 'inner')
# KARI_amyloid['amyloid_present'] = 'yes'
# KARI_amyloid.loc[(pd.isnull(KARI_amyloid['amyloid_date']) == True), 'amyloid_present'] = 'no'
KARI_amyloid.loc[(pd.isnull(KARI_amyloid['Brain-Stem']) == True), 'amyloid_present'] = 'no'

print('Number of vol cortical columns selected from KARI Amyloid = {}'.format(len(cortical_cols_kari_amyloid)))
print('Number of vol subcortical columns selected from KARI Amyloid = {}\n'.format(len(subcortical_cols_kari_amyloid)))

print('CDR distribution for KARI amyloid SUVR = \n{}'.format(KARI_amyloid['cdr'].value_counts()))
print('Number of patients having KARI MRI vol = {}'.format(KARI_MRI.ID.nunique()))
print('Number of patients having MRI vol and Amyloid SUVR = {}'.format(len(KARI_amyloid.loc[KARI_amyloid['amyloid_present'] == 'yes'])))

# for col in cortical_cols_kari_amyloid, subcortical_cols_kari_amyloid:
#     KARI_amyloid[col] = KARI_amyloid[col].fillna(KARI_amyloid[col].mean())

#######################################################
##---------------- P-value maps between ADNI and KARI ---
#######################################################

#plot_p_maps_amyloid(ADNI_amyloid, KARI_amyloid, cortical_cols_MRI, subcortical_cols_MRI)


##------------ Combine both dataframes -----------------
amyloid_SUVR_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'amyloid_present', 'amyloid_date',  'CDR_Date', 'cdr'] + cortical_cols_MRI + subcortical_cols_MRI

ADNI_amyloid_final= ADNI_amyloid.rename(columns = {'RID':'ID', 'AGE':'Age', 'PTGENDER':'Sex', 'CDGLOBAL':'cdr'}).drop(columns = ['DX_bl']).reindex(columns=amyloid_SUVR_cols)
KARI_amyloid_final = KARI_amyloid.rename(columns = {'sex':'Sex'}).reindex(columns=amyloid_SUVR_cols)
ADNI_amyloid_final['dataset'] = 'ADNI'
KARI_amyloid_final['dataset'] = 'KARI'

amyloid_both = pd.concat([ADNI_amyloid_final, KARI_amyloid_final])

amyloid_both = amyloid_both.rename(columns={c: c+'_Asuvr' for c in amyloid_both.columns if c not in ['ID', 'Age', 'Sex', 'MR_Date', 'amyloid_present', 'amyloid_date',  'CDR_Date', 'cdr', 'dataset']})


#----------------------------------------------------------
#---------------- Tau SUVR --------------------------
#----------------------------------------------------------

#######################################################
##---------------- ADNI -----------------------------
#######################################################

pet_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/PET_Image_Analysis'
temp_tau = pd.read_csv(os.path.join(pet_path, 'UCBERKELEYAV1451_04_26_22.csv')).sort_values(by = ['RID', 'EXAMDATE']).reset_index(drop = True)

ADNI_tau_suvr, cortical_cols_tau, subcortical_cols_tau = get_adni_tau_SUVR(temp_tau)

ADNI_tau_suvr = ADNI_tau_suvr.rename(columns = {'Left-Thalamus-Proper':'Left-Thalamus', 'Right-Thalamus-Proper':'Right-Thalamus', 'Left-Ventraldc':'Left-VentralDC', 'Right-Ventraldc':'Right-VentralDC', 'Brainstem':'Brain-Stem'})

ADNI_tau_suvr = pd.merge(ADNI_tau_suvr, all_atn_adni[['RID', 'tau_date', 'tau_present']], left_on = ['RID', 'EXAMDATE'], right_on = ['RID', 'tau_date'], how = 'right').drop(columns = 'EXAMDATE').reset_index(drop = True)

cols_add = ['RID', 'MR_Date', 'DX_bl', 'AGE', 'PTGENDER', 'CDR_Date', 'CDGLOBAL']
ADNI_tau = pd.merge(ADNI_tau_suvr, ADNI_MRI[cols_add], on = 'RID', how = 'inner')
ADNI_tau['tau_present'] = 'yes'
ADNI_tau.loc[(pd.isnull(ADNI_tau['tau_date']) == True), 'tau_present'] = 'no'

print('CDR distribution for ADNI tau SUVR = \n{}'.format(ADNI_tau['CDGLOBAL'].value_counts()))
print('Number of patients having ADNI MRI vol = {}'.format(ADNI_MRI.RID.nunique()))
print('Number of patients having MRI vol and Tau SUVR = {}'.format(len(ADNI_tau.loc[ADNI_tau['tau_present'] == 'yes'])))


#######################################################
##---------------- KARI -----------------------------
#######################################################

adni_kari_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/adni_kari_ROI_names_dict.csv'))
adni_kari_subcortical_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/adni_kari_subcortical_SUVR_dict.csv'))

kari_imaging_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/imaging'

tau = pd.read_excel(os.path.join(kari_imaging_path, 'tau.xlsx')).sort_values(by = ['ID', 'PET_Date']).reset_index(drop = True)

av1451_kari, cortical_cols_kari_tau, subcortical_cols_kari_tau = kari_brain_tau_SUVR(tau, adni_kari_dict, adni_kari_subcortical_dict)

KARI_tau_suvr = pd.merge(av1451_kari, all_atn_kari[['ID', 'tau_date', 'tau_present']], left_on = ['ID', 'PET_Date'], right_on = ['ID', 'tau_date'], how = 'right').drop(columns = 'PET_Date').reset_index(drop = True)

cols_add = ['ID', 'MR_Date', 'sex', 'Age', 'CDR_Date', 'cdr']
KARI_tau = pd.merge(KARI_tau_suvr, KARI_MRI[cols_add], on = 'ID', how = 'inner')
# KARI_amyloid['tau_present'] = 1
# KARI_tau.loc[(pd.isnull(KARI_tau['tau_date']) == True), 'tau_present'] = 0
KARI_tau.loc[(pd.isnull(KARI_tau['Brain-Stem']) == True), 'tau_present'] = 'no'

print('Number of vol cortical columns selected from KARI Tau = {}'.format(len(cortical_cols_kari_tau)))
print('Number of vol subcortical columns selected from KARI Tau = {}\n'.format(len(subcortical_cols_kari_tau)))

print('CDR distribution for KARI Tau SUVR = \n{}'.format(KARI_tau['cdr'].value_counts()))
print('Number of patients having KARI MRI vol = {}'.format(KARI_MRI.ID.nunique()))
print('Number of patients having MRI vol and Tau SUVR = {}'.format(len(KARI_tau.loc[KARI_tau['tau_present'] == 'yes'])))

# for col in cortical_cols_kari_tau, subcortical_cols_kari_tau:
#     KARI_tau[col] = KARI_amyloid[col].fillna(KARI_tau[col].mean())


#######################################################
##---------------- P-value maps between ADNI and KARI ---
#######################################################

#plot_p_maps_tau(ADNI_tau, KARI_tau, cortical_cols_MRI, subcortical_cols_MRI)

##------------ Combine both dataframes -----------------
tau_SUVR_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'tau_present', 'tau_date',  'CDR_Date', 'cdr'] + cortical_cols_MRI + subcortical_cols_MRI

ADNI_tau_final= ADNI_tau.rename(columns = {'RID':'ID', 'AGE':'Age', 'PTGENDER':'Sex', 'CDGLOBAL':'cdr'}).drop(columns = ['DX_bl']).reindex(columns=tau_SUVR_cols)
KARI_tau_final = KARI_tau.rename(columns = {'sex':'Sex'}).reindex(columns=tau_SUVR_cols)
ADNI_tau_final['dataset'] = 'ADNI'
KARI_tau_final['dataset'] = 'KARI'

tau_both = pd.concat([ADNI_tau_final, KARI_tau_final])

tau_both = tau_both.rename(columns={c: c+'_Tsuvr' for c in tau_both.columns if c not in ['ID', 'Age', 'Sex', 'MR_Date', 'tau_present', 'tau_date',  'CDR_Date', 'cdr', 'dataset']})



#----------------------------------------------------------------------------------------------
#---------------- Merge all 3 mopdalities. Select controls and cases --------------------------
#----------------------------------------------------------------------------------------------

mri_vol_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'CDR_Date', 'cdr'] + cortical_cols_MRI + subcortical_cols_MRI
amyloid_SUVR_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'amyloid_present', 'amyloid_date',  'CDR_Date', 'cdr'] + cortical_cols_MRI + subcortical_cols_MRI
tau_SUVR_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'tau_present', 'tau_date',  'CDR_Date', 'cdr'] + cortical_cols_MRI + subcortical_cols_MRI

MRI_both = merge_MRI_both(ADNI_MRI, KARI_MRI, mri_vol_cols)
amyloid_both = merge_amyloid_both(ADNI_amyloid, KARI_amyloid, amyloid_SUVR_cols)
tau_both = merge_tau_both(ADNI_tau, KARI_tau, tau_SUVR_cols)

a_n_merged, df_merged, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols = merge_allmod(MRI_both, amyloid_both, tau_both)
print(a_n_merged.cdr.value_counts())


##--------------Select Healthy controls (CDR = 0 and Amyloid Negative) ------

apos_kari = kari_amyloid_pos(av45, a_n_merged)
apos_adni = adni_amyloid_pos(adni_amyloid, a_n_merged)

reqd_cols = ['ID', 'amyloid_centiloid', 'amyloid_positive', 'cdr', 'stage', 'dataset']
apos_both = pd.concat([apos_adni[reqd_cols], apos_kari[reqd_cols]])

a_n_merged = a_n_merged.merge(apos_both, on = ['ID', 'cdr', 'dataset'], how = 'inner')
print(a_n_merged.stage.value_counts())

a_n_merged.loc[((a_n_merged.Sex == 'Male') | (a_n_merged.Sex == 'M')), 'Sex'] = 1
a_n_merged.loc[((a_n_merged.Sex == 'Female') | (a_n_merged.Sex == 'F')), 'Sex'] = 0

print('ADNI disease status distribution')
print(a_n_merged.loc[a_n_merged.dataset == 'ADNI'].stage.value_counts())

print('KARI disease status distribution')
print(a_n_merged.loc[a_n_merged.dataset == 'KARI'].stage.value_counts())


################################################################
#------------ Data harmonization (COMBAT) ---------------------
###############################################################

############################################
#--Option 1 : Learn COMBAT model on train data (healthy controls).  
#Apply pre-trained model on remaining disease patients
############################################

print('Option 1 : Learn COMBAT model on train data (healthy controls). Apply pre-trained model on remaining disease patients')

mri_harm_CN, amyloid_harm_CN, tau_harm_CN, mri_harm_rest, amyloid_harm_rest, tau_harm_rest, covars_CN, covars_rest = harmonization_step_stratify(a_n_merged, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)
common_cols = ['ID', 'Age', 'Sex', 'MR_Date', 'CDR_Date', 'tau_present', 'amyloid_positive', 'amyloid_centiloid', 'cdr', 'stage', 'dataset']

##---------CN -----------
a_n_merged_CN = a_n_merged.loc[a_n_merged['stage'] == 'cdr = 0 amyloid negative'].reset_index(drop = True)

a_n_merged_harm_CN = a_n_merged_CN.copy()
a_n_merged_harm_CN[MRI_vol_cols] = mri_harm_CN
a_n_merged_harm_CN[amyloid_SUVR_cols] = amyloid_harm_CN
#a_n_merged_harm_CN[tau_SUVR_cols] = tau_harm_CN

tau_present_id_CN = a_n_merged_CN.loc[a_n_merged_CN.tau_present == 'yes'].ID.unique()
a_n_merged_tau_harm_CN = a_n_merged_harm_CN.loc[a_n_merged_harm_CN.ID.isin(tau_present_id_CN)].reset_index(drop = True)
a_n_merged_tau_harm_CN[tau_SUVR_cols] = tau_harm_CN

for i1 in range(len(a_n_merged_harm_CN)):
    for i2 in range(len(a_n_merged_tau_harm_CN)):
        if a_n_merged_harm_CN.loc[i1, 'ID'] == a_n_merged_tau_harm_CN.loc[i2, 'ID']:
            a_n_merged_harm_CN.loc[i1, tau_SUVR_cols] = a_n_merged_tau_harm_CN.loc[i2, tau_SUVR_cols]

a_n_merged_harm_CN =  a_n_merged_harm_CN[common_cols + MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols]

##---------disease -----------
a_n_merged_rest = a_n_merged.loc[a_n_merged['stage'] != 'cdr = 0 amyloid negative'].reset_index(drop = True)

a_n_merged_harm_rest = a_n_merged_rest.copy()
a_n_merged_harm_rest[MRI_vol_cols] = mri_harm_rest
a_n_merged_harm_rest[amyloid_SUVR_cols] = amyloid_harm_rest
#a_n_merged_harm_rest[tau_SUVR_cols] = tau_harm_rest

tau_present_id_rest = a_n_merged_rest.loc[a_n_merged_rest.tau_present == 'yes'].ID.unique()
a_n_merged_tau_harm_rest = a_n_merged_harm_rest.loc[a_n_merged_harm_rest.ID.isin(tau_present_id_rest)].reset_index(drop = True)
a_n_merged_tau_harm_rest[tau_SUVR_cols] = tau_harm_rest

for i1 in range(len(a_n_merged_harm_rest)):
    for i2 in range(len(a_n_merged_tau_harm_rest)):
        if a_n_merged_harm_rest.loc[i1, 'ID'] == a_n_merged_tau_harm_rest.loc[i2, 'ID']:
            a_n_merged_harm_rest.loc[i1, tau_SUVR_cols] = a_n_merged_tau_harm_rest.loc[i2, tau_SUVR_cols]

a_n_merged_harm_rest =  a_n_merged_harm_rest[common_cols + MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols]

plot_harmonization_effect(a_n_merged_CN, a_n_merged_harm_CN, a_n_merged_rest, a_n_merged_harm_rest, 'precuneus_left_Nvol', 'precuneus_left_Asuvr', 'precuneus_left_Tsuvr')

a_n_merged_harm = pd.concat([a_n_merged_harm_CN, a_n_merged_harm_rest]).reset_index(drop = True)

# plt.subplots(figsize = (16,4))
# plt.subplot(1,2,1)
# plt.title('Before harmonization')
# sns.boxplot(y=a_n_merged['precuneus_left_Nvol'], x=a_n_merged["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.subplot(1,2,2)
# plt.title('After harmonization')
# sns.boxplot(y=a_n_merged_harm['precuneus_left_Nvol'], x=a_n_merged_harm["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.suptitle('Harmonization of {} (healthy controls + diseae)'.format('precuneus_left_Nvol'))


# plt.subplots(figsize = (16,4))
# plt.subplot(1,2,1)
# plt.title('Before harmonization')
# sns.boxplot(y=a_n_merged['precuneus_left_Asuvr'], x=a_n_merged["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.subplot(1,2,2)
# plt.title('After harmonization')
# sns.boxplot(y=a_n_merged_harm['precuneus_left_Asuvr'], x=a_n_merged_harm["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.suptitle('Harmonization of {} (healthy controls + diseae)'.format('precuneus_left_Asuvr'))

# plt.subplots(figsize = (16,4))
# plt.subplot(1,2,1)
# plt.title('Before harmonization')
# sns.boxplot(y=a_n_merged['precuneus_left_Tsuvr'], x=a_n_merged["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.subplot(1,2,2)
# plt.title('After harmonization')
# sns.boxplot(y=a_n_merged_harm['precuneus_left_Tsuvr'], x=a_n_merged_harm["dataset"], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.suptitle('Harmonization of {} (healthy controls + diseae)'.format('precuneus_left_Tsuvr'))

#------------------------------------------

print('MRI SUVR p-values before harmonization')
p_mri_CN, p_mri_precl, p_mri_cdr05, p_mri_cdr1 = plot_p_maps_MRI_harm(a_n_merged_CN, a_n_merged_rest, common_cols, MRI_vol_cols)

print('MRI SUVR p-values after harmonization')
p_mri_harm_CN, p_mri_harm_precl, p_mri_harm_cdr05, p_mri_harm_cdr1 = plot_p_maps_MRI_harm(a_n_merged_harm_CN, a_n_merged_harm_rest, common_cols, MRI_vol_cols)

#--------------------------------------------

print('Amyloid SUVR p-values before harmonization')
p_amyloid_CN, p_amyloid_precl, p_amyloid_cdr05, p_amyloid_cdr1 = plot_p_maps_amyloid_harm(a_n_merged_CN, a_n_merged_rest, common_cols, amyloid_SUVR_cols)

print('Amyloid SUVR p-values after harmonization')
p_amyloid_harm_CN, p_amyloid_harm_precl, p_amyloid_harm_cdr05, p_amyloid_harm_cdr1 = plot_p_maps_amyloid_harm(a_n_merged_harm_CN, a_n_merged_harm_rest, common_cols, amyloid_SUVR_cols)

#--------------------------

print('Tau SUVR p-values before harmonization')
p_tau_CN, p_tau_precl, p_tau_cdr05, p_tau_cdr1  = plot_p_maps_tau_harm(a_n_merged_CN, a_n_merged_rest, common_cols, tau_SUVR_cols)

print('Tau SUVR p-values after harmonization')
p_tau_harm_CN, p_tau_harm_precl, p_tau_harm_cdr05, p_tau_harm_cdr1 = plot_p_maps_tau_harm(a_n_merged_harm_CN, a_n_merged_harm_rest, common_cols, tau_SUVR_cols)


#######################################################
#---------------- Saving dataframes -------------------
#######################################################

a_n_merged.to_csv('./saved_dataframes/a_n_merged.csv')
a_n_merged_harm.to_csv('./saved_dataframes/a_n_merged_harm.csv')

with open("./saved_dataframes/MRI_vol_cols", "wb") as fp:
    pickle.dump(MRI_vol_cols, fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "wb") as fp:
    pickle.dump(amyloid_SUVR_cols, fp)
    
with open("./saved_dataframes/tau_SUVR_cols", "wb") as fp:
    pickle.dump(tau_SUVR_cols, fp)

MRI_both.to_csv('./saved_dataframes/MRI_both.csv')
amyloid_both.to_csv('./saved_dataframes/amyloid_both.csv')
tau_both.to_csv('./saved_dataframes/tau_both.csv')

all_atn_adni.to_csv('./saved_dataframes/all_atn_adni.csv')
all_atn_kari.to_csv('./saved_dataframes/all_atn_kari.csv')