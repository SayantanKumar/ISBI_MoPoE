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

tadpole_challenge_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/tadpole_challenge'
roi_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis'

#########################################################################
#------------------- Relevant funtions ---------------------------------
#########################################################################

def merge_atn_dates_adni(roi_path, pet_path, missing = False):
    
    #roi_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis'
    
    ucsf_data = pd.read_csv(os.path.join(roi_path, 'UCSFFSX6_03_02_21.csv'))
    mri_date = ucsf_data[['RID', 'EXAMDATE']].rename(columns = {'EXAMDATE':'mri_date'})

    #pet_path = '/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/PET_Image_Analysis'
    temp_amyloid = pd.read_csv(os.path.join(pet_path, 'UCBERKELEYAV45_04_26_22.csv'))
    amyloid_date = temp_amyloid[['RID', 'EXAMDATE']].rename(columns = {'EXAMDATE':'amyloid_date'})

    temp_tau = pd.read_csv(os.path.join(pet_path, 'UCBERKELEYAV1451_04_26_22.csv'))
    tau_date = temp_tau[['RID', 'EXAMDATE']].rename(columns = {'EXAMDATE':'tau_date'})

    n_a, n_a_t = track_dates_atn_adni(mri_date, amyloid_date, tau_date)
    
    all_atn_adni = n_a_t.copy()
    
    if missing == False:
        all_atn_adni = n_a_t.loc[(n_a_t['tau_present'] == 'yes') & (n_a_t['amyloid_present'] == 'yes')]

    return all_atn_adni

#---------------------------------
#---------------------------------

def track_dates_atn_adni(mri_date, amyloid_date, tau_date):
    
    n_a = pd.merge(mri_date, amyloid_date, on = 'RID', how = 'left')
    n_a['n_a_diff'] = abs(pd.to_datetime(n_a['mri_date']) - pd.to_datetime(n_a['amyloid_date'])).dt.days

    n_a['amyloid_present'] = 'no'
    n_a.loc[(pd.notnull(n_a['amyloid_date']) == True), 'amyloid_present'] = 'yes'
    n_a.loc[n_a['n_a_diff'] >= 400, 'amyloid_present'] = 'no'
    n_a = n_a.sort_values(by = ['RID', 'mri_date', 'n_a_diff'])#.drop_duplicates()
    #n_a = n_a.drop_duplicates(subset = ['RID', 'mri_date'], keep = 'first').reset_index(drop = True)

    n_a_t = pd.merge(n_a, tau_date, on = 'RID', how = 'left')
    n_a_t['n_t_diff'] = abs(pd.to_datetime(n_a_t['mri_date']) - pd.to_datetime(n_a_t['tau_date'])).dt.days
    n_a_t['a_t_diff'] = abs(pd.to_datetime(n_a_t['amyloid_date']) - pd.to_datetime(n_a_t['tau_date'])).dt.days

    n_a_t['tau_present'] = 'no'
    n_a_t.loc[(n_a_t['n_t_diff'] <= 400) & (n_a_t['a_t_diff'] <= 400), 'tau_present'] = 'yes'
    n_a_t.loc[(n_a_t['n_t_diff'] <= 400) & (pd.isnull(n_a_t['amyloid_present']) == True), 'tau_present'] = 'yes'
    n_a_t.loc[(n_a_t['n_t_diff'] <= 400) & (n_a_t['amyloid_present'] == 'no'), 'tau_present'] = 'yes'

    n_a_t = n_a_t.sort_values(by = ['amyloid_present', 'tau_present'], ascending = False).drop_duplicates(subset = 'RID').reset_index(drop = True)

    return n_a, n_a_t

#---------------------------------
#---------------------------------

def merge_atn_dates_kari(mri, adni_kari_MRI_cort_dict, adni_kari_MRI_subcort_dict, kari_imaging_path, missing = False):
    
    #mri = pd.read_excel(os.path.join(kari_imaging_path, 'mri_t3.xlsx'))
    mri_date = mri[['ID', 'MR_Date']].rename(columns = {'MR_Date':'mri_date'})

    adni_kari_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/adni_kari_ROI_names_dict.csv'))
    adni_kari_subcortical_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/adni_kari_subcortical_SUVR_dict.csv'))

    av45 = pd.read_excel(os.path.join(kari_imaging_path, 'av45.xlsx'))
    amyloid_date = av45[['ID', 'PET_Date']].rename(columns = {'PET_Date':'amyloid_date'})

    tau = pd.read_excel(os.path.join(kari_imaging_path, 'tau.xlsx'))
    tau_date = tau[['ID', 'PET_Date']].rename(columns = {'PET_Date':'tau_date'})

    n_a_kari, n_a_t_kari = track_dates_atn_kari(mri_date, amyloid_date, tau_date)

    all_atn_kari = n_a_t_kari.copy()
    if missing == False:
        all_atn_kari = n_a_t_kari.loc[(n_a_t_kari['tau_present'] == 'yes') & (n_a_t_kari['amyloid_present'] == 'yes')]
    
    return all_atn_kari



#-----------------------------------------------

def track_dates_atn_kari(mri_date, amyloid_date, tau_date):
    
    n_a = pd.merge(mri_date, amyloid_date, on = 'ID', how = 'left')
    n_a['n_a_diff'] = abs(pd.to_datetime(n_a['mri_date']) - pd.to_datetime(n_a['amyloid_date'])).dt.days

    n_a['amyloid_present'] = 'no'
    n_a.loc[(pd.notnull(n_a['amyloid_date']) == True), 'amyloid_present'] = 'yes'
    n_a.loc[n_a['n_a_diff'] >= 365, 'amyloid_present'] = 'no'
    n_a = n_a.sort_values(by = ['ID', 'mri_date', 'n_a_diff'])#.drop_duplicates()
    #n_a = n_a.drop_duplicates(subset = ['RID', 'mri_date'], keep = 'first').reset_index(drop = True)

    n_a_t = pd.merge(n_a, tau_date, on = 'ID', how = 'left')
    n_a_t['n_t_diff'] = abs(pd.to_datetime(n_a_t['mri_date']) - pd.to_datetime(n_a_t['tau_date'])).dt.days
    n_a_t['a_t_diff'] = abs(pd.to_datetime(n_a_t['amyloid_date']) - pd.to_datetime(n_a_t['tau_date'])).dt.days

    n_a_t['tau_present'] = 'no'
    n_a_t.loc[(n_a_t['n_t_diff'] <= 400) & (n_a_t['a_t_diff'] <= 400), 'tau_present'] = 'yes'
    n_a_t.loc[(n_a_t['n_t_diff'] <= 400) & (pd.isnull(n_a_t['amyloid_present']) == True), 'tau_present'] = 'yes'
    n_a_t.loc[(n_a_t['n_t_diff'] <= 400) & (n_a_t['amyloid_present'] == 'no'), 'tau_present'] = 'yes'

    n_a_t = n_a_t.sort_values(by = ['amyloid_present', 'tau_present'], ascending = False).drop_duplicates(subset = 'ID').reset_index(drop = True)

    return n_a, n_a_t

#---------------------------------
#---------------------------------

def get_adni_mri_vol(adnimerge_bl, ucsf_data, all_atn_adni):
    
    freesurfer_cols = [col for col in ucsf_data.columns if 'ST' in col]
    all_cols =  ['RID', 'EXAMDATE'] + freesurfer_cols
    adni_atn = ucsf_data[all_cols].drop(columns = 'STATUS').merge(all_atn_adni[['RID', 'mri_date']], left_on = ['RID', 'EXAMDATE'], right_on = ['RID', 'mri_date'], how = 'right').drop_duplicates(subset = ['RID', 'EXAMDATE'], keep = 'first')

    temp_adni_atn = pd.merge(adnimerge_bl, adni_atn, on = ['RID'], how = 'right').drop(columns = ['ST8SV', 'EXAMDATE_x']).rename(columns = {'EXAMDATE_y':'EXAMDATE'})

    #-----------------------------------------

    input_data = temp_adni_atn.copy()

    other_cols = ['RID', 'EXAMDATE', 'DX_bl', 'AGE', 'PTGENDER', 'Intracranial_vol']
    #####-----------------------------------------------------------------------------------------
    cortical_cols = input_data.loc[:, input_data.columns.str.endswith('CV')].columns.to_list() # 69
    subcortical_cols = input_data.loc[:, input_data.columns.str.endswith('SV')].columns.to_list() # 49

    surface_area_cols = input_data.loc[:, input_data.columns.str.endswith('SA')].columns.to_list() # 70
    mean_cortical_thickness_cols = input_data.loc[:, input_data.columns.str.endswith('TA')].columns.to_list() # 68
    std_cortical_thickness_cols = input_data.loc[:, input_data.columns.str.endswith('TS')].columns.to_list() # 68

    #----------------------------------------------------------------------------

    ucsf_dict = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/MR_Image_Analysis/UCSFFSX6_DICT_08_27_19.csv')
    ucsf_dict = ucsf_dict.loc[ucsf_dict.FLDNAME.str.startswith('ST')][['FLDNAME', 'TEXT']].dropna().set_index('FLDNAME')

    ##-----------------Cortical-------------------------------------------

    a_list_cortical = list(ucsf_dict.loc[cortical_cols]["TEXT"].values) #column to list
    string_cortical = " ".join(a_list_cortical) # list of rows to string
    words_cortical = re.findall("(\w+)", string_cortical) # split to  single list of words

    cortical_roi = [item for item in words_cortical if words_cortical.count(item) == 1] #list of words that appear multiple times

    ##-----------------Subcortical-------------------------------------------

    a_list_subcortical = list(ucsf_dict.loc[subcortical_cols]["TEXT"].values) #column to list
    string_subcortical = " ".join(a_list_subcortical) # list of rows to string
    words_subcortical = re.findall("(\w+)", string_subcortical) # split to  single list of words

    subcortical_roi = [item for item in words_subcortical if words_subcortical.count(item) == 1] #list of words that appear multiple times

    ##-----------------------------------------------------------------

    cortical_rename_dict = {i:j for i,j in zip(cortical_cols,cortical_roi)}
    subcortical_rename_dict = {i:j for i,j in zip(subcortical_cols,subcortical_roi)}

    fs_cort = input_data[cortical_cols].rename(columns=cortical_rename_dict, inplace=False)
    fs_subcort = input_data[subcortical_cols].rename(columns=subcortical_rename_dict, inplace=False)

    subcort_remove_cols = ['ThirdVentricle', 'FourthVentricle', 'LeftCerebellumCortex', 'LeftCerebellumWM', 'LeftInferiorLateralVentricle', 'LeftLateralVentricle', 'RightCerebellumCortex', 'RightCerebellumWM', 'RightInferiorLateralVentricle', 'RightLateralVentricle', 'OpticChiasm', 'LeftVessel', 'RightVessel', 'NonWMHypoIntensities', 'LeftCorticalGM','RightCorticalGM', 'CorticalGM', 'LeftCorticalWM', 'RightCorticalWM', 'CorticalWM', 'SubcorticalGM', 'TotalGM', 'SupraTentorial', 'WMHypoIntensities']

    fs_subcort = fs_subcort.drop(columns = subcort_remove_cols)

    cort_remove_cols = ['LeftBankssts', 'Icv', 'RightBankssts']
    fs_cort = fs_cort.drop(columns = cort_remove_cols)

    ##------------------------------------------------------------------------------------
    cortical_cols = fs_cort.columns.to_list()
    subcortical_cols = fs_subcort.columns.to_list()

    input_data = pd.concat([input_data[other_cols], fs_cort, fs_subcort], axis = 1)

    ##-------------------------------------------------------------------------

    fs_cols = cortical_cols + subcortical_cols 

    fs_features_adni = input_data.copy()

    for col in fs_features_adni[cortical_cols + subcortical_cols].columns:
        fs_features_adni[col] = fs_features_adni[col]/fs_features_adni['Intracranial_vol']

    #fs_features_adni = fs_features_adni.dropna(subset = ['Intracranial_vol'])
    fs_features_adni = fs_features_adni[other_cols + fs_cols].copy()

    for i in fs_cols:
        fs_features_adni[i] = fs_features_adni[i].fillna(fs_features_adni[i].mean())

    N_adni_vol = fs_features_adni[other_cols + cortical_cols + subcortical_cols]

    cortical_cols_MRI, subcortical_cols_MRI, mri_adni_vol =  convert_cols_ggseg(N_adni_vol, cortical_cols, subcortical_cols)
    ADNI_MRI_vol = pd.concat([N_adni_vol[other_cols], mri_adni_vol], axis = 1)

    print('Number of vol cortical columns selected from ADNI T1 MRI = {}'.format(len(cortical_cols_MRI)))
    print('Number of vol subcortical columns selected from ADNI T1 MRI= {}\n'.format(len(subcortical_cols_MRI)))

    #ADNI_MRI_vol_cdr0, ADNI_MRI_vol_cdr0_5, ADNI_MRI_vol_cdr1 = plot_ADNI_MRI_vol_maps(ADNI_MRI_vol, cortical_cols_MRI, subcortical_cols_MRI)

    #ADNI_MRI_vol = ADNI_MRI_vol.loc[ADNI_MRI_vol.RID.isin(all_atn_adni.RID.unique())].drop(columns = 'Intracranial_vol')
    
    return ADNI_MRI_vol, cortical_cols_MRI, subcortical_cols_MRI



#--------------------------------------------

def add_cdr_mri_adni(cdr, ADNI_MRI_vol):
    
    cdr = cdr.sort_values(by = ['RID', 'USERDATE']).rename(columns = {'USERDATE':'CDR_Date'})[['RID', 'CDR_Date', 'CDGLOBAL']]
    cdr_adni = cdr.loc[cdr.RID.isin(ADNI_MRI_vol.RID.unique())].reset_index(drop = True)

    main_df = ADNI_MRI_vol.copy()[['RID', 'EXAMDATE']]

    date_diff = pd.merge(main_df, cdr_adni, on = 'RID', how = 'left')
    date_diff['mri_cdr_diff'] = abs(pd.to_datetime(date_diff['EXAMDATE']) - pd.to_datetime(date_diff['CDR_Date'])).dt.days

    date_diff = date_diff.sort_values(by = ['RID', 'EXAMDATE', 'mri_cdr_diff']).drop_duplicates(subset = ['RID'], keep = 'first').reset_index(drop = True).drop(columns = ['EXAMDATE', 'mri_cdr_diff'])

    ADNI_MRI = pd.merge(ADNI_MRI_vol, date_diff, on = 'RID', how = 'inner')

    #print('CDR distribution for ADNI MRI vol = \n{}'.format(ADNI_MRI['CDGLOBAL'].value_counts()))

    return ADNI_MRI


#--------------------------------------------
#--------------------------------------------

def get_kari_mri_vol(mri, all_atn_kari, adni_kari_MRI_cort_dict, adni_kari_MRI_subcort_dict, cortical_cols_MRI, subcortical_cols_MRI):
    
    mri_kari = mri.merge(all_atn_kari[['ID', 'mri_date']], left_on = ['ID', 'MR_Date'], right_on = ['ID', 'mri_date'], how = 'right').drop_duplicates(subset = ['ID', 'MR_Date'], keep = 'first')

    subcort_cols_kari = list(adni_kari_MRI_subcort_dict['KARI MRI'].str.strip().values)
    cort_cols_kari = list(adni_kari_MRI_cort_dict['KARI MRI'].str.strip().values)

    mri_temp = mri_kari.copy()

    mri_temp_subcort = mri_temp[subcort_cols_kari].rename(columns=dict(zip(adni_kari_MRI_subcort_dict['KARI MRI'].str.strip().values, adni_kari_MRI_subcort_dict['ggseg'].values))).dropna(how = 'all')
    mri_temp_subcort = mri_temp_subcort[subcortical_cols_MRI] ## reordering columns in KARI as in ADNI

    mri_temp_cort = mri_temp[cort_cols_kari].rename(columns=dict(zip(adni_kari_MRI_cort_dict['KARI MRI'].str.strip().values, adni_kari_MRI_cort_dict['ggseg MRI'].values))).dropna(how = 'all')
    mri_temp_cort = mri_temp_cort[cortical_cols_MRI] ## reordering columns in KARI as in ADNI

    #cdr = pd.read_excel(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/clinical_core/mod_b4_cdr.xlsx'))[['ID', 'TESTDATE', 'cdr']]
    #cdr_kari = cdr.sort_values(by = ['ID', 'TESTDATE'], ascending = True).drop_duplicates(subset = ['ID'], keep = 'first').reset_index(drop = True)

    KARI_MRI_vol = pd.concat([mri_temp[['ID', 'MR_Date']], mri_temp_cort, mri_temp_subcort], axis = 1).dropna(thresh = len(cort_cols_kari) + len(subcort_cols_kari)).reset_index(drop = True)

    #KARI_MRI_vol = cdr_kari.merge(kari_mri_cdr, on = 'ID', how = 'right')

    assert list(mri_temp_cort.columns.values) == cortical_cols_MRI
    assert list(mri_temp_subcort.columns.values) == subcortical_cols_MRI

    print('Number of vol cortical columns selected from KARI T1 MRI = {}'.format(len(mri_temp_cort.columns.values)))
    print('Number of vol subcortical columns selected from KARI T1 MRI= {}\n'.format(len(mri_temp_subcort.columns.values)))

    #print('CDR distribution for KARI MRI vol = \n{}'.format(KARI_MRI_vol.cdr.value_counts()))

    icv_mri_kari = mri_kari[['ID', 'MR_TOTV_INTRACRANIAL']].dropna().reset_index(drop = True)
    icv_mri_kari_vol = icv_mri_kari.merge(KARI_MRI_vol, on = 'ID', how = 'inner')

    for col in icv_mri_kari_vol[cortical_cols_MRI + subcortical_cols_MRI].columns.values:
        icv_mri_kari_vol[col] = icv_mri_kari_vol[col]/icv_mri_kari_vol['MR_TOTV_INTRACRANIAL']

    KARI_MRI_vol = icv_mri_kari_vol.drop(columns = 'MR_TOTV_INTRACRANIAL')

    return KARI_MRI_vol


#-----------------------------------

def add_cdr_mri_kari(cdr, KARI_MRI_vol):
    
    cdr_kari = cdr.loc[cdr.ID.isin(KARI_MRI_vol.ID.unique())].reset_index(drop = True)

    main_df = KARI_MRI_vol.copy()[['ID', 'MR_Date']]

    date_diff = pd.merge(main_df, cdr_kari, on = 'ID', how = 'left')
    date_diff['mri_cdr_diff'] = abs(pd.to_datetime(date_diff['MR_Date']) - pd.to_datetime(date_diff['TESTDATE'])).dt.days

    date_diff = date_diff.sort_values(by = ['ID', 'MR_Date', 'mri_cdr_diff']).drop_duplicates(subset = ['ID'], keep = 'first').reset_index(drop = True).drop(columns = ['MR_Date', 'mri_cdr_diff'])

    KARI_MRI = pd.merge(KARI_MRI_vol, date_diff, on = 'ID', how = 'inner')

    #print('CDR distribution for KARI MRI vol = \n{}'.format(KARI_MRI['cdr'].value_counts()))

    return KARI_MRI

#################################################################

def convert_cols_ggseg(table, cortical_cols, subcortical_cols):
    
    temp_mat = table.copy()

    cortical_cols_lh = [col for col in cortical_cols if 'Left' in col]
    cortical_cols_rh = [col for col in cortical_cols if 'Right' in col]

    cortical_cols_lh_new = list(temp_mat[cortical_cols_lh].columns.str.lower().str.replace('left',''))
    cortical_cols_rh_new = list(temp_mat[cortical_cols_rh].columns.str.lower().str.replace('right',''))

    cortical_cols_lh_new = ['{}_{}'.format(a1, b1) for b1 in ['left'] for a1 in cortical_cols_lh_new]
    cortical_cols_rh_new = ['{}_{}'.format(a2, b2) for b2 in ['right'] for a2 in cortical_cols_rh_new]

    temp_mat[cortical_cols_lh_new] = temp_mat[cortical_cols_lh].rename(columns=dict(zip(cortical_cols_lh_new, cortical_cols_lh)))
    temp_mat[cortical_cols_rh_new] = temp_mat[cortical_cols_rh].rename(columns=dict(zip(cortical_cols_rh_new, cortical_cols_rh)))

    #-----------------------------------------------
    

    subcort_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/ROI_Naming_Conversion/ADNI_KARI_MRI_subcort_dict.csv'))
    
    cols_remove = ['ThirdVentricle', 'FourthVentricle', 'LeftCerebellumCortex', 'LeftCerebellumWM', 'LeftInferiorLateralVentricle', 'LeftLateralVentricle', 'RightCerebellumCortex', 'RightCerebellumWM', 'RightInferiorLateralVentricle', 'RightLateralVentricle']
    subcort_dict = subcort_dict.loc[~subcort_dict['ADNI MRI'].isin(cols_remove)]
    
    subcort_dict['ADNI MRI'] = subcort_dict['ADNI MRI'].str.strip()

    temp_mat_subcort = temp_mat[subcort_dict['ADNI MRI'].values].rename(columns=dict(zip(subcort_dict['ADNI MRI'].values, subcort_dict['ggseg'].values)))
    
    subcortical_cols_MRI = list(temp_mat_subcort.columns.values)
    cortical_cols_MRI = cortical_cols_lh_new + cortical_cols_rh_new
    
    temp_mat = pd.concat([temp_mat[cortical_cols_MRI], temp_mat_subcort], axis = 1)

    return cortical_cols_MRI, subcortical_cols_MRI, temp_mat

#cortical_cols_lh_new, cortical_cols_rh_new, subcortical_cols_ggseg, temp_mat = convert_cols_ggseg(dev_bvae_c3_n1, cortical_cols, subcortical_cols)


#########################################################################
#----------------------------------------------------
#########################################################################

#--------------------------------------------------

def get_adni_amyloid_SUVR(temp_amyloid):
    
    summary_cols_amyloid = ['SUMMARYSUVR_WHOLECEREBNORM', 'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF', 'SUMMARYSUVR_COMPOSITE_REFNORM', 'SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF']

    amyloid_adni = temp_amyloid.sort_values(by = ['RID', 'EXAMDATE'])#.drop_duplicates(subset = 'RID', keep = 'first').reset_index(drop = True)
    amyloid_adni_suvr_cols = amyloid_adni.filter(like='SUVR').columns.values
    amyloid_adni_vol_cols = amyloid_adni.filter(like='VOLUME').columns.values

    amyloid_adni_suvr = amyloid_adni[['RID', 'EXAMDATE'] + list(amyloid_adni_suvr_cols)]

    cortical_cols_remove = ['CTX_LH_BANKSSTS_SUVR', 'CTX_RH_BANKSSTS_SUVR', 'CTX_RH_UNKNOWN_SUVR', 'CTX_LH_UNKNOWN_SUVR']
    subcortical_cols_remove = ['LEFT_VESSEL_SUVR', 'RIGHT_VESSEL_SUVR','OPTIC_CHIASM_SUVR', 'NON_WM_HYPOINTENSITIES_SUVR', 'LEFT_INF_LAT_VENT_SUVR', 'RIGHT_INF_LAT_VENT_SUVR', 'VENTRICLE_4TH_SUVR', 'VENTRICLE_5TH_SUVR', 'CEREBELLUMGREYMATTER_SUVR',
     'WHOLECEREBELLUM_SUVR', 'ERODED_SUBCORTICALWM_SUVR', 'COMPOSITE_REF_SUVR', 'FRONTAL_SUVR', 'CINGULATE_SUVR', 'PARIETAL_SUVR', 'TEMPORAL_SUVR', 'COMPOSITE_SUVR', 'VENTRICLE_3RD_SUVR', 'WM_HYPOINTENSITIES_SUVR', 'CSF_SUVR', 'LEFT_LATERAL_VENTRICLE_SUVR', 'RIGHT_LATERAL_VENTRICLE_SUVR',
                              'LEFT_CEREBELLUM_CORTEX_SUVR', 'LEFT_CEREBELLUM_WHITE_MATTER_SUVR', 'LEFT_CEREBRAL_WHITE_MATTER_SUVR', 'RIGHT_CEREBELLUM_CORTEX_SUVR','RIGHT_CEREBELLUM_WHITE_MATTER_SUVR', 'RIGHT_CEREBRAL_WHITE_MATTER_SUVR', 'INFERIORCEREBELLUM_SUVR','BRAAK1_SUVR', 'BRAAK34_SUVR', 'META_TEMPORAL_SUVR', 'BRAAK56_SUVR']

    cortical_cols_amyloid = [col for col in amyloid_adni_suvr.columns if 'CTX' in col and 'SUVR' in col and col not in cortical_cols_remove]
    subcortical_cols_amyloid = [col for col in amyloid_adni_suvr.columns if 'SUVR' in col and col not in cortical_cols_amyloid and col not in ['RID'] and col not in summary_cols_amyloid and col not in subcortical_cols_remove and col not in cortical_cols_remove]

    amyloid_adni_suvr = amyloid_adni_suvr[['RID', 'EXAMDATE'] + cortical_cols_amyloid + subcortical_cols_amyloid]

    combined_amyloid_subcort, subcort_roi_cols_plot = adni_brain_modality_subcortical_SUVR(amyloid_adni_suvr, subcortical_cols_amyloid)
    combined_amyloid_cort, cort_roi_cols_plot = adni_brain_modality_cortical_SUVR(amyloid_adni_suvr, cortical_cols_amyloid)

    print('Number of SUVR cortical columns selected from ADNI av45 = {}'.format(len(cort_roi_cols_plot)))
    print('Number of SUVR subcortical columns selected from ADNI av45= {}\n'.format(len(subcort_roi_cols_plot)))

    #plot_ADNI_Amyloid_SUVR_maps(combined_amyloid_subcort, combined_amyloid_subcort_cdr0, combined_amyloid_subcort_cdr0_5, combined_amyloid_subcort_cdr1, subcort_roi_cols_plot, combined_amyloid_cort, combined_amyloid_cort_cdr0, combined_amyloid_cort_cdr0_5, combined_amyloid_cort_cdr1, cort_roi_cols_plot)

    ADNI_amyloid_suvr = pd.merge(combined_amyloid_cort, combined_amyloid_subcort, on = ['RID', 'EXAMDATE'], how = 'inner')
    
    return ADNI_amyloid_suvr, cort_roi_cols_plot, subcort_roi_cols_plot


#---------------------------------------------------------------

##---------------------------------

def adni_brain_modality_subcortical_SUVR(amyloid_adni_suvr, subcortical_cols_amyloid):
    
    temp_amyloid = amyloid_adni_suvr.copy()

    subcortical_cols_lh = [col for col in subcortical_cols_amyloid if 'LEFT_' in col]
    subcortical_cols_rh = [col for col in subcortical_cols_amyloid if 'RIGHT_' in col]
    subcortical_cols_oth = [col for col in subcortical_cols_amyloid if 'LEFT_' not in col and 'RIGHT_' not in col]

    subcortical_cols_lh_new = list(temp_amyloid[subcortical_cols_lh].columns.str.lower().str.replace('_suvr','').str.replace('_','-').str.title())
    subcortical_cols_rh_new = list(temp_amyloid[subcortical_cols_rh].columns.str.lower().str.replace('_suvr','').str.replace('_','-').str.title())
    subcortical_cols_oth_new = list(temp_amyloid[subcortical_cols_oth].columns.str.lower().str.replace('_suvr','').str.title().str.replace('Cc','CC'))

    temp_amyloid[subcortical_cols_lh_new] = temp_amyloid[subcortical_cols_lh].rename(columns=dict(zip(subcortical_cols_lh_new, subcortical_cols_lh)))
    temp_amyloid[subcortical_cols_rh_new] = temp_amyloid[subcortical_cols_rh].rename(columns=dict(zip(subcortical_cols_rh_new, subcortical_cols_rh)))
    temp_amyloid[subcortical_cols_oth_new] = temp_amyloid[subcortical_cols_oth].rename(columns=dict(zip(subcortical_cols_oth_new, subcortical_cols_oth)))

    combined_amyloid_subcort = pd.concat([temp_amyloid[['RID', 'EXAMDATE']], temp_amyloid[subcortical_cols_lh_new], temp_amyloid[subcortical_cols_rh_new], temp_amyloid[subcortical_cols_oth_new]], axis = 1)

    subcort_roi_cols_plot = list(combined_amyloid_subcort.drop(columns = ['RID', 'EXAMDATE']).columns.values)

    return combined_amyloid_subcort, subcort_roi_cols_plot

##----------------------------------------------------

def adni_brain_modality_cortical_SUVR(amyloid_adni_suvr, cortical_cols_amyloid):
    
    temp_amyloid = amyloid_adni_suvr.copy()
    
    cortical_cols_lh = [col for col in cortical_cols_amyloid if 'CTX_LH_' in col and 'SUVR' in col]
    cortical_cols_rh = [col for col in cortical_cols_amyloid if 'CTX_RH_' in col and 'SUVR' in col]

    cortical_cols_lh_new = list(temp_amyloid[cortical_cols_lh].columns.str.lower().str.replace('suvr','left').str.replace('ctx_lh_',''))
    cortical_cols_rh_new = list(temp_amyloid[cortical_cols_rh].columns.str.lower().str.replace('suvr','right').str.replace('ctx_rh_',''))

    temp_amyloid[cortical_cols_lh_new] = temp_amyloid[cortical_cols_lh].rename(columns=dict(zip(cortical_cols_lh_new, cortical_cols_lh)))
    temp_amyloid[cortical_cols_rh_new] = temp_amyloid[cortical_cols_rh].rename(columns=dict(zip(cortical_cols_rh_new, cortical_cols_rh)))

    combined_amyloid_cort = pd.concat([temp_amyloid[['RID', 'EXAMDATE']], temp_amyloid[cortical_cols_lh_new], temp_amyloid[cortical_cols_rh_new]], axis = 1)

    cort_roi_cols_plot = list(combined_amyloid_cort.drop(columns = ['RID', 'EXAMDATE']).columns.values)

    return combined_amyloid_cort, cort_roi_cols_plot


#--------------------------------------------------

def kari_brain_amyloid_SUVR(av45, adni_kari_dict, adni_kari_subcortical_dict):
    
    kari_cort_lh_cols = [col for col in av45.columns.values if '_rsf_' not in col and 'fsuvr_l_ctx' in col]
    kari_cort_rh_cols = [col for col in av45.columns.values if '_rsf_' not in col and 'fsuvr_r_ctx' in col]

    kari_cort_lh = adni_kari_dict.loc[adni_kari_dict.KARI_name.str.contains('L_CTX')].reset_index(drop = True)
    kari_cort_rh = adni_kari_dict.loc[adni_kari_dict.KARI_name.str.contains('R_CTX')].reset_index(drop = True)

    kari_cort_lh['KARI_name'] = kari_cort_lh['KARI_name'].str.lower()
    kari_cort_lh['ROI_name'] = kari_cort_lh['ROI_name'].str.replace('ctx_lh_','') 
    kari_cort_lh['ROI_name'] = kari_cort_lh['ROI_name'] + '_left'
    kari_cort_lh['KARI_name'] = 'av45_fsuvr_' + kari_cort_lh['KARI_name'] 

    kari_cort_rh['KARI_name'] = kari_cort_rh['KARI_name'].str.lower()
    kari_cort_rh['ROI_name'] = kari_cort_rh['ROI_name'].str.replace('ctx_rh_','') 
    kari_cort_rh['ROI_name'] = kari_cort_rh['ROI_name'] + '_right'
    kari_cort_rh['KARI_name'] = 'av45_fsuvr_' + kari_cort_rh['KARI_name']

    av45_new_cort_lh = av45[kari_cort_lh_cols].rename(columns=dict(zip(kari_cort_lh['KARI_name'].values, kari_cort_lh['ROI_name'].values)))
    av45_new_cort_rh = av45[kari_cort_rh_cols].rename(columns=dict(zip(kari_cort_rh['KARI_name'].values, kari_cort_rh['ROI_name'].values))).dropna()

    av45_cort = pd.concat([av45_new_cort_lh, av45_new_cort_rh], axis = 1).dropna().reset_index(drop = True)
    av45_cort = av45_cort.drop(columns = ['bankssts_left', 'bankssts_right', 'Right_Cerebellum_Cortex_right', 'Left_Cerebellum_Cortex_left', 'corpuscallosum_left', 'corpuscallosum_right'])
    
    cortical_cols_kari = list(av45_cort.columns.values)
    
    #**********************************************************
    
    subcort_cols_remove = ['av45_fsuvr_l_substnca_ngra', 'av45_fsuvr_r_substnca_ngra']
    kari_subcort_lh_cols = [col for col in av45.columns.values if '_rsf_' not in col and 'ctx' not in col and 'wm' not in col and '_l_' in col and 'fsuvr' in col and col not in subcort_cols_remove]
    kari_subcort_rh_cols = [col for col in av45.columns.values if '_rsf_' not in col and 'ctx' not in col and 'wm' not in col and '_r_' in col and 'fsuvr' in col and col not in subcort_cols_remove]
    kari_subcort_oth_cols  =  ['av45_fsuvr_tot_brainstem', 'av45_fsuvr_crpclm_ant', 'av45_fsuvr_crpclm_cntrl', 'av45_fsuvr_crpclm_mid_ant', 'av45_fsuvr_crpclm_mid_post','av45_fsuvr_crpclm_post']

    #adni_kari_subcortical_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/adni_kari_subcortical_dict.csv'))

    av45_subcort = av45[kari_subcort_lh_cols + kari_subcort_rh_cols + kari_subcort_oth_cols].rename(columns=dict(zip(adni_kari_subcortical_dict['KARI Amyloid Name'].values, adni_kari_subcortical_dict['ADNI Name'].values))).dropna()
    
    subcortical_cols_kari = list(av45_subcort.columns.values)
    #subcortical_cols_kari = list(av45_kari[subcortical_cols_kari].columns.str.lstrip()
                                 
    #**********************************************************
    
    av45_kari = pd.concat([av45[['ID', 'PET_Date']],  av45_cort, av45_subcort], axis = 1).dropna().reset_index(drop = True)

#     cdr_amyloid_kari = cdr_kari.merge(av45_kari, on = 'ID', how = 'inner')
#     amyloid_kari_cdr0 = cdr_amyloid_kari.loc[cdr_amyloid_kari.cdr == 0]
#     amyloid_kari_cdr0_5 = cdr_amyloid_kari.loc[cdr_amyloid_kari.cdr == 0.5]
#     amyloid_kari_cdr1 = cdr_amyloid_kari.loc[cdr_amyloid_kari.cdr >= 1]

    return av45_kari, cortical_cols_kari, subcortical_cols_kari
    
    
#----------------------------------------------------------------------------

def get_adni_tau_SUVR(temp_tau):
    
    tau_adni_suvr_cols = temp_tau.filter(like='SUVR').columns.values

    summary_cols_tau = ['SUMMARYSUVR_WHOLECEREBNORM', 'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF', 'SUMMARYSUVR_COMPOSITE_REFNORM', 'SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF']

    tau_adni_suvr = temp_tau[['RID', 'EXAMDATE'] + list(tau_adni_suvr_cols)]

    cortical_cols_remove = ['CTX_LH_BANKSSTS_SUVR', 'CTX_RH_BANKSSTS_SUVR', 'CTX_RH_UNKNOWN_SUVR', 'CTX_LH_UNKNOWN_SUVR']
    subcortical_cols_remove = ['LEFT_VESSEL_SUVR', 'RIGHT_VESSEL_SUVR','OPTIC_CHIASM_SUVR', 'NON_WM_HYPOINTENSITIES_SUVR', 'LEFT_INF_LAT_VENT_SUVR', 'RIGHT_INF_LAT_VENT_SUVR', 'VENTRICLE_4TH_SUVR', 'VENTRICLE_5TH_SUVR', 'CEREBELLUMGREYMATTER_SUVR',
     'WHOLECEREBELLUM_SUVR', 'ERODED_SUBCORTICALWM_SUVR', 'COMPOSITE_REF_SUVR', 'FRONTAL_SUVR', 'CINGULATE_SUVR', 'PARIETAL_SUVR', 'TEMPORAL_SUVR', 'COMPOSITE_SUVR', 'VENTRICLE_3RD_SUVR', 'WM_HYPOINTENSITIES_SUVR', 'CSF_SUVR', 'LEFT_LATERAL_VENTRICLE_SUVR', 'RIGHT_LATERAL_VENTRICLE_SUVR',
                              'LEFT_CEREBELLUM_CORTEX_SUVR', 'LEFT_CEREBELLUM_WHITE_MATTER_SUVR', 'LEFT_CEREBRAL_WHITE_MATTER_SUVR', 'RIGHT_CEREBELLUM_CORTEX_SUVR','RIGHT_CEREBELLUM_WHITE_MATTER_SUVR', 'RIGHT_CEREBRAL_WHITE_MATTER_SUVR', 'INFERIORCEREBELLUM_SUVR','BRAAK1_SUVR', 'BRAAK34_SUVR', 'META_TEMPORAL_SUVR', 'BRAAK56_SUVR']

    cortical_cols_tau = [col for col in tau_adni_suvr.columns if 'CTX' in col and 'SUVR' in col and col not in cortical_cols_remove]
    subcortical_cols_tau = [col for col in tau_adni_suvr.columns if 'SUVR' in col and col not in cortical_cols_tau and col not in cortical_cols_remove and col not in ['RID'] and col not in summary_cols_tau and col not in subcortical_cols_remove]

    tau_adni_suvr = tau_adni_suvr[['RID', 'EXAMDATE'] + cortical_cols_tau + subcortical_cols_tau]

    combined_tau_subcort, subcort_roi_cols_plot = adni_brain_modality_subcortical_SUVR(tau_adni_suvr, subcortical_cols_tau)
    combined_tau_cort, cort_roi_cols_plot = adni_brain_modality_cortical_SUVR(tau_adni_suvr, cortical_cols_tau)

    ADNI_tau_suvr = pd.merge(combined_tau_cort, combined_tau_subcort, on = ['RID', 'EXAMDATE'], how = 'inner')
    
    print('Number of SUVR cortical columns selected from ADNI av45 = {}'.format(len(cort_roi_cols_plot)))
    print('Number of SUVR subcortical columns selected from ADNI av45= {}\n'.format(len(subcort_roi_cols_plot)))

    return ADNI_tau_suvr, cortical_cols_tau, subcortical_cols_tau

#----------------------------------------------------------

def kari_brain_tau_SUVR(av1451, adni_kari_dict, adni_kari_subcortical_dict):
    
    kari_cort_lh_cols = [col for col in av1451.columns.values if '_rsf_' not in col and 'fsuvr_l_ctx' in col]
    kari_cort_rh_cols = [col for col in av1451.columns.values if '_rsf_' not in col and 'fsuvr_r_ctx' in col]

    kari_cort_lh = adni_kari_dict.loc[adni_kari_dict.KARI_name.str.contains('L_CTX')].reset_index(drop = True)
    kari_cort_rh = adni_kari_dict.loc[adni_kari_dict.KARI_name.str.contains('R_CTX')].reset_index(drop = True)

    kari_cort_lh['KARI_name'] = kari_cort_lh['KARI_name'].str.lower()
    kari_cort_lh['ROI_name'] = kari_cort_lh['ROI_name'].str.replace('ctx_lh_','') 
    kari_cort_lh['ROI_name'] = kari_cort_lh['ROI_name'] + '_left'
    kari_cort_lh['KARI_name'] = 'av1451_fsuvr_' + kari_cort_lh['KARI_name'] 

    kari_cort_rh['KARI_name'] = kari_cort_rh['KARI_name'].str.lower()
    kari_cort_rh['ROI_name'] = kari_cort_rh['ROI_name'].str.replace('ctx_rh_','') 
    kari_cort_rh['ROI_name'] = kari_cort_rh['ROI_name'] + '_right'
    kari_cort_rh['KARI_name'] = 'av1451_fsuvr_' + kari_cort_rh['KARI_name']

    av1451_new_cort_lh = av1451[kari_cort_lh_cols].rename(columns=dict(zip(kari_cort_lh['KARI_name'].values, kari_cort_lh['ROI_name'].values))).dropna()
    av1451_new_cort_rh = av1451[kari_cort_rh_cols].rename(columns=dict(zip(kari_cort_rh['KARI_name'].values, kari_cort_rh['ROI_name'].values))).dropna()

    av1451_cort = pd.concat([av1451_new_cort_lh, av1451_new_cort_rh], axis = 1).dropna().reset_index(drop = True)
    av1451_cort = av1451_cort.drop(columns = ['bankssts_left', 'bankssts_right', 'Right_Cerebellum_Cortex_right', 'Left_Cerebellum_Cortex_left', 'corpuscallosum_left', 'corpuscallosum_right'])
    
    cortical_cols_kari = list(av1451_cort.columns.values)
    
    #**********************************************************
    
    subcort_cols_remove = ['av1451_fsuvr_l_substnca_ngra', 'av1451_fsuvr_r_substnca_ngra']
    kari_subcort_lh_cols = [col for col in av1451.columns.values if '_rsf_' not in col and 'ctx' not in col and 'wm' not in col and '_l_' in col and 'fsuvr' in col and col not in subcort_cols_remove]
    kari_subcort_rh_cols = [col for col in av1451.columns.values if '_rsf_' not in col and 'ctx' not in col and 'wm' not in col and '_r_' in col and 'fsuvr' in col and col not in subcort_cols_remove]
    kari_subcort_oth_cols  =  ['av1451_fsuvr_tot_brainstem', 'av1451_fsuvr_crpclm_ant', 'av1451_fsuvr_crpclm_cntrl', 'av1451_fsuvr_crpclm_mid_ant', 'av1451_fsuvr_crpclm_mid_post','av1451_fsuvr_crpclm_post']

    #adni_kari_subcortical_dict = pd.read_csv(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/adni_kari_subcortical_dict.csv'))

    av1451_subcort = av1451[kari_subcort_lh_cols + kari_subcort_rh_cols + kari_subcort_oth_cols].rename(columns=dict(zip(adni_kari_subcortical_dict['KARI Tau Name'].values, adni_kari_subcortical_dict['ADNI Name'].values))).dropna()
    
    subcortical_cols_kari = list(av1451_subcort.columns.values)
    #subcortical_cols_kari = list(av45_kari[subcortical_cols_kari].columns.str.lstrip()
                                 
    #**********************************************************
    
    av1451_kari = pd.concat([av1451[['ID', 'PET_Date']],  av1451_cort, av1451_subcort], axis = 1).dropna().reset_index(drop = True)

    return av1451_kari, cortical_cols_kari, subcortical_cols_kari
    