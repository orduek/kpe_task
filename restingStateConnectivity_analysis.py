#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:44:04 2019

@author: Or Duek
Resting state connectivity analysis of KPE data
"""

#%% set working variables
work_dir = '/media/Data/work/connectivity'

# %%load parcellation (Yeo? / Shen?)
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import plotting
import scipy
# yeo
#atlas_filename = '/home/or/Downloads/1000subjects_reference_Yeo/Yeo_JNeurophysiol11_SplitLabels/MNI152/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz'
#atlas_labes = pd.read_csv('/home/or/Downloads/1000subjects_reference_Yeo/Yeo_JNeurophysiol11_SplitLabels/Yeo2011_17networks_N1000.split_components.glossary.csv')
#coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)
# shen
atlas_filename = '/home/or/Downloads/shenPar/shen_1mm_268_parcellation.nii.gz'
atlas_labes = pd.read_csv('/home/or/Downloads/shenPar/shen_268_parcellation_networklabels.csv')
colors = pd.read_csv('/home/or/Downloads/shenPar/shen_268_parcellation_networklabels_colors.csv')
coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)

atlas_labes = np.array(atlas_labes)
atlas_labes.shape

#%% Methods for analysis
def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['csf', 'white_matter', 'framewise_displacement',
                          'a_comp_cor_00', 'a_comp_cor_01',	'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']] # can add 'global_signal' also
     # change NaN of FD to zero
    finalConf = np.array(finalConf)
    finalConf[0,2] = 0 # if removing FD than should remove this one also
    return finalConf

# build method for creating time series for subjects
def timeSeries(func_files, confound_files):
    total_subjects = [] # creating an empty array that will hold all subjects matrix 
    # This function needs a masker object that will be defined outside the function
    for func_file, confound_file in zip(func_files, confound_files):
        print(f"proccessing file {func_file}") # print file name
        confoundClean = removeVars(confound_file)
        confoundArray = confoundClean#confoundClean.values
        time_series = masker.fit_transform(func_file, confounds=confoundArray)
        #time_series = extractor.fit_transform(func_file, confounds=confoundArray)
        #masker.fit_transform(func_file, confoundArray)
        total_subjects.append(time_series)
    return total_subjects

# contrasting two timePoints
def contFuncs(time_series1, time_series2):
    twoMinusOneMat = []
    for scanMatrix, scanMatrix2 in zip(time_series1, time_series2):
        a = scanMatrix2 - scanMatrix
        twoMinusOneMat.append(a)
    return np.array(twoMinusOneMat)

# create correlation matrix per subject
def createCorMat(time_series):
    # create correlation matrix for each subject
    fullMatrix = []
    for time_s in time_series:
        correlation_matrix = correlation_measure.fit_transform([time_s])[0]
        fullMatrix.append(correlation_matrix)
    return fullMatrix


#%% Set masker object
# Here you set the specific methods for masking and correlation. Please see Nilearn website for more info.
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiLabelsMasker
# in this mask we standardize the values, so mean is 0 and between -1 to 1
# masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, smoothing_fwhm = 6,
#                          memory="/home/oad4/scratch60/shenPar_nilearn",high_pass=.01 , low_pass = .1, t_r=1, verbose=5)

# use different masker when using Yeo atlas. 
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, smoothing_fwhm = 6,
                        memory="/media/Data/nilearn",t_r=1, verbose=5, high_pass=.01 , low_pass = .1) # As it is task based we dont' bandpassing high_pass=.01 , low_pass = .1)
                           
from nilearn.connectome import ConnectivityMeasure
#correlation_measure = ConnectivityMeasure(kind='partial correlation') # can choose partial - it might be better
correlation_measure = ConnectivityMeasure(kind='correlation') # can choose partial - it might be better        

#%% load files (image, task events)
subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464', '1468', '1480', '1499']
subject_list_1 = ['008','1223','1253','1263','1293','1307','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464', '1468', '1480', '1499']
subject_list3 =['008','1223','1263','1293','1307','1322','1339','1343','1356','1364','1369','1387','1390','1403','1464', '1499'] # subjects who has 3 scans '1351',

def fileList(subjects, session):
    func_files = ['/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-%s/ses-%s/func/sub-%s_ses-%s_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' % (sub,session,sub,session) for sub in subjects]
    return func_files

def confList(subjects, session):
    confound_files = ['/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-%s/ses-%s/func/sub-%s_ses-%s_task-rest_desc-confounds_regressors.tsv' % (sub,session,sub,session) for sub in subjects]
    return confound_files

#%%
    
#%% create time series (with condounders)
session1 = timeSeries(func_files=fileList(subject_list_1,'1'), confound_files=confList(subject_list_1, '1'))
session2 = timeSeries(func_files=fileList(subject_list_1,'2'), confound_files=confList(subject_list_1, '2'))
session3 = timeSeries(func_files=fileList(subject_list3,'3'), confound_files=confList(subject_list3, '3'))

#%%
os.chdir('/home/or/kpe_conn/ShenParc')
np.save("session_1Timeseries_ShenRS",session1) # saving array
np.save("session_2TimeseriesShenRS",session2)
np.save("session_3TimeseriesShenRS", session3)


#%% Correlations
   
cor1 = createCorMat(session1)
cor2 = createCorMat(session2)
cor3 = createCorMat(session3)

#%% NBS
cor1Reshape = np.moveaxis(np.array(cor1),0,-1)
cor2Reshape = np.moveaxis(np.array(cor2),0,-1)
cor3Reshape = np.moveaxis(np.array(cor3),0,-1)
from bct import nbs
# we compare ket1 and ket3
pval, adj, _ = nbs.nbs_bct(cor1Reshape, cor2Reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
# no difference in RS across groups

# Compare first and 3rd
pval, adj, _ = nbs.nbs_bct(cor1Reshape, cor2Reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)