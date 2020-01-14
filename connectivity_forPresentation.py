#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:56:36 2020

@author: Or Duek
Running conenctivity analysis for the KPE study
TOC

1. KPE task based connectivity analysis
2. KPE RS connectivity analysis
3. Seed based
"""

#%% [Set vars]

work_dir = '/media/Data/work'
#%% [import stuff]
import os
os.chdir('/home/or/kpe_task_analysis')
import nilearn
import pandas as pd
import numpy as np
import scipy
from connUtils import timeSeriesSingle, removeVars, createCorMat, stratifyTimeseries, contFuncs

#%% [Subject list]
subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464', '1468', '1480', '1499']
subject_list3 =['008','1223','1263','1293','1307','1322','1339','1343','1356','1364','1369','1387','1390','1403','1464', '1499'] # subjects who has 3 scans '1351',

func_template = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confound_template = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-Memory_desc-confounds_regressors.tsv'
event_file_template = '/media/Data/PTSD_KPE/condition_files/sub-{sub}_ses-{session}.csv'


#%%[Task Based]
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation') # can choose partial - it might be better   

# # We want to run the analysis using two different atlases: 1. Shen and 2. AAL (for now)
# aal = nilearn.datasets.fetch_atlas_aal(data_dir = work_dir)

# # start with AAL

# # take each subject - create timeline - stratify to events - save. Move to next sub
# os.chdir('/home/or/kpe_conn/aal')

# def createTask_time(session, subject_list):
#     for sub in subject_list:
#         func_file = func_template.format(sub = sub,session = session) # use template and load specific files
#         confound_file = confound_template.format(sub = sub, session = session)
#         event_file = event_file_template.format(sub=sub, session=session)
#         timeline = timeSeriesSingle(func_file, confound_file, aal.maps)
#         stratifyTimeseries(event_file, timeline, sub, 1)


# createTask_time('2',subject_list) # create a task timeseries for all subjects

timeFile_template = '/home/or/kpe_conn/aal/session_{session}/subject_{sub}/traumaTrials.npy'
# create conn matrix for each trial type
session = '2'
ses_2_trauma_corr = []
ses_2_trauma_corr_z = []
for sub in subject_list:
    file = timeFile_template.format(session = session, sub=sub)
    cor = correlation_measure.fit_transform([np.load(file)])[0]
    cor_z = np.arctan(cor) # fisher-z transformation
    ses_2_trauma_corr.append(cor)
    ses_2_trauma_corr_z.append(cor_z)

#%% reshaping and analysis
# reshape array
cor_z_array_2 = np.array(ses_2_trauma_corr_z)
reshape_ses1_trauma = np.moveaxis(np.array(cor_z_array),0,-1)
reshape_Ses2_trauma = np.moveaxis(np.array(cor_z_array_2),0,-1)
#%% create a symmetric matrix for CPM
sym_mat = []
for i in range(len(ses_1_trauma_corr_z)):
    x = nilearn.connectome.sym_matrix_to_vec(ses_1_trauma_corr_z[i], discard_diagonal=False)
    x_mat = nilearn.connectome.vec_to_sym_matrix(x)
    sym_mat.append(x_mat)

sym_mat = np.array(sym_mat)
sym_mat = np.moveaxis(sym_mat,0,-1)
scipy.io.savemat('ses2_trauma.mat', dict(x=sym_mat))

#%% run NBS
from bct import nbs
# we compare ket1 and ket3
pval, adj, _ = nbs.nbs_bct(reshape_ses1_trauma, reshape_Ses2_trauma, thresh=2.5, tail='both',k=500, paired=True, verbose = True)


# there is a difference. Lets plot the network
# first create diff
diffMat_2_1 = contFuncs(ses_1_trauma_corr, ses_2_trauma_corr)

# reshape and save as .mat for CPM
diffMat_21_CPM = np.moveaxis(np.array(diffMat_2_1),0, -1)
scipy.io.savemat('diffMat_2_1.mat', dict(x=diffMat_21_CPM))
diffMat_2_1_thr = np.array(diffMat_2_1) * adj
diffMat_2_1_thr_avergae = np.mean(diffMat_2_1_thr, axis = 0)
# how many edges?
np.sum(adj)
# 1310 edges survived
np.savetxt('diffMat_2_1_thr.csv', diffMat_2_1_thr_avergae)

# create matrix for qgraphs in R
aal = nilearn.datasets.fetch_atlas_aal()
labels = aal.labels
np.savetxt('aal_labels.csv', np.array(labels))
#%% Create group specific arrays
