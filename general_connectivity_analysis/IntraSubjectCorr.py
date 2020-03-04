#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:30:11 2020

@author: Or DUek
Create Intra-Subject Correlation focusing on the amygdala. 
Basic method:
    1. Extract Amygdala timeline
    2. Average time-course of each script (trauma1, trauma2, trauma3, sad1...relax1...) - should use beta from 1st level instead.
    3. We end up with a matrix of voxels X score for each script
    4. We can correlate the matrices across different runs

Second method (Ifat's idea):
    1. Extract Amg timeline
    2. Average across voxels
    3. corrlate timecourse between different trauma scripts within session (and can be across also)
    
"""
## imports
import sys
import subprocess
import pandas as pd
from nilearn import input_data
from connUtils import removeVars
import numpy as np
#from nipype.interfaces.freesurfer import MRIConvert

project_dir = '/media/Data/work/intraSubject_kpe/'


#%%
def createSessionMat(sub, ses, roi_num, roi_name):
    func_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-Memory_space-T1w_desc-preproc_bold.nii.gz'
    
    lookup_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_desc-aparcaseg_dseg.nii.gz'
    output_file = '/media/Data/KPE_results/sub-' + sub + '_ses-' + ses + '_' + roi_name
    confound_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-Memory_desc-confounds_regressors.tsv'
    events_file = '/media/Data/PTSD_KPE/condition_files/withNumbers/sub-{sub}_ses-{ses}.csv'
  

    cmd = ['fslmaths', str(lookup_file.format(sub=sub, ses=ses)), '-thr', str(roi_num), '-uthr', str(roi_num), str(output_file)]

    if subprocess.call(cmd)==0:
        print(f'Successesfuly ran sub {sub}')
    else:
        print(f'failure to run subject {sub}')
    out_file = output_file + '.nii.gz'
    seed_masker = input_data.NiftiMasker(mask_img = out_file,
        smoothing_fwhm=1,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=1.,
        memory='/media/Data/nilearn', memory_level=1, verbose=2)
    
    time_series = seed_masker.fit_transform(func_file.format(sub=sub, ses=ses), removeVars(confound_file.format(sub=sub, ses=ses)))

    events = pd.read_csv(events_file.format(sub=sub, ses=ses), sep = "\t")

    matrix = []
    dictTrials = {}
    t_i = 1 # trauma script index
    s_i = 1 # sad script index
    r_i = 1 # relax index
    for line in events.iterrows():
        print (f' Proccessing line {line}')
        numberRow = line[0] # take row number to add to matrix name later
        onset = round(line[1].onset) # take onset and round it
        duration = round(line[1].duration)
        trial = line[1].trial_type
        if trial=='trauma':
            trial = trial + str(t_i)
            t_i = t_i +1
        elif trial=='sad':
            trial = trial + str(s_i)
            s_i = s_i +1
        elif trial=='relax':
            trial = trial + str(r_i)
            r_i = r_i +1
        
        trial_type = line[1].trial_type
        specTimeline = time_series[onset:(onset+duration),:]
        matrix.append(specTimeline)
        dictTrials.update({trial: specTimeline})
    
    #matrix = np.array(matrix)
    # Create one dimension ndArray from a list of lists
    
   # df = pd.DataFrame.from_dict(dictTrials, orient='index')
    #df.to_csv(project_dir+sub+'_ses-' + ses +'.csv', index=False)
    return dictTrials
# average time-course of each script
# combine to one matrix (Nvoxels X scriptAverage)
#%% 
# correlate each row of the two matrices
# from scipy import stats
# totalCor = []
# for line1, line2 in zip(df1.iterrows(), df2.iterrows()):
#    a1 = line1[1][0]
#    a2 = line2[1][0]
#    cor = []
#    for i in range(a1.shape[1]):
#        cor.append(stats.pearsonr(a1[:,i], a2[:,i])[0])
#    totalCor.append(cor)

# import seaborn as sns
# import matplotlib.pyplot as plt
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# #cmap = sns.diverging_palette(110, 10, as_cmap=True)
# cmap = sns.color_palette("coolwarm")
# k = np.array(totalCor)
# sns.heatmap(k, cmap=cmap)

#%% Ifat method

def calculateCor(df):
    from scipy import stats
    trauma_arr = []
    for line in df.items():
        print(line[1])
        
        if line[0].find('trauma')!= -1:
            trauma_arr.append(line[1])
        
    # average across columns and correlate
    trauma1_mean = np.average(trauma_arr[0], axis=1)
    trauma2_mean = np.average(trauma_arr[1], axis=1)
    trauma3_mean = np.average(trauma_arr[2], axis=1)
    # correlate trauma1 and 2
    trt1_2 = stats.pearsonr(trauma1_mean, trauma2_mean)
    trt1_3 = stats.pearsonr(trauma1_mean, trauma3_mean)
    trt2_3 = stats.pearsonr(trauma2_mean, trauma3_mean)
    return [trt1_2,trt1_3, trt2_3]
#%% resting state calculations
def createSessionMatRest(sub, ses, roi_num, roi_name):
    func_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_space-T1w_desc-preproc_bold.nii.gz'
    
    lookup_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_desc-aparcaseg_dseg.nii.gz'
    output_file = '/media/Data/KPE_results/sub-' + sub + '_ses-' + ses + '_' + 'rest_' + roi_name
    confound_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_desc-confounds_regressors.tsv'

  

    cmd = ['fslmaths', str(lookup_file.format(sub=sub, ses=ses)), '-thr', str(roi_num), '-uthr', str(roi_num), str(output_file)]

    if subprocess.call(cmd)==0:
        print(f'Successesfuly ran sub {sub}')
    else:
        print(f'failure to run subject {sub}')
    out_file = output_file + '.nii.gz'
    seed_masker = input_data.NiftiMasker(mask_img = out_file,
        smoothing_fwhm=1,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=1.,
        memory='/media/Data/nilearn', memory_level=1, verbose=2)
    
    time_series = seed_masker.fit_transform(func_file.format(sub=sub, ses=ses), removeVars(confound_file.format(sub=sub, ses=ses)))
    
    duration_seg = time_series.shape[0]
    duration = round(duration_seg / 5)
    matrix = []
    dictTrials = {}
    onset = 0
    for i in range(5):
        trial = 'part' + str(i)
        specTimeline = time_series[onset:(onset+duration),:]
        matrix.append(specTimeline)
        dictTrials.update({trial: specTimeline})
        onset = onset + duration
    return dictTrials

def calculateCorRS(df):
    from scipy import stats
    rest_arr = []
    for line in df.items():
        print(line[1])
        rest_arr.append(line[1])
    # average across columns and correlate
    part1_mean = np.average(rest_arr[0], axis=1)
    part2_mean = np.average(rest_arr[1], axis=1)
    part3_mean = np.average(rest_arr[2], axis=1)
    part4_mean = np.average(rest_arr[3], axis=1)
    part5_mean = np.average(rest_arr[4], axis=1)
    # correlate trauma1 and 2
    rest1_2 = stats.pearsonr(part1_mean, part2_mean)
    rest1_3 = stats.pearsonr(part1_mean, part3_mean)
    rest1_4 = stats.pearsonr(part1_mean, part4_mean)
    rest1_5 = stats.pearsonr(part1_mean, part5_mean)
    rest2_3 = stats.pearsonr(part2_mean, part3_mean)
    rest2_4 = stats.pearsonr(part2_mean, part4_mean)
    return [rest1_2,rest1_3, rest1_4,rest1_5,rest2_3, rest2_4]

