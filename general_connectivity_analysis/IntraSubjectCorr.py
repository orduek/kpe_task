#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:30:11 2020

@author: Or DUek
Create Intra-Subject Correlation focusing on the amygdala. 
Basic method:
    1. Extract Amygdala timeline
    2. Average time-course of each script (trauma1, trauma2, trauma3, sad1...relax1...)
    3. We end up with a matrix of voxels X score for each script
    4. We can correlate the matrices across different runs
"""
## imports
import sys
import subprocess
import pandas as pd
from nilearn import input_data
from connUtils import removeVars
import numpy as np
from nipype.interfaces.freesurfer import MRIConvert

project_dir = '/media/Data/work/intraSubject_kpe/'


roi_num = 18 # set ROI
roi_name = "leftAmg" # set name
sub = '1223' # set subject
ses = '1' # set session
# First extract amygdala from native space
#%%
def createSessionMat(sub, ses, roi_num, roi_name):
    func_file = '/media/Data/KPE_BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-Memory_bold.nii.gz'
    mask_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_desc-brain_mask.nii.gz'
    #lookup_file = '/media/Data/KPE_BIDS/reconOutput/sub-{sub}_ses-{ses}/mri/aparc+aseg.mgz'
    lookup_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_desc-aparcaseg_dseg.nii.gz'
    anat_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_desc-preproc_T1w.nii.gz'
    seg_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_label-WM_probseg.nii.gz'
    #lookup_file_nifti = project_dir + sub + '_ses-' + ses + '.nii.gz'
    output_file = '/media/Data/KPE_results/sub-' + sub + '_ses-' + ses + '_' + roi_name
    confound_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-Memory_desc-confounds_regressors.tsv'
    events_file = '/media/Data/PTSD_KPE/condition_files/sub-{sub}_ses-{ses}.csv'
    ##
    #mc = MRIConvert()
    #mc.inputs.in_file = lookup_file.format(sub=sub, ses=ses)
    #mc.inputs.out_file = lookup_file_nifti
    #mc.inputs.out_type = 'niigz'
    #mc.inputs.out_orientation = 'RAS'
    #mc.run()

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
    for line in events.iterrows():
        print (f' Proccessing line {line}')
        numberRow = line[0] # take row number to add to matrix name later
        onset = round(line[1].onset) # take onset and round it
        duration = round(line[1].duration)
        trial_type = line[1].trial_type
        specTimeline = time_series[onset:(onset+duration),:]
        matrix.append(specTimeline)
        
    matrix = np.array(matrix)
    df = pd.DataFrame({'trial_type': events.trial_type})
    df = pd.concat([df,pd.DataFrame(matrix)], axis=1)
    df.to_csv(project_dir+sub+'_ses-' + ses +'.csv', index=False)
    return df
# average time-course of each script
# combine to one matrix (Nvoxels X scriptAverage)
#%%
df1 = createSessionMat('1315','1', 54, 'rightAmg')
df2 = createSessionMat('1315','2', 54, 'rightAmg')
# correlate each row of the two matrices
from scipy import stats
totalCor = []
for line1, line2 in zip(df1.iterrows(), df2.iterrows()):
   a1 = line1[1][0]
   a2 = line2[1][0]
   cor = []
   for i in range(a1.shape[1]):
       cor.append(stats.pearsonr(a1[:,i], a2[:,i])[0])
   totalCor.append(cor)

import seaborn as sns
import matplotlib.pyplot as plt
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(110, 10, as_cmap=True)
cmap = sns.color_palette("coolwarm")
k = np.array(totalCor)
sns.heatmap(k, cmap=cmap)
#%% Registration of native space EPI with average reconstructed T1 (should do before extracting amygdala)
from nipype.interfaces import afni
al_ea = afni.AlignEpiAnatPy()
al_ea.inputs.anat = anat_file.format(sub=sub)
al_ea.inputs.in_file = func_file.format(sub=sub, ses=ses)
al_ea.inputs.epi_base = 0
al_ea.inputs.epi_strip = '3dAutomask'
al_ea.inputs.volreg = 'off'
al_ea.inputs.tshift = 'off'
al_ea.inputs.epi2anat = True
al_ea.inputs.save_skullstrip = True

al_ea.cmdline 
al_ea.run()

from nipype.interfaces import afni
a2n = afni.AFNItoNIFTI()
a2n.inputs.in_file = '/home/or/kpe_task_analysis/sub-1315_ses-1_task-Memory_bold_al+orig.BRIK'
a2n.inputs.out_file =  'sub-1315_epialign.nii'
a2n.cmdline
a2n.run()


allineate = afni.Allineate()
allineate.inputs.in_file = func_file.format(sub=sub, ses=ses)
allineate.inputs.reference = lookup_file.format(sub=sub)
allineate.cmdline
allineate.run()


#%%
!epi_reg --epi=func_file.format(sub=sub,ses=ses) --t1=anat_file.format(sub=sub) --t1brain=mask_file.format(sub=sub,ses=ses) --out=<output name>
cmd = ['epi_reg', '--wmseg=' + str(seg_file.format(sub=sub)),'--epi=' + str(func_file.format(sub=sub, ses=ses)), '--t1=' + str(anat_file.format(sub=sub)), '--t1brain=' + str(mask_file.format(sub=sub,ses=ses)),
       '--out=' +str(output_file)]

subprocess.call(cmd)

epi_reg --wmseg=T1_wmseg.nii.gz --epi=func.nii.gz --t1=T1.nii.gz --t1brain=T1_brain.nii.gz --out=func2struct_nofmap



#%%
from nipype.interfaces.ants import MeasureImageSimilarity
sim = MeasureImageSimilarity()
sim.inputs.dimension = 3
sim.inputs.metric = 'MI'
sim.inputs.fixed_image = 'T1.nii'
sim.inputs.moving_image = 'resting.nii'
sim.inputs.metric_weight = 1.0
sim.inputs.radius_or_number_of_bins = 5
sim.inputs.sampling_strategy = 'Regular'
sim.inputs.sampling_percentage = 1.0
sim.inputs.fixed_image_mask = 'mask.nii'
sim.inputs.moving_image_mask = 'mask.nii.gz'
sim.cmdline