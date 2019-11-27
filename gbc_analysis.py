#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:37:15 2019

@author: Or Duek
Global Correlation and Global Correlation regression
"""
import nilearn
from nilearn import input_data
import pandas as pd
import numpy as np

mask_file = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-008/ses-1/func/sub-008_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'#set a file to create the same voxel map for all subjects

brain_masker = input_data.NiftiMasker(mask_img = mask_file,
        smoothing_fwhm=4,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=1.,
        memory='/media/Data/nilearn', memory_level=1, verbose=2)

#from nilearn.regions import Parcellations
#ward = Parcellations(method='ward', n_parcels=1000,
#                     standardize=True, detrend=True, smoothing_fwhm=2.,
#                     low_pass=0.1, high_pass=0.01, t_r=1.,
#                     memory='/media/Data/nilearn', memory_level=1,
#                     verbose=1)


def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    import numpy as np
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['csf', 'white_matter', 'framewise_displacement',
                          'a_comp_cor_00', 'a_comp_cor_01',	'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']] # can add 'global_signal' also
     # change NaN of FD to zero
    finalConf = np.array(finalConf)
    finalConf[0,2] = 0 # if removing FD than should remove this one also
    return finalConf


#from nilearn.input_data import NiftiMapsMasker
#from nilearn.input_data import NiftiLabelsMasker
## in this mask we standardize the values, so mean is 0 and between -1 to 1
## masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, smoothing_fwhm = 6,
##                          memory="/home/oad4/scratch60/shenPar_nilearn",high_pass=.01 , low_pass = .1, t_r=1, verbose=5)
#
## use different masker when using Yeo atlas. 
#masker = NiftiLabelsMasker(labels_img=ward_labels_img, standardize=True, smoothing_fwhm = 6,
#                        memory="/media/Data/nilearn",t_r=1, verbose=5, high_pass=.01 , low_pass = .1) # As it is task based we dont' bandpassing high_pass=.01 , low_pass = .1)

#%% run and create voxelwise timeseries
    
func_filename =  '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-008/ses-2/func/sub-008_ses-2_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confound_filename = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-008/ses-2/func/sub-008_ses-2_task-rest_desc-confounds_regressors.tsv'


brain_time_series = brain_masker.fit_transform(func_filename, confounds=removeVars(confound_filename))
np.save('brainSeries', brain_time_series)


arr = np.load('brainSeries.npy', mmap_mode='r')
#ward_time_series = ward.fit_transform(func_filename, confounds=confound_filename)
# create correlation matrix
from nilearn.connectome import ConnectivityMeasure
#correlation_measure = ConnectivityMeasure(kind='partial correlation') # can choose partial - it might be better
correlation_measure = ConnectivityMeasure(kind='correlation') # can choose partial - it might be better      
correlation_matrix = correlation_measure.fit_transform([arr])[0]

# run a loop 
brainSeries = np.reshape(arr, (brain_time_series.shape[1], brain_time_series.shape[0]))
cor = np.corrcoef(brainSeries)
#%%

ward_labels_img = ward.labels_img_

# Now, ward_labels_img are Nifti1Image object, it can be saved to file
# with the following code:
ward_labels_img.to_filename('ward_parcellation.nii.gz')

from nilearn import plotting
from nilearn.image import mean_img, index_img

first_plot = plotting.plot_roi(ward_labels_img, title="Ward parcellation",
                               display_mode='xz')

# Grab cut coordinates from this plot to use as a common for all plots
cut_coords = first_plot.cut_coords

#%%
time_seris = masker.fit_transform(func_filename, confounds=removeVars(confound_filename))

#%%
correlation_matrix = correlation_measure.fit_transform([time_seris])[0]
z_transformed = np.arctan(correlation_matrix)
#%% Calculate global correlation for each 

# I need to sum up each row, remove 1 (correlation with self) and divide by 1000 (ncol). 


trt_ses1 = creatSubCor(subject_list, '2', 'traumaTrials.npy') 

# use the 1st trauma script connectivity measure 
totalGBC = []
for mat in trt_ses1:
    print(mat)
    z_transformed = np.arctan(mat)
    meanMat = np.mean(z_transformed, axis = 1)
    totalGBC.append(meanMat)
    
totalGBC = np.array(totalGBC)
np.save('totalGBC_2stTrauma', totalGBC)


