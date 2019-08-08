#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:52:11 2019

@author: Or Duek
Nistat based first level analysis
Taken from example page: https://nistats.github.io/auto_examples/01_tutorials/plot_first_level_model_details.html#sphx-glr-auto-examples-01-tutorials-plot-first-level-model-details-py

"""
#%% Confound method
def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['a_comp_cor_00', 'a_comp_cor_01',	'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']] # can add 'global_signal' also
     # change NaN of FD to zero
    finalConf = np.array(finalConf)
    
    return finalConf
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fmri_img = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/fmriprep/sub-1223/ses-1/func/sub-1223_ses-1_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confoundFile = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/fmriprep/sub-1223/ses-1/func/sub-1223_ses-1_task-Memory_desc-confounds_regressors.tsv'
t_r = 1
events_file = '/media/Data/PTSD_KPE/condition_files/sub-1223_ses-1.csv'

events= pd.read_csv(events_file)
events = events[['trial_type','onset','duration']]


from nistats.first_level_model import FirstLevelModel
first_level_model = FirstLevelModel(t_r,  hrf_model='spm', memory = '/media/Data/work', smoothing_fwhm=6, memory_level=2)


first_level_model = first_level_model.fit(fmri_img, events=events, confounds=pd.DataFrame(removeVars(confoundFile)))
design_matrix = first_level_model.design_matrices_[0]

from nistats.reporting import plot_design_matrix
plot_design_matrix(design_matrix)

plt.show()




def make_localizer_contrasts(design_matrix):
    """ returns a dictionary of four contrasts, given the design matrix"""

    # first generate canonical contrasts
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])

    # Add more complex contrasts
    contrasts['Trauma'] = (contrasts['trauma']
                          )
    contrasts['Sad'] = (contrasts['sad']
                          )
    contrasts['Relax'] = (contrasts['relax']
                          )
    #contrasts['computation'] = contrasts['calculaudio'] + contrasts['calculvideo']
    #contrasts['sentences'] = contrasts['phraseaudio'] + contrasts['phrasevideo']

    # Short dictionary of more relevant contrasts
    contrasts = {
        'Trauma-Relax': (contrasts['Trauma']
                       - contrasts['Relax']
                       ),
        'Trauma-Sad': contrasts['Trauma'] - contrasts['Sad'],
        
    }
    return contrasts


contrasts = make_localizer_contrasts(design_matrix)
plt.figure(figsize=(5, 9))
from nistats.reporting import plot_contrast_matrix
for i, (key, values) in enumerate(contrasts.items()):
    ax = plt.subplot(5, 1, i + 1)
    plot_contrast_matrix(values, design_matrix=design_matrix, ax=ax)

plt.show()

from nilearn import plotting

def plot_contrast(first_level_model):
    """ Given a first model, specify, enstimate and plot the main contrasts"""
    design_matrix = first_level_model.design_matrices_[0]
    # Call the contrast specification within the function
    contrasts = make_localizer_contrasts(design_matrix)
    fig = plt.figure(figsize=(11, 3))
    # compute the per-contrast z-map
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        ax = plt.subplot(1, len(contrasts), 1 + index)
        z_map = first_level_model.compute_contrast(
            contrast_val, output_type='z_score')
        plotting.plot_stat_map(
            z_map, display_mode='z', threshold=2.5, title=contrast_id, axes=ax, 
            cut_coords=1)

plot_contrast(first_level_model)
plt.show()

