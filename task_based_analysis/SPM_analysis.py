#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:29:02 2019

@author: Or Duek
Running same analysis but with SPM instead of FSL
This analysis uses fmriPrep output files, and using a method described her: https://www.biorxiv.org/content/10.1101/694364v1
but with SPM as the analysis software. 
You need to organize event files (tab seperated) for each session.
"""

import os  # system functions

from nipype import config
# config.enable_provenance()

from nipype.interfaces import spm, fsl

import scipy.io as spio
import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.rapidart as ra  # artifact detection
import nipype.algorithms.modelgen as model  # model specification
from nipype.algorithms.rapidart import ArtifactDetect
from nipype.algorithms.misc import Gunzip
from nipype import Node, Workflow, MapNode
from nipype import SelectFiles
from os.path import join as opj


from nipype.interfaces.matlab import MatlabCommand
#mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
MatlabCommand.set_default_paths('/home/or/Downloads/spm12/') # set default SPM12 path in my computer. 

data_dir = os.path.abspath('/media/Data/KPE_BIDS/derivatives/fmriprep')
output_dir = '/media/Drobo/work/kpeTask'
fwhm = 4
tr = 1
#%% Methods 
def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch
    removeTR = 0
    # Process the events file
    events = pd.read_csv(events_file, sep=r'\s+')

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()

    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    np.savetxt(out_motion, regress_data[motion_columns].values, '%g')
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    runinfo = Bunch(
        scans=in_file,
        conditions=list(set(events.trial_type_N.values)),
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:
        event = events[events.trial_type_N.str.match(condition)]

        runinfo.onsets.append(np.round(event.onset.values-removeTR, 3).tolist()) # added -removeTR to align to the onsets after removing X number of TRs from the scan
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values[removeTR:,].T.tolist() # adding removeTR to cut the first rows

    return [runinfo], str(out_motion)
#%%
subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464','1468', '1480','1499']
# Map field names to individual subject runs.
session = '1' # choose session

#%% Adding data sink
########################################################################
# Datasink
datasink = Node(nio.DataSink(base_directory='/media/Drobo/work/KPE_SPM/Sink_ses-1'),
                                         name="datasink")
                       

#%% Gourp analysis - based on SPM - should condifer the fsl Randomize option (other script)
# OneSampleTTestDesign - creates one sample T-Test Design
onesamplettestdes = Node(spm.OneSampleTTestDesign(),
                         name="onesampttestdes")

# EstimateModel - estimates the model
level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

# EstimateContrast - estimates group contrast
level2conestimate = Node(spm.EstimateContrast(group_contrast=True),
                         name="level2conestimate")
cont1 = ['Group', 'T', ['mean'], [1]]
level2conestimate.inputs.contrasts = [cont1]

# Which contrasts to use for the 2nd-level analysis
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005']

# Threshold - thresholds contrasts
level2thresh = Node(spm.Threshold(contrast_index=1,
                              use_topo_fdr=True,
                              use_fwe_correction=False, # here we can use fwe or fdr
                              extent_threshold=10,
                              height_threshold= 0.05,
                              extent_fdr_p_threshold = 0.05,
                              height_threshold_type='p-value'),
                              
                                   name="level2thresh")
 #Infosource - a function free node to iterate over the list of subject names
infosource = Node(util.IdentityInterface(fields=['contrast_id']),
                  name="infosource")
infosource.iterables = [('contrast_id', contrast_list)]

# SelectFiles - to grab the data (alternative to DataGrabber)

templates = {'cons': opj('/media/Data/KPE_results/work/1stLevel/_sub*/', 
                         '{contrast_id}.nii')}
selectfiles = Node(SelectFiles(templates,
                               
                               sort_filelist=True),
                   name="selectfiles")



l2analysis = Workflow(name='spm_l2analysisWorking')
l2analysis.base_dir = opj(data_dir, '/media/Data/work/KPE_SPM')

l2analysis.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                               ]),
                    (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                    
                    (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                          'spm_mat_file')]),
                    (level2estimate, level2conestimate, [('spm_mat_file',
                                                          'spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')]),
                    (level2conestimate, level2thresh, [('spm_mat_file',
                                                        'spm_mat_file'),
                                                       ('spmT_images',
                                                        'stat_image'),
                                                       ]),
                    (level2conestimate, datasink, [('spm_mat_file',
                        '2ndLevel.@spm_mat'),
                       ('spmT_images',
                        '2ndLevel.@T'),
                       ('con_images',
                        '2ndLevel.@con')]),
                    (level2thresh, datasink, [('thresholded_map',
                                               '2ndLevel.@threshold')]),
                                                        ])
# %%                                                     
l2analysis.run('MultiProc', plugin_args={'n_procs': 4})

# %% plotting 2nd level results
from nilearn.plotting import plot_glass_brain
import nilearn.plotting
import glob
conImages = glob.glob('/media/Data/work/KPE_SPM/Sink/2ndLevel/_contrast_id_con_000*/spmT_0001.nii')
for conImage in conImages:
    plot_glass_brain(conImage,colorbar=True,
     threshold=2.3, display_mode='lyrz', black_bg=True)#, vmax=10);      
    
plot_glass_brain('/media/Data/work/KPE_SPM/Sink_ses-1/2ndLevel/_contrast_id_con_0001/spmT_0001_thr.nii', threshold = 2.3)
plot_glass_brain('/media/Data/work/KPE_SPM/Sink_ses-1/2ndLevel/_contrast_id_con_0004/spmT_0001_thr.nii', threshold = 2.3)
plot_glass_brain('/media/Data/work/KPE_SPM/Sink_ses-1/2ndLevel/_contrast_id_con_0005/spmT_0001_thr.nii', threshold = 2.3)

### 2 session
plot_glass_brain('/media/Data/work/KPE_SPM/Sink_ses-2/2ndLevel/_contrast_id_con_0001/spmT_0001.nii', threshold = 5.5, colorbar=True)
plot_glass_brain('/media/Data/work/KPE_SPM/Sink_ses-2/2ndLevel/_contrast_id_con_0005/spmT_0001_thr.nii', threshold = 2.3)
plot_glass_brain('/media/Data/work/KPE_SPM/Sink_ses-2/2ndLevel/_contrast_id_con_0004/spmT_0001_thr.nii', threshold = 2.3)
%matplotlib qt
nilearn.plotting.plot_stat_map(nilearn.image.smooth_img('/media/Data/work/KPE_SPM/Sink_ses-2/2ndLevel/_contrast_id_con_0001/spmT_0001.nii', 3), display_mode='x',
                                      threshold=5.5, bg_img=anatimg, dim=1)#%% Stat maps
for con_image in conImages:
    nilearn.plotting.plot_stat_map(con_image, display_mode='ortho',
                              threshold=2.3)#, cut_coords=(20,55)) #, 10, 15), dim=1) 
#%% Surface display
import nibabel as nib

from nilearn.plotting import plot_stat_map
from nilearn import datasets, surface, plotting
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()
from nilearn import surface

texture = surface.vol_to_surf('/media/Data/work/KPE_SPM/Sink_ses-2/2ndLevel/_contrast_id_con_0001/spmT_0001_thr.nii', fsaverage.pial_right)
from nilearn import plotting
a = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-1263/ses-1/func/sub-1263_ses-1_task-Memory_space-fsaverage5_hemi-R.func.gii'
plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                            title='Surface right hemisphere', colorbar=True,
                            threshold=0., bg_map=a)#fsaverage.sulc_right)
big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
fmri_img = nib.load('/media/Data/work/KPE_SPM/Sink_ses-2/2ndLevel/_contrast_id_con_0001/spmT_0001_thr.nii')
big_texture_right = surface.vol_to_surf(fmri_img, big_fsaverage.pial_right)
%matplotlib inline

plotting.plot_surf_stat_map(big_fsaverage.infl_right,
                            big_texture_right, 
                            hemi='right', colorbar=True,
                            title='',
                            threshold=3, 
                            bg_map=big_fsaverage.sulc_right)
plotting.show()
%matplotlib qt
plotting.view_surf(big_fsaverage.infl_left, big_texture_right, threshold='95%',
                          bg_map=big_fsaverage.sulc_left)

