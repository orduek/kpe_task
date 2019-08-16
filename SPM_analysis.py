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

data_dir = os.path.abspath('/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep')
output_dir = '/media/Data/work/kpeTask'
fwhm = 6
tr = 1
#%%
#%% Methods 
def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

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
        conditions=list(set(events.trial_type.values)),
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:
        event = events[events.trial_type.str.match(condition)]

        runinfo.onsets.append(np.round(event.onset.values, 3).tolist())
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values.T.tolist()

    return [runinfo], str(out_motion)
#%%
subject_list = ['1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464']
# Map field names to individual subject runs.


infosource = pe.Node(util.IdentityInterface(fields=['subject_id'
                                            ],
                                    ),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'func': '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
             'mask': '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-Memory_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
             'regressors': '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-Memory_desc-confounds_regressors.tsv',
             'events': '/media/Data/PTSD_KPE/condition_files/sub-{subject_id}_ses-1.csv'}
selectfiles = pe.Node(nio.SelectFiles(templates,
                               base_directory=data_dir),
                   name="selectfiles")

#%%

# Extract motion parameters from regressors file
runinfo = pe.Node(util.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo')

# Set the column names to be used from the confounds file
runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
    ['a_comp_cor_%02d' % i for i in range(6)] + ['cosine%02d' % i for i in range(4)]
#%%
cont1 = ['Trauma>Sad', 'T', ['trauma', 'sad'], [1, -1]]
cont2 = ['Trauma>Relax', 'T', ['trauma', 'relax'], [1, -1]]
cont3 = ['Sad>Relax', 'T', ['sad', 'relax'], [1, -1]]
contrasts = [cont1, cont2, cont3]
#%%
gunzip = MapNode(Gunzip(), name='gunzip',
                 iterfield=['in_file'])

#%% Addinf simple denozining procedures (remove dummy scans, smoothing, art detection) 
#extract = Node(fsl.ExtractROI(t_min=4, t_size=-1, output_type='NIFTI'),
#               name="extract")

smooth = Node(spm.Smooth(), name="smooth", fwhm = fwhm)
# Artifact Detection - determines outliers in functional images
#art = Node(ArtifactDetect(norm_threshold=2,
#                          zintensity_threshold=3,
#                          mask_type='spm_global',
#                          parameter_source='FSL',
#                          use_differences=[True, False],
#                          plot_type='svg'),
#           name="art")
#%%

################################################################


modelspec = Node(interface=model.SpecifySPMModel(), name="modelspec") 
modelspec.inputs.concatenate_runs = False
modelspec.inputs.input_units = 'scans'
modelspec.inputs.output_units = 'secs'
#modelspec.inputs.outlier_files = '/media/Data/R_A_PTSD/preproccess_data/sub-1063_ses-01_task-3_bold_outliers.txt'
modelspec.inputs.time_repetition = 1.  # make sure its with a dot 
modelspec.inputs.high_pass_filter_cutoff = 128.

################################################
#modelspec.inputs.subject_info = subjectinfo(subject_id) # run per subject

level1design = pe.Node(interface=spm.Level1Design(), name="level1design") #, base_dir = '/media/Data/work')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = 1.
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'

#######################################################################################################################
# Initiation of a workflow
wfSPM = Workflow(name="l1spm", base_dir="/media/Data/work/KPE_SPM")
wfSPM.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id')]),
        (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
        (selectfiles, gunzip, [('func','in_file')]),
        (gunzip, smooth, [('out_file','in_files')]),
        (smooth, runinfo, [('smoothed_files','in_file')]),
        (smooth, modelspec, [('smoothed_files', 'functional_runs')]),   
        (runinfo, modelspec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
        
        ])
wfSPM.connect([(modelspec, level1design, [("session_info", "session_info")])])




##########################################################################3

level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical': 1}

contrastestimate = pe.Node(
    interface=spm.EstimateContrast(), name="contrastestimate")
#contrastestimate.inputs.contrasts = contrasts
contrastestimate.overwrite = True
contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}
contrastestimate.inputs.contrasts = contrasts


########################################################################
#%% Connecting level1 estimation and contrasts
wfSPM.connect([
         (level1design, level1estimate, [('spm_mat_file','spm_mat_file')]),
         (level1estimate, contrastestimate,
            [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'),
            ('residual_image', 'residual_image')]),
    ])



###############################################################

#%% Adding data sink
########################################################################
# Datasink
datasink = Node(nio.DataSink(base_directory='/media/Data/work/KPE_SPM/Sink'),
                                         name="datasink")
                       

wfSPM.connect([
       # here we take only the contrast ad spm.mat files of each subject and put it in different folder. It is more convenient like that. 
       (contrastestimate, datasink, [('spm_mat_file', '1stLevel.@spm_mat'),
                                              ('spmT_images', '1stLevel.@T'),
                                              ('con_images', '1stLevel.@con'),
                                              ('spmF_images', '1stLevel.@F'),
                                              ('ess_images', '1stLevel.@ess'),
                                              ])
        ])

#%%
wfSPM.run('MultiProc', plugin_args={'n_procs': 3})
#%% simple graph
import nilearn.plotting
%matplotlib qt 
%matplotlib inline
import matplotlib.pyplot as plt
import glob
anatimg = '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-1464/anat/sub-1464_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
tImage = glob.glob('/media/Data/work/KPE_SPM/Sink/1stLevel/_subject_id_1464/spmT_00*.nii')
for con_image in tImage:
    nilearn.plotting.plot_glass_brain(con_image,
                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3)

conImage = glob.glob('/media/Data/work/KPE_SPM/Sink/1stLevel/_subject_id_1464/con_00*.nii') 
for con_image in tImage:
    nilearn.plotting.plot_stat_map(nilearn.image.smooth_img(con_image, 6), display_mode='x',
                                      threshold=2.3, bg_img=anatimg, dim=1) #, cut_coords=(-5, 0, 5, 10, 15), dim=1)
        
#    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(con_image, 6),
#                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3, bg_img=antimg)
#%%
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
contrast_list = ['con_0001', 'con_0002', 'con_0003']

# Threshold - thresholds contrasts
level2thresh = Node(spm.Threshold(contrast_index=1,
                              use_topo_fdr=False,
                              use_fwe_correction=False,
                              #extent_threshold=10,
                              height_threshold= 0.005,
                              extent_fdr_p_threshold = 0.05,
                              height_threshold_type='p-value'),
                              
                                   name="level2thresh")
 #Infosource - a function free node to iterate over the list of subject names
infosource = Node(util.IdentityInterface(fields=['contrast_id']),
                  name="infosource")
infosource.iterables = [('contrast_id', contrast_list)]

# SelectFiles - to grab the data (alternative to DataGrabber)
templates = {'cons': opj('/media/Data/work/KPE_SPM/Sink/1stLevel/_sub*/', 
                         '{contrast_id}.nii')}
selectfiles = Node(SelectFiles(templates,
                               base_directory='/media/Data/work',
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
#%%                                                     
l2analysis.run('MultiProc', plugin_args={'n_procs': 4})

#%% plotting 2nd level results
from nilearn.plotting import plot_glass_brain
import nilearn.plotting
import glob
conImages = glob.glob('/media/Data/work/KPE_SPM/Sink/2ndLevel/_contrast_id_con_000*/spmT_0001.nii')
for conImage in conImages:
    plot_glass_brain(conImage,colorbar=True,
     threshold=3, display_mode='lyrz', black_bg=True)#, vmax=10);      
#%% Stat maps
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

texture = surface.vol_to_surf(conImages[2], fsaverage.pial_right)
from nilearn import plotting

plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                            title='Surface right hemisphere', colorbar=True,
                            threshold=1., bg_map=fsaverage.sulc_right)
big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
fmri_img = nib.load(conImages[2])
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