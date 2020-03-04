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

data_dir = os.path.abspath('/media/Data/')
output_dir = '/media/Data/work/kpeTask'
removeTR = 4
fwhm = 4
tr = 1
#%% Methods 
def _bids2nipypeinfo(in_file, events_file, regressors_file, removeTR = 4,
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
    np.savetxt(out_motion, regress_data[motion_columns].values[removeTR:,], '%g')
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    runinfo = Bunch(
        scans=in_file,
        conditions=list(set(events.trial_type_30.values)),
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:
        event = events[events.trial_type_30.str.match(condition)]

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
subject_list = ['008']#,'1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464','1468', '1480','1499']
# Map field names to individual subject runs.
session = '1' # choose session

infosource = pe.Node(util.IdentityInterface(fields=['subject_id'
                                            ],
                                    ),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'func': data_dir +  '/KPE_BIDS/derivatives/fmriprep/sub-{subject_id}/ses-' + session + '/func/sub-{subject_id}_ses-' + session + '_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
             'mask': data_dir + '/KPE_BIDS/derivatives/fmriprep/sub-{subject_id}/ses-' + session + '/func/sub-{subject_id}_ses-' + session + '_task-Memory_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
             'regressors': data_dir + '/KPE_BIDS/derivatives/fmriprep/sub-{subject_id}/ses-' + session + '/func/sub-{subject_id}_ses-' + session + '_task-Memory_desc-confounds_regressors.tsv',
             'events':  data_dir + '/KPE_BIDS/condition_files/withNumbers/sub-{subject_id}_ses-' + session + '_30sec_window' + '.csv'}
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

cont1 = ['Trauma1_0>Sad1_0', 'T', ['trauma1_0', 'sad1_0'], [1, -1]]
cont2 = ['Trauma1_0>Relax1_0', 'T', ['trauma1_0', 'relax1_0'], [1, -1]]
cont3 = ['Sad1_0>Relax1_0', 'T', ['sad1_0', 'relax1_0'], [1, -1]]
cont4 = ['Sad1', 'T', ['sad1_0'], [1]]
cont5 = ['Trauma1_0>Trauma1_1_2', 'T', ['trauma1_0', 'trauma1_1','trauma1_2'], [1, -0.5, -0.5]]
cont6 = ['Trauma1 > Trauma2', 'T', ['trauma1_0', 'trauma1_1', 'trauma1_2', 'trauma1_3', 'trauma2_0', 'trauma2_1', 'trauma2_2', 'trauma2_3'], [0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25 ]]
contrasts = [cont1, cont2, cont3, cont4, cont5, cont6]
#%% Addinf simple denozining procedures (remove dummy scans, smoothing, art detection) 
extract = Node(fsl.ExtractROI(t_size=-1, output_type='NIFTI'),
              name="extract")
extract.inputs.t_min = removeTR

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
modelspec.inputs.input_units = 'secs'
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
wfSPM = Workflow(name="l1spm", base_dir= output_dir)
wfSPM.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id')]),
        (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
        (selectfiles, extract, [('func','in_file')]),
        (extract, smooth, [('roi_file','in_files')]),
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
datasink = Node(nio.DataSink(base_directory= output_dir),
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


wfSPM.run('MultiProc', plugin_args={'n_procs': 5})

