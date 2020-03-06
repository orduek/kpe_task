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
import pandas as pd

from nipype.interfaces.matlab import MatlabCommand
#mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
MatlabCommand.set_default_paths('/home/or/Downloads/spm12/') # set default SPM12 path in my computer. 

data_dir = os.path.abspath('/media/Data/KPE_BIDS/derivatives/fmriprep')
#output_dir = '/media/Data/KPE_results/work/kpeTask_ses2/1stLevel/_subject_id_1263/con_0001.nii
fwhm = 4
tr = 1

input_dir = '/media/Data/KPE_results/work/kpeTask_ses2/1stLevel'

#%%
medication_cond = pd.read_csv('/home/or/kpe_task_analysis/task_based_analysis/kpe_sub_condition.csv')

ketamine_list = list(medication_cond['scr_id'][medication_cond['med_cond']==1])
ket_list = []
for subject in ketamine_list:
    print(subject)
    sub = subject.split('KPE')[1]
    ket_list.append(sub)

ket_list.remove('1223')
midazolam_list = list(medication_cond['scr_id'][medication_cond['med_cond']==0])
mid_list = []
for subject in midazolam_list:
    print(subject)
    sub = subject.split('KPE')[1]
    mid_list.append(sub)
mid_list.remove('1480')

#%% Adding data sink
########################################################################
# Datasink
datasink = Node(nio.DataSink(base_directory='/media/Data/work/KPE_SPM_ses2_group/Sink'),
                                         name="datasink")
                       

#%% Gourp analysis - based on SPM - should condifer the fsl Randomize option (other script)
# OneSampleTTestDesign - creates one sample T-Test Design

twoSampleTtest = Node(spm.TwoSampleTTestDesign(), 
                      name = 'twosampleTtest')

# EstimateModel - estimates the model
level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

# EstimateContrast - estimates group contrast
level2conestimate = Node(spm.EstimateContrast(group_contrast=True),
                         name="level2conestimate")
#cont1 = ['Group', 'T', ['mean'], [1]]
cont1 = ['ketamine > Midazolam', 'T', ['Group_{1}','Group_{2}'], [1, -1]]
cont2 = ['midazolam > ketamine', 'T', ['Group_{1}','Group_{2}'], [-1, 1]]

level2conestimate.inputs.contrasts = [cont1, cont2]
level2conestimate.inputs.group_contrast = True
# Which contrasts to use for the 2nd-level analysis
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005', 'con_0006']

# Threshold - thresholds contrasts
level2thresh = MapNode(spm.Threshold(contrast_index=1,
                              use_topo_fdr=True,
                              use_fwe_correction=False, # here we can use fwe or fdr
                              extent_threshold=10,
                              height_threshold= 0.001,
                              extent_fdr_p_threshold = 0.05,
                              height_threshold_type='p-value'),
                                    iterfield = ['stat_image'],
                                   name="level2thresh")
 #Infosource - a function free node to iterate over the list of subject names
infosource = Node(util.IdentityInterface(fields=['contrast_id']),
                  name="infosource")
infosource.iterables = [('contrast_id', contrast_list)]

# SelectFiles - to grab the data (alternative to DataGrabber)


templates = {'cons': opj(input_dir, '_subject_id_{subject_id}', '{contrast_id}.nii')}
selectfilesKet = MapNode(SelectFiles(templates), iterfield=['subject_id'],
                                         name="selectfilesKet")
selectfilesKet.inputs.subject_id = ket_list

selectfilesMid = MapNode(SelectFiles(templates), iterfield = ['subject_id'],
                                                              
                   name="selectfilesMid")
selectfilesMid.inputs.subject_id = mid_list


l2analysis = Workflow(name='spm_l2analysisGroup')
l2analysis.base_dir = '/media/Data/work/KPE_SPM_ses2'

l2analysis.connect([(infosource, selectfilesKet, [('contrast_id', 'contrast_id'),
                                               ]),
                    (infosource, selectfilesMid, [('contrast_id', 'contrast_id'),
                                               ]),
                    
                    (selectfilesKet, twoSampleTtest, [('cons', 'group1_files')]),
                    (selectfilesMid, twoSampleTtest, [('cons', 'group2_files')]),
                    
                    (twoSampleTtest, level2estimate, [('spm_mat_file',
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
l2analysis.run('MultiProc', plugin_args={'n_procs': 5})


