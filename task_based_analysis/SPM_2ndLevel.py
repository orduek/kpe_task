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

input_dir = '/media/Data/KPE_results/work/kpeTask_ses2/1stLevel'


#%% Adding data sink
########################################################################
# Datasink
datasink = Node(nio.DataSink(base_directory='/media/Drobo/work/KPE_SPM/Sink_ses-2'),
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
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005', 'con_0006']

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

templates = {'cons': opj(input_dir, '_sub*', '{contrast_id}.nii')}
selectfiles = Node(SelectFiles(templates,
                               
                               sort_filelist=True),
                   name="selectfiles")



l2analysis = Workflow(name='spm_l2analysisWorking_Ses1')
l2analysis.base_dir = opj(data_dir, '/media/Data/work/KPE_SPM_ses2')

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


