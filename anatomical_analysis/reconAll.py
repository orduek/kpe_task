#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:42:40 2019

@author: Or Duek
This script runs recon-all on all (number of) sessions and then continue to label subregions of hippocampus and amygdala
"""


import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from nipype.interfaces.freesurfer.preprocess import ReconAll
from nipype.interfaces.utility import Function
import nipype.interfaces.utility as util  # utility
import os

from hippSeg import CustomHippoSeg # this is a costum made node - so you'll need the file hippSeg.py in the same folder
data_dir = '/media/Data/kpe_forFmriPrep/'
subjects_dir = '/home/or/Documents/reconAll' # this is FreeSurfer's subject dir. i.e output

wf = pe.Workflow(name="l1workflow")
wf.base_dir = os.path.abspath('/media/Data/workdir')
#
#

# Create a simple function that takes recon_all output and returns a string 
def changeToString(arr):
    return arr[0]


subject_list = ['1263']
session_list = ['2']
# Map field names to individual subject runs.


infosource = pe.Node(util.IdentityInterface(fields=['subject_id'
                                            ],
                                    ),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]


infosource_task = pe.Node(util.IdentityInterface(fields = ['session']),
                                                 name = "infosource_task")
infosource_task.iterables = [('session', session_list)]
# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'anat': '/media/Data/kpe_forFmriPrep/sub-{subject_id}/ses-{session}/anat/sub-{subject_id}_ses-{session}_T1w.nii.gz',
             }
selectfiles = pe.Node(nio.SelectFiles(templates,
                               base_directory=data_dir),
                   name="selectfiles")

recon_all = pe.MapNode(
    interface=ReconAll(),
    name='recon_all',
    iterfield=['subject_id'])
recon_all.inputs.subject_id = subject_list
recon_all.inputs.directive = "all"
if not os.path.exists(subjects_dir):
    os.mkdir(subjects_dir)
recon_all.inputs.subjects_dir = subjects_dir #Here we use specific directory, in order to avoid crash when trying to create simlinks. 
#recon_all.inputs.hippocampal_subfields_T1 = True # add hippocampal subfields

# home made Node to run hippocampus and amygdala segmentation.
## Now we should add another node that will run the hippocampal subfield script:
## Look here for instructions on how to run and wrap it in a node?
## https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala

hippSeg = pe.Node(interface = CustomHippoSeg(), name = 'hippSeg')
#hippSeg.inputs.subject_dir = subjects_dir

changeToString= pe.Node(name='changeToString',
               interface=Function(input_names=['arr'],
                                  output_names=['arr'],
                                  function=changeToString))
  
wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
            (infosource, recon_all, [('subject_id', 'subject_id')]),
            (infosource_task, selectfiles, [('session','session')]),
            (selectfiles, recon_all, [('anat','T1_files')]),
            (infosource, hippSeg, [('subject_id', 'subject_id')]),
            (recon_all, changeToString, [('subjects_dir', 'arr')]),
            (changeToString, hippSeg, [('arr','subject_dir')])
        ])


wf.run("MultiProc", plugin_args={'n_procs': 4})

