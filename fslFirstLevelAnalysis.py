#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:09:02 2019

@author: Or Duek
Taken from : https://gist.github.com/daeh/1f04a98c91e1a30d455379dc5983031c
"""

import sys

print(sys.executable)

from IPython.display import Image # Debug

import os  # system functions
from os.path import join as opj

import numpy as np
import pandas as pd

import glob
import json

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
from nipype.interfaces import utility as niu  # Utilities
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as modelgen  # model generation
import nipype.algorithms.rapidart as ra  # artifact detection

from bids.grabbids import BIDSLayout

from nipype.workflows.fmri.fsl import (create_modelfit_workflow, create_fixed_effects_flow)

from nipype import config
#config.enable_debug_mode()
os.chdir('/home/or/Documents/dicom_niix')
import readConditionFiles_r_aPTSD
from extractSubdata import extractInfo

# For testing (COMMENT)
project = 'PTSD_Risk_Amb'
#subject = ['1063' , '1072', '1206', '1244','1273' ,'1291', '1305', '1340', '1345', '1346'] 
task = ['3','4','5','6']
model = 'testmodel'
n_runs = 4
runs = list(range(1, n_runs+1))
f_contrasts = True

# Location of project
# scratch_dir = os.environ['SCRATCH']
project_dir = '/media/Data/FromHPC/output/fmriprep'
#derivatives_dir = opj(project_dir, 'derivatives')
#work_dir = opj('/media/Data/', 'work_analysis', 'l1_model', 'task-{}_model-{}_sub-{}'.format(task, model, subject))
#if not os.path.exists(work_dir): os.makedirs(work_dir, exist_ok=True)


TR = 1.0
fwhm_thr = 5.0
hpcutoff = 128.0
film_thr = -1000.0 # This was a nightmare. Cf. https://github.com/nipy/nipype/issues/2532#issuecomment-380064103 
film_ms = int(5) # Susan mask size, not fwhm

model_serial_correlations = False

inputnode = pe.Node(niu.IdentityInterface(fields=['subject_id', 'task','condition'],
                                          mandatory_inputs=True),
                    'inputnode')
inputnode.iterables = [('task', task)]
#inputnode.inputs.project = project
#inputnode.inputs.subject_id = subject
#inputnode.inputs.task = task
#inputnode.inputs.model = model

# Templates for DataGrabber
func_template = '/media/Data/FromHPC/output/fmriprep/sub-%s/ses-1/func/sub-%s_ses-1_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
anat_template = '/media/Data/FromHPC/output/fmriprep/sub-%s/anat/sub-%s_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
mask_template = '/media/Data/FromHPC/output/fmriprep/sub-%s/ses-1/func/sub-%s_ses-1_task-%s_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
#confounds_template = 'derivatives/fmriprep/sub-%s/func/sub-%s_task-%s_*_run-%03d_desc-confounds_regressors.tsv'
#events_template = 'bids/sub-%s/func/sub-%s_task-%s_*_run-%03d_events.tsv'
# model_template = 'l1_model_specification/task-%s_model-%s/%s*run'LossRisk', 'LossAmb'-%02d.json'
datasource = pe.Node(nio.DataGrabber(infields=['subject_id',
                                               'task',
                                               
                                               ],
                                    outfields=['struct',
                                               'func',
                                               'mask',
                                                 ]),
                 name =   'datasource')

datasource.inputs.base_directory = project_dir
datasource.inputs.template = '*'
datasource.inputs.sort_filelist = True

datasource.inputs.field_template = dict(struct=anat_template,
                                       func=func_template,
                                       mask=mask_template,
                                      )
datasource.inputs.template_args = dict(struct=[['subject_id', 'subject_id']],
                                      func=[['subject_id', 'subject_id', 'task']],
                                      mask=[['subject_id', 'subject_id', 'task']],
                                      )

def iterateIn (subject_id, data_dir, numTask):
    import os
    os.chdir('/home/or/Documents/dicom_niix')
    import readConditionFiles_r_aPTSD
    from extractSubdata import extractInfo
    
    import pandas as pd
    from nipype.interfaces.base import Bunch
    os.chdir('/media/Data/work')
    from readConditionFiles_r_aPTSD import loadmat, readConditions, organizeBlocks 
    from bids.grabbids import BIDSLayout
    
    info = extractInfo(subject_id,  data_dir)
    #contrasts = info[1]
    model_spec = []
    if numTask == '3':
        filename = info[2][0]
        contrasts = info[1]#[0]
        model_spec = info[0][0]
        condition = info[3][0]
    elif numTask == '4':
        filename = info[2][1]
        contrasts = info[1]#[1]
        model_spec = info[0][1]
        condition = info[3][1]
    elif numTask=='5':
        filename = info[2][2]
        contrasts = info[1]#[2]
        model_spec = info[0][2]
        condition = info[3][2]
    elif numTask =='6':
        filename = info[2][3]
        contrasts = info[1]#[3]
        model_spec = info[0][3]
        condition = info[3][3]
    
    return  (contrasts, model_spec, condition) #(filename,

model_param_gen = pe.Node(niu.Function(input_names=['subject_id', 'data_dir', 'numTask'],
                               output_names=['contrasts', 'model_spec', 'condition'],
                               function=iterateIn),
                        
                      name='model_param_gen')
#model_param_gen = pe.Node(niu.Function(input_names=['subject_id', 'project_dir'],
#                                     output_names=['subject_info', 'contrasts', 'filename'],
#                                     function=extractInfo),
#                        'model_param_gen')

model_param_gen.inputs.data_dir = project_dir

#%% Create simple workflow that takes moel specifications and conditions for each subject and each run. then run the level1 workflow
#wfSubjectCond = pe.Workflow('subjectInfo', base_dir='/media/Data/work')
#wfSubjectCond.connect([
#        (inputnode, datasource, [('subject_id', 'subject_id'), ('task', 'task'),
#        ]), 
#    (inputnode, model_param_gen, [('subject_id', 'subject_id'),('task','numTask')]),
#        ])

#%%
level1design = pe.Node(fsl.model.Level1Design(),
                  'level1design')
level1design.inputs.bases = {'dgamma':{'derivs': True}}
level1design.inputs.model_serial_correlations = model_serial_correlations
level1design.inputs.interscan_interval = TR


modelspec = pe.Node(modelgen.SpecifyModel(time_repetition=TR, input_units='scans', high_pass_filter_cutoff=hpcutoff),
                   'modelspec')


mask = pe.Node(fsl.maths.ApplyMask(),
              'mask')

feat_model_gen = pe.Node(fsl.model.FEATModel(),
                   name='feat_model_gen')


modelestimate = pe.Node(fsl.FILMGLS(mask_size=film_ms, threshold=film_thr), 
                       name='modelestimate')

modelestimate.inputs.autocorr_noestimate = not model_serial_correlations
if model_serial_correlations: modelestimate.inputs.smooth_autocorr=True

datasink = pe.Node(nio.DataSink(base_directory='/media/Data/work/datasinkFSL'),
                                         name="datasink")

pass_run_data = pe.Node(niu.IdentityInterface(fields = ['mask', 'dof_file', 'copes', 'varcopes']), 'pass_run_data')


join_run_data = pe.JoinNode(
        niu.IdentityInterface(fields=['masks', 'dof_files', 'copes', 'varcopes']),
        joinsource='inputnode',
        joinfield=['masks', 'dof_files', 'copes', 'varcopes'],
        name='join_run_data')



def sort_filmgls_output(copes_grouped_by_run, varcopes_grouped_by_run):
    
    def reshape_lists(files_grouped_by_run):
        import numpy as np
        if not isinstance(files_grouped_by_run, list):
            files = [files_grouped_by_run]
        else:
            files = files_grouped_by_run
            
        if all(len(x) == len(files[0]) for x in files):
            n_files = len(files[0])
        else:
            ('{}DEBUG - files {}'.format('-=-', len(files)))
            print(files)

        all_files = np.array(files).flatten()
        files_grouped_by_contrast = all_files.reshape(int(len(all_files) / n_files), n_files).T.tolist()
        
        return files_grouped_by_contrast
    
    copes_grouped_by_contrast = reshape_lists(copes_grouped_by_run)
    varcopes_grouped_by_contrast = reshape_lists(varcopes_grouped_by_run)
    
#    print('{}DEBUG - copes_grouped_by_contrast {}'.format('==-', len(copes_grouped_by_contrast)))
#    print(copes_grouped_by_contrast)
#    
#    print('{}DEBUG - varcopes_grouped_by_contrast {}'.format('---', len(varcopes_grouped_by_contrast)))
#    print(varcopes_grouped_by_contrast)
    
    return copes_grouped_by_contrast, varcopes_grouped_by_contrast


group_by_contrast = pe.Node(niu.Function(input_names=['copes_grouped_by_run', 'varcopes_grouped_by_run'], output_names=['copes_grouped_by_contrast', 'varcopes_grouped_by_contrast'], function=sort_filmgls_output), name='group_by_contrast')

fixed_fx = create_fixed_effects_flow()


pickfirst = lambda x: x[0]

num_copes = lambda x: len(x)
#%%


#%%

    

level1_workflow = pe.Workflow('testingSomething', base_dir = '/media/Data/work')#work_dir)

level1_workflow.connect([
    ### Build first level model
    (inputnode, datasource, [
        ('subject_id', 'subject_id'),
        ('task', 'task'),
        ]), 
    # (inputnode, model_param_gen, [('subject_id', 'subject_id'),('task','numTask')]),
    #(model_param_gen, datasource, [('condition','condition')]),

    (datasource, modelspec, [('func', 'functional_runs')]),
    (model_param_gen, modelspec, [('model_spec', 'subject_info')]),
    (model_param_gen, level1design, [('contrasts', 'contrasts')]),
    (modelspec, level1design, [('session_info', 'session_info')]),
    (level1design, feat_model_gen, [
        ('fsf_files', 'fsf_file'),
        ('ev_files', 'ev_files')]),
    (datasource, mask, [
        ('mask', 'mask_file'),
        ('func', 'in_file')]),
    
    ### Prep functional data
    (mask, modelestimate, [('out_file', 'in_file')]),
    
    ### Estimate model
    (feat_model_gen, modelestimate, [
        ('design_file', 'design_file'),
        ('con_file', 'tcon_file'),
        ('fcon_file', 'fcon_file')]),

    (datasource, pass_run_data, [('mask', 'mask')]),
    (modelestimate, pass_run_data, [
        ('copes', 'copes'),
        ('varcopes', 'varcopes'),
        ('dof_file', 'dof_file'),
    ]),
    
    ### Aggregate run across runs
    (pass_run_data, join_run_data, [
        ('mask', 'masks'),
        ('dof_file', 'dof_files'),
        ('copes', 'copes'),
        ('varcopes', 'varcopes'),
        
    ]),
    
    (join_run_data, group_by_contrast, [
        ('copes', 'copes_grouped_by_run'),
        ('varcopes', 'varcopes_grouped_by_run')
    ]),

    ### Write out model files
    (feat_model_gen, datasink, [('design_file', 'design.@design_matrix')]),
#     (mask, datasink, [('out_file', 'input.@masked_functional')]),
    (modelestimate, datasink, [
        ('zstats', 'film.@zstats'),
        ('copes', 'film.@copes'),
        ('varcopes', 'film.@varcopes'),
        ('param_estimates', 'film.@parameter_estimates'),
        ('dof_file', 'film.@dof'),
    ]),
     
    ### Fixed Effects Model
    (group_by_contrast, fixed_fx, [
        ('copes_grouped_by_contrast', 'inputspec.copes'),
        ('varcopes_grouped_by_contrast', 'inputspec.varcopes'),
    ]),
    
    (join_run_data, fixed_fx, [
        (('masks', pickfirst), 'flameo.mask_file'),
        ('dof_files', 'inputspec.dof_files'),
        (('copes', num_copes), 'l2model.num_copes'), ### number of runs
    ]),
    
    ### Write out fixed effects results
    (fixed_fx, datasink, [
        ('outputspec.res4d', 'flameo.@res4d'), 
        ('outputspec.copes', 'flameo.@copes'), 
        ('outputspec.varcopes', 'flameo.@varcopes'),
        ('outputspec.zstats', 'flameo.@zstats'), 
        ('outputspec.tstats', 'flameo.@tstats')
        ]),
])
    
    
    
level1_workflow.write_graph(graph2use='colored', format='png', simple_form=True)
level1_workflow.write_graph(graph2use='orig', dotfilename=opj(level1_workflow.base_dir, 'l1', 'graph_orig.dot'), format='png', simple_form=True)
level1_workflow.write_graph(graph2use='flat', dotfilename=opj(level1_workflow.base_dir, 'l1', 'graph_flat.dot'), format='png', simple_form=True)
level1_workflow.write_graph(graph2use='exec', dotfilename=opj(level1_workflow.base_dir, 'l1', 'graph_exec.dot'), format='png', simple_form=True)

Image(filename=opj(level1_workflow.base_dir, 'l1', 'graph_exec.png'))


#level1_workflow.run('MultiProc', plugin_args={'n_procs': 4})

#%%

subjectList = ['1063']# , '1072', '1206', '1244','1273' ,'1291', '1305', '1340', '1345', '1346'] 
subjectNode = pe.Node(niu.IdentityInterface(fields=[ 'subject_id', 'task'],
                                          mandatory_inputs=True),
                    'subjectNode')
subjectNode.iterables = [('subject_id', subjectList),('task',task)]

l1Combine = pe.Workflow('emptyEV', base_dir = '/media/Data/work') #'subjectFSLAddContrasts'

l1Combine.connect([
        (subjectNode, model_param_gen, [('subject_id','subject_id'),('task','taskNum')]),
        (subjectNode, level1_workflow, [('subject_id','inputnode.subject_id')]),
        (model_param_gen, level1_workflow,[('condition','inputnode.condition')]),
        ])
    
l1Combine.write_graph(graph2use='colored', format='png', simple_form=True)
Image(filename=opj(l1Combine.base_dir, 'l1', 'graph_exec.png'))

l1Combine.run('MultiProc', plugin_args={'n_procs': 4})
