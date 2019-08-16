#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:57:41 2019

@author: Or Duek
Writing first & second level analysis for KPE study
Based on poldrak's lab - paper publishe in BioRxiv: https://www.biorxiv.org/content/10.1101/694364v1

"""
#%% Load packages
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
from nipype.interfaces.io import BIDSDataGrabber
from niworkflows.interfaces.bids import DerivativesDataSink# as BIDSDerivatives


import os
#%% data info
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
data_dir = os.path.abspath('/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep')
output_dir = '/media/Data/work/kpeTask'
work_dir = '/home/or/work/FSL_KPE'
fwhm = 6
tr = 1
#%% might not need to run
#from bids import BIDSLayout
#layout = BIDSLayout('/media/Data/kpe_forFmriPrep', derivatives=True)
#layout.get_subjects()
#layout.get_modalities()

#layout.get_tasks()
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


infosource = pe.Node(niu.IdentityInterface(fields=['subject_id'
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
runinfo = pe.Node(niu.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo')

# Set the column names to be used from the confounds file
runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
    ['a_comp_cor_%02d' % i for i in range(6)] + ['cosine%02d' % i for i in range(4)]



# SUSAN smoothing
susan = create_susan_smooth()
susan.inputs.inputnode.fwhm = fwhm

workflow = pe.Workflow(name='firstLevelKPE',base_dir=work_dir)


#workflow.run('MultiProc', plugin_args={'n_procs': 3}) # doesn't work on local - not enough memory

#workflow.run() # on local




#%%
l1_spec = pe.Node(SpecifyModel(
    parameter_source='FSL',
    input_units='secs',
    high_pass_filter_cutoff=100,
    time_repetition = tr,
), name='l1_spec')

# l1_model creates a first-level model design
l1_model = pe.Node(fsl.Level1Design(
    bases={'dgamma': {'derivs': True}},
    model_serial_correlations=True,
    interscan_interval = tr,
    contrasts=[('Trauma-Sad', 'T', ['trauma', 'sad'], [1, -1]), 
               ('Trauma-Relax', 'T', ['trauma','relax'], [1,-1]),
               ('Sad-Relax', 'T', ['sad','relax'], [1,-1])],
    # orthogonalization=orthogonality,
), name='l1_model')

# feat_spec generates an fsf model specification file
feat_spec = pe.Node(fsl.FEATModel(), name='feat_spec')
# feat_fit actually runs FEAT
feat_fit = pe.Node(fsl.FEAT(), name='feat_fit') #, mem_gb=12)

feat_select = pe.Node(nio.SelectFiles({
    'cope': 'stats/cope*.nii.gz',
    'pe': 'stats/pe[0-9][0-9].nii.gz',
    'tstat': 'stats/tstat*.nii.gz',
    'varcope': 'stats/varcope*.nii.gz',
    'zstat': 'stats/zstat*.nii.gz',
}), name='feat_select')

ds_cope = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='cope',
    desc='intask'), name='ds_cope', run_without_submitting=True)

ds_varcope = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='varcope',
    desc='intask'), name='ds_varcope', run_without_submitting=True)

ds_zstat = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='zstat',
    desc='intask'), name='ds_zstat', run_without_submitting=True)

ds_tstat = pe.Node(DerivativesDataSink(
    base_directory=str(output_dir), keep_dtype=False, suffix='tstat',
    desc='intask'), name='ds_tstat', run_without_submitting=True)

workflow.connect([
    (infosource, selectfiles, [('subject_id', 'subject_id')]),
    (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
    (selectfiles, susan, [('func', 'inputnode.in_files'), ('mask','inputnode.mask_file')]),
    (susan, runinfo, [('outputnode.smoothed_files', 'in_file')]),
    (susan, l1_spec, [('outputnode.smoothed_files', 'functional_runs')]),
    (selectfiles, ds_cope, [('func', 'source_file')]),
    (selectfiles, ds_varcope, [('func', 'source_file')]),
    (selectfiles, ds_zstat, [('func', 'source_file')]),
    (selectfiles, ds_tstat, [('func', 'source_file')]),
   
    (runinfo, l1_spec, [
        ('info', 'subject_info'),
        ('realign_file', 'realignment_parameters')]),
    (l1_spec, l1_model, [('session_info', 'session_info')]),
    (l1_model, feat_spec, [
        ('fsf_files', 'fsf_file'),
        ('ev_files', 'ev_files')]),
    (l1_model, feat_fit, [('fsf_files', 'fsf_file')]),
    (feat_fit, feat_select, [('feat_dir', 'base_directory')]),
    (feat_select, ds_cope, [('cope', 'in_file')]),
    (feat_select, ds_varcope, [('varcope', 'in_file')]),
    (feat_select, ds_zstat, [('zstat', 'in_file')]),
    (feat_select, ds_tstat, [('tstat', 'in_file')]),
])


#%%  Plot the workflow
#workflow.write_graph("workflow_graph.dot", graph2use='colored', format='png', simple_form=True)
#from IPython.display import Image
#Image(filename='/media/Data/work/firstLevelKPE/workflow_graph.png')
#workflow.write_graph(graph2use='flat')

#%% Run workflow
workflow.run('MultiProc', plugin_args={'n_procs': 2}) # doesn't work on local - not enough memory

#workflow.run() # on local
#%% Now run second level
cope_list = ['1','2','3']

workflow2nd = pe.Workflow(name="2nd_level", base_dir=work_dir)

copeInput = pe.Node(niu.IdentityInterface(
        fields = ['cope']),
        name = 'copeInput')
        
copeInput.iterables= [('cope', cope_list)]


#inputnode = pe.Node(niu.IdentityInterface(
#    fields=['group_mask', 'in_copes', 'in_varcopes']),
#    name='inputnode')

#num_copes = 3

# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'in_copes': '/media/Data/work/firstLevelKPE/_subject_id_*/feat_fit/run0.feat/stats/cope{cope}.nii.gz',
             'mask': '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-*/ses-1/func/sub-*_ses-1_task-Memory_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
             'in_varcopes': '/media/Data/work/firstLevelKPE/_subject_id_*/feat_fit/run0.feat/stats/varcope{cope}.nii.gz',
             }
selectCopes = pe.Node(nio.SelectFiles(templates,
                               base_directory=data_dir),
                   name="selectCopes")

#%%

copemerge    = pe.Node(interface=fsl.Merge(dimension='t'),
                          name="copemerge")

varcopemerge = pe.Node(interface=fsl.Merge(dimension='t'),
                       name="varcopemerge")

maskemerge = pe.Node(interface=fsl.Merge(dimension='t'),
                       name="maskemerge")
copeImages = glob.glob('/media/Data/work/firstLevelKPE/_subject_id_*/feat_fit/run0.feat/stats/cope1.nii.gz')
copemerge.inputs.in_files = copeImages



# Configure FSL 2nd level analysis
l2_model = pe.Node(fsl.L2Model(), name='l2_model')

flameo_ols = pe.Node(fsl.FLAMEO(run_mode='ols'), name='flameo_ols')
def _len(inlist):
    print (len(inlist))
    return len(inlist)
### use randomize
rand = pe.Node(fsl.Randomise(),
                            name = "randomize") 


rand.inputs.mask = '/media/Data/work/KPE_SPM/fslRandomize/group_mask.nii.gz' # group mask file (was created earlier)
rand.inputs.one_sample_group_mean = True
rand.inputs.tfce = True
rand.inputs.vox_p_values = True
rand.inputs.num_perm = 200
# Thresholding - FDR ################################################
# Calculate pvalues with ztop
fdr_ztop = pe.Node(fsl.ImageMaths(op_string='-ztop', suffix='_pval'),
                   name='fdr_ztop')
# Find FDR threshold: fdr -i zstat1_pval -m <group_mask> -q 0.05
# fdr_th = <write Nipype interface for fdr>
# Apply threshold:
# fslmaths zstat1_pval -mul -1 -add 1 -thr <fdr_th> -mas <group_mask> \
#     zstat1_thresh_vox_fdr_pstat1

# Thresholding - FWE ################################################
# smoothest -r %s -d %i -m %s
# ptoz 0.05 -g %f
# fslmaths %s -thr %s zstat1_thresh

# Thresholding - Cluster ############################################
# cluster -i %s -c %s -t 3.2 -p 0.05 -d %s --volume=%s  \
#     --othresh=thresh_cluster_fwe_zstat1 --connectivity=26 --mm

workflow2nd.connect([
    (copeInput, selectCopes, [('cope', 'cope')]),
    (selectCopes, copemerge, [('in_copes','in_files')]),
    (selectCopes, varcopemerge, [('in_varcopes','in_files')]),
    (selectCopes, maskemerge, [('mask','in_files')]),
    (selectCopes, l2_model, [(('in_copes', _len), 'num_copes')]),
    (copemerge, flameo_ols, [('merged_file', 'cope_file')]),
    (varcopemerge, flameo_ols, [('merged_file', 'var_cope_file')]),
    (maskemerge, flameo_ols, [('merged_file', 'mask_file')]),
    (l2_model, flameo_ols, [('design_mat', 'design_file'),
                            ('design_con', 't_con_file'),
                            ('design_grp', 'cov_split_file')]),
    (copemerge, rand, [('merged_file','in_file')]),
  #  (maskemerge, rand, [('merged_file','mask')]),
    (l2_model, rand, [('design_con','tcon'), ('design_mat','design_mat')]),
    (maskemerge, fdr_ztop, [('merged_file','mask_file')]),
    (flameo_ols, fdr_ztop, [('zstats','in_file')]),
])
#%%
workflow2nd.run()

#%% plot results
%matplotlib qt
from nilearn.plotting import plot_glass_brain
import nilearn.plotting
import glob
conImages = glob.glob('/media/Data/work/KPE_SPM/Sink/2ndLevel/_contrast_id_con_000*/spmT_0001.nii')
fig = plot_glass_brain('/home/or/work/FSL_KPE/2nd_level/_cope_1/randomize/randomise_tstat1.nii.gz',colorbar=True,
     threshold=3, display_mode='lyrz', black_bg=True)#, vmax=10);   
#%%
fig = nilearn.plotting.plot_stat_map('/home/or/work/FSL_KPE/2nd_level/_cope_3/randomize/randomise_tstat1.nii.gz', alpha=0.5 )#, cut_coords=(0, 45, -7))
fig.add_contours('/home/or/work/FSL_KPE/2nd_level/_cope_3/randomize/randomise_tfce_corrp_tstat1.nii.gz', levels=[0.95], colors='b')