#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:31:12 2019

@author: or
"""


#from __future__ import print_function
#from builtins import str
#from builtins import range

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

from nipype.algorithms.misc import Gunzip
from nipype import Node, Workflow, MapNode
from nipype import SelectFiles
from os.path import join as opj

os.chdir('/home/or/Documents/dicom_niix')
import readConditionFiles_r_aPTSD


from nipype.interfaces.matlab import MatlabCommand
#mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
MatlabCommand.set_default_paths('/home/or/Downloads/spm12/') # set default SPM12 path in my computer. 


# Specify the location of the data.
data_dir = os.path.abspath('/media/Data/FromHPC/output/fmriprep')
from bids.grabbids import BIDSLayout

layout = BIDSLayout(data_dir)
checkGet  = layout.get(type="bold", extensions="nii.gz")
checkGet[0].subject
layout.get(type="bold", task="3", session="1", extensions="nii.gz")[0].filename

# Specify the subject directories
subject_list = ['1063' , '1072', '1206', '1244','1273' ,'1291', '1305', '1340', '1345', '1346']
# Map field names to individual subject runs.
task_list = ['3','4','5','6']

infosource = Node(util.IdentityInterface(fields=['subject_id'
                                            ],
                                    ),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]

## SelectFiles - to grab the data (alternativ to DataGrabber)
#templates = {'func': '/media/Data/FromHPC/output/fmriprep/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'}
#selectfiles = Node(SelectFiles(templates,
#                               base_directory=data_dir),
#                   name="selectfiles")
#
info = dict(
    func = [['subject_id', ['3', '4', '5', '6']]],
    
    )
# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'func': '/media/Data/FromHPC/output/fmriprep/sub-%s/ses-1/func/sub-*_ses-1_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'}
#selectfiles = Node(SelectFiles(templates,
#                               base_directory=data_dir),
#                   name="selectfiles")

datasource = pe.Node(
    interface=nio.DataGrabber(
        infields=['subject_id'], outfields=['func']),
    name='datasource')
datasource.inputs.base_directory = data_dir
datasource.inputs.template = '/media/Data/FromHPC/output/fmriprep/sub-%s/ses-1/func/sub-*_ses-1_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True
# add unzip node
# Gunzip - unzip functional
gunzip = MapNode(Gunzip(), name='gunzip',
                 iterfield=['in_file'])

#os.chdir('/media/Data/work')



###########################
def subjectinfo(subject_id):
   
    from readConditionFiles_r_aPTSD import loadmat, _check_keys, _todict, readConditions
    import scipy.io as spio
    import os,json,glob,sys
    ###############################################################################################
    # Define experiment things (data_dir = filder where data files are present. )
    data_dir ='/media/Data/FromHPC/output/fmriprep'
    from bids.grabbids import BIDSLayout
    layout = BIDSLayout(data_dir)
    tasks = ['3','4','5','6'] # number of task (i.e. block. corresponding to file name)
    source_epi = layout.get(type="bold", session="1", extensions="nii.gz", subject = subject_id)
    ###############################################################################################
    def organizeBlocks(subNum):
        
    # Read both mat files (first timestamp)
    # check first block of each day. 
    # check thrird of each day
    # sort
        orderArray = []
        matFileLoss = '/media/Drobo/Levy_Lab/Projects/R_A_PTSD_Imaging/Data/Behavior data/Behavior_fitpar/Behavior data fitpar_091318/RA_LOSS_%s_fitpar.mat'%subNum
        matFileGain = '/media/Drobo/Levy_Lab/Projects/R_A_PTSD_Imaging/Data/Behavior data/Behavior_fitpar/Behavior data fitpar_091318/RA_GAINS_%s_fitpar.mat'%subNum
        metaDataLoss = loadmat(matFileLoss)
        metaDataGain = loadmat(matFileGain)
        a= {'3rdLoss':list(vars(metaDataLoss['Data']['trialTime'][62])['trialStartTime']), '1stLoss':list(vars(metaDataLoss['Data']['trialTime'][0])['trialStartTime']), '1stGain':list(vars(metaDataGain['Data']['trialTime'][0])['trialStartTime']), '3rdGain':list(vars(metaDataGain['Data']['trialTime'][62])['trialStartTime'])}
        s = [(k, a[k]) for k in sorted(a, key=a.get, reverse=False)]
        for k, v in s:
             print (k, v)
             orderArray.append(k)
        totalEvent = []
        for n in orderArray:
            print (n)
            if n=='1stLoss':
                # run loss mat file on redConcitions function on first two blocks (i.e. 0, 31)
                print (n)
                for x in [0,31]:
                    event = readConditions(matFileLoss, x)
                    event['condition'] = 'Loss'
                    totalEvent.append(event)
            elif n=='1stGain':
                # run Gain mat file on reCondition function
                print (n)
                for x in [0,31]:
                    event = readConditions(matFileGain, x)
                    event['condition'] = 'Gain'
                    totalEvent.append(event)
            elif n=='3rdLoss':
                # run loss from 3rd block (i.e. 62, 93)
                print (n)
                for x in [62, 93]:
                    event = readConditions(matFileLoss, x)
                    event['condition'] = 'Loss'
                    totalEvent.append(event)
            elif n=='3rdGain':
                # run gains from 3rd block
                print (n)
                for x in [62, 93]:
                    event = readConditions(matFileGain, x)
                    event['condition'] = 'Gain'
                    totalEvent.append(event)
            else:
                print ('The condition ' + n + ' is not clear.')
            
            # the end result is an array of data sets per each run (i.e. block) - called totalEvent
        return totalEvent
    
    
    # creates full table of subject info (conditions, runs etc.)
   # onsets = ([])
    import pandas as pd
    eventsTotal = organizeBlocks(subject_id)
    for i in range(len(eventsTotal)):
        print (i)
        eventsTotal[i]['condName'] = 'test'
        for n in range(1,len(eventsTotal[i])+1):
            if eventsTotal[i].condition[n] =='Gain':
                if eventsTotal[i].trial_type[n] == 'risk':
                    eventsTotal[i]['condName'][n] = 'GainRisk'
                else:
                    eventsTotal[i]['condName'][n] = 'GainAmb'
            if eventsTotal[i].condition[n] == 'Loss':
                if eventsTotal[i].trial_type[n] == 'risk':
                    eventsTotal[i]['condName'][n] = 'LossRisk'
                else:
                    eventsTotal[i]['condName'][n] = 'LossAmb'
    #events = eventsTotal[0]# the first ses
    
    from nipype.interfaces.base import Bunch
    #from copy import deepcopy
    print("Subject ID: %s\n" % str(subject_id))
    output = []
    contrasts = []
    order = [] # specify order of conditions for the contrasts
    #names = ['GainRisk', 'GainAmb','LossRisk', 'LossAmb']
    for r in range(4): # from 1-4
        
        print (r)
        confounds = pd.read_csv(os.path.join(data_dir, 
                                        "sub-%s"%subject_id, "ses-%s"%source_epi[r].session, "func", 
                                        "sub-%s_ses-%s_task-%s_desc-confounds_regressors.tsv"%(source_epi[r].subject, source_epi[r].session, tasks[r])),
                                           sep="\t", na_values="n/a")
        # put here ifs to build bunch for each run according to the conditions. 
        if eventsTotal[r].condition[r+1]=='Loss':
            print ('LOSS')
           
            output.insert(r,
                          Bunch(conditions = ['LossRisk','LossAmb'],
                                onsets = [
                                         list(eventsTotal[r][eventsTotal[r].condName=='LossRisk'].onset),
                                         list(eventsTotal[r][eventsTotal[r].condName=='LossAmb'].onset)],
                                durations = [
                                             list(eventsTotal[r][eventsTotal[r].condName=='LossRisk'].duration),
                                             list(eventsTotal[r][eventsTotal[r].condName=='LossAmb'].duration)],
                                             regressors=[list(confounds.framewise_displacement.fillna(0)),
                                                         list(confounds.a_comp_cor_00),
                                                         list(confounds.a_comp_cor_01),
                                                         list(confounds.a_comp_cor_02),
                                                         list(confounds.a_comp_cor_03),
                                                         list(confounds.a_comp_cor_04),
                                                         list(confounds.a_comp_cor_05),
                                                         ],
                             regressor_names=['FramewiseDisplacement',
                                              'aCompCor0',
                                              'aCompCor1',
                                              'aCompCor2',
                                              'aCompCor3',
                                              'aCompCor4',
                                              'aCompCor5'],
                         
                      
                               
                                                                  ) )
            order.append('Loss')
            
        elif eventsTotal[r].condition[r+1]=='Gain':
            print ('Gain')
            
            output.insert(r,
                          Bunch(conditions = ['GainRisk','GainAmb'],
                                onsets = [list(eventsTotal[r][eventsTotal[r].condName=='GainRisk'].onset),
                                         list(eventsTotal[r][eventsTotal[r].condName=='GainAmb'].onset)],
                                durations = [list(eventsTotal[r][eventsTotal[r].condName=='GainRisk'].duration),
                                             list(eventsTotal[r][eventsTotal[r].condName=='GainAmb'].duration),
                                             ],
                                             regressors=[list(confounds.framewise_displacement.fillna(0)),
                         list(confounds.a_comp_cor_00),
                         list(confounds.a_comp_cor_01),
                         list(confounds.a_comp_cor_02),
                         list(confounds.a_comp_cor_03),
                         list(confounds.a_comp_cor_04),
                         list(confounds.a_comp_cor_05),
                        ],
             regressor_names=['FramewiseDisplacement',
                              'aCompCor0',
                              'aCompCor1',
                              'aCompCor2',
                              'aCompCor3',
                              'aCompCor4',
                              'aCompCor5'],
                        
             
                              )
                               
                            
                                )
            order.append('Gain')
    # creating contrasts for each subject
    condition_names = ['GainRisk', 'GainAmb' ,'LossRisk', 'LossAmb']  
    if order[0] == 'Gain':
        GainRisk_cond = ['GainRisk','T', condition_names ,[1,0,0,0],[1,1,0,0]] # set this contrast only to the first two runs
        GainAmb_cond = ['GainAmb','T', condition_names ,[0,1,0,0],[1,1,0,0]]
        LossRisk_cond = ['LossRisk','T', condition_names,[0,0,1,0],[0,0,1,1]]
        LossAmb_cond = ['LossAmb','T',condition_names,[0,0,0,1],[0,0,1,1]]
        negGainRisk_cond = ['GainRisk','T', condition_names ,[-1,0,0,0],[1,1,0,0]]
    else:
        LossRisk_cond = ['LossRisk','T', condition_names ,[0,0,1,0],[1,1,0,0]]
        LossAmb_cond = ['LossAmb','T', condition_names ,[0,0,0,1],[1,1,0,0]]
        GainRisk_cond = ['GainRisk','T', condition_names,[1,0,0,0],[0,0,1,1]]
        GainAmb_cond = ['GainAmb','T',condition_names,[0,1,0,0],[0,0,1,1]]
        negGainRisk_cond = ['GainRisk','T', condition_names ,[-1,0,0,0],[0,0,1,1]]
                
            
                            
    
  #  Gain_vs_Loss = ["Gain vs. Loss",'T', condition_names ,[0.5, 0.5, -0.5, -0.5]]
    Risk_vs_Amb = ["Risk vs. Amb",'T', condition_names ,[0.5, -0.5, 0.5, -0.5]]
   
    
    gain_total = ["Gain", 'F', [GainRisk_cond, GainAmb_cond]]
    loss_total = ["Loss", 'F', [LossAmb_cond, LossRisk_cond]]
    contrasts=[GainRisk_cond, GainAmb_cond, LossRisk_cond, LossAmb_cond, Risk_vs_Amb , negGainRisk_cond, gain_total, loss_total]
    return (output, contrasts)


#########################################
#Adding subjectInfo function as a node

# Get Subject Info - get subject specific condition information
getsubjectinfo = Node(util.Function(input_names=['subject_id'],
                               output_names=['subject_info','contrasts'],
                               function=subjectinfo),
                        
                      name='getsubjectinfo')


################################################################


modelspec = Node(interface=model.SpecifySPMModel(), name="modelspec") 
modelspec.inputs.concatenate_runs = False
modelspec.inputs.input_units = 'scans'
modelspec.inputs.output_units = 'scans'
#modelspec.inputs.outlier_files = '/media/Data/R_A_PTSD/preproccess_data/sub-1063_ses-01_task-3_bold_outliers.txt'
modelspec.inputs.time_repetition = 1.  # make sure its with a dot 
modelspec.inputs.high_pass_filter_cutoff = 128.

################################################
#modelspec.inputs.subject_info = subjectinfo(subject_id) # run per subject
modelspec.base_dir = '/media/Data/work'

level1design = pe.Node(interface=spm.Level1Design(), name="level1design") #, base_dir = '/media/Data/work')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = 1.
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'

#######################################################################################################################
# Initiation of a workflow
wfSPM = Workflow(name="l1spm", base_dir="/media/Data/work")
wfSPM.connect([
        (infosource, datasource, [('subject_id','subject_id')]),
        (datasource, gunzip, [('func','in_file')]),
        (gunzip, modelspec, [('out_file', 'functional_runs')]),   
        (infosource, getsubjectinfo, [('subject_id', 'subject_id')]),
        (getsubjectinfo,modelspec, [('subject_info', 'subject_info')]),
        
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



########################################################################
#%% Connecting level1 estimation and contrasts
wfSPM.connect([
         (level1design, level1estimate, [('spm_mat_file','spm_mat_file')]),
         (getsubjectinfo, contrastestimate, [('contrasts' ,'contrasts')]),
         (level1estimate, contrastestimate,
            [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'),
            ('residual_image', 'residual_image')]),
    ])



###############################################################

#%% Adding data sink
########################################################################
# Datasink
datasink = Node(nio.DataSink(base_directory='/media/Data/work/datasink'),
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
#%% Creating graphs         
                          
################################################################################
# Graph of workflow
wfSPM.write_graph("workflow_graph.dot", graph2use='colored', format='png', simple_form=True)
from IPython.display import Image
Image(filename="/media/Data/work/l1spm/workflow_graph.png")
wfSPM.write_graph(graph2use='flat')


wfSPM.write_graph(graph2use='flat')
Image(filename = '/media/Data/work/l1spm/graph_detailed.png')
wfSPM.run('MultiProc', plugin_args={'n_procs': 6})                          
                    

#%% Single subject (1st level) T test graphs                        
#############################################################################
#Some T contrast plots:
from nilearn.plotting import plot_stat_map
anatimg = '/media/Data/FromHPC/output/fmriprep/sub-1072/anat/sub-1072_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
plot_stat_map(
    '/media/Data/work/datasink/2ndLevel/_contrast_id_con_0001/spmT_0001_thr.nii', title='LossRisk - fwhm=6',
    bg_img=anatimg, threshold=3, display_mode='x', cut_coords=(-5, 0, 5, 10, 15), dim=0);
        
plot_stat_map(
    '/media/Data/work/datasink/1stLevel/_subject_id_1072/spmT_0003.nii', title='Risk Gain - fwhm=6',
    bg_img=anatimg, threshold=3, display_mode='x', cut_coords=(-5, 0, 5, 10, 15), dim=0);


plot_stat_map(
    '/media/Data/work/datasink/1stLevel/_subject_id_1072/spmT_0005.nii', title='Risk vs. Amb - fwhm=6',
    bg_img=anatimg, threshold=3, display_mode='x', cut_coords=(-5, 0, 5, 10, 15), dim=0);

plot_stat_map(
    '/media/Data/work/datasink/1stLevel/_subject_id_1072/spmT_0006.nii', title='Gain vs. Loss - fwhm=6',
    bg_img=anatimg, threshold=3, display_mode='y', cut_coords=(-5, 0, 5, 10, 15), dim=0);

plot_stat_map(
    '/media/Data/work/datasink/1stLevel/_subject_id_1206/spmT_0006.nii', title='Gain vs. Loss - fwhm=6',
    bg_img=anatimg, threshold=3, display_mode='x', cut_coords=(-5, 0, 5, 10, 15), dim=0);

plot_stat_map(
    '/media/Data/work/datasink/1stLevel/_subject_id_1206/spmT_0002.nii', title='Loss Risk- fwhm=6',
    bg_img=anatimg, threshold=3, display_mode='x', cut_coords=(-5, 0, 5, 10, 15), dim=0);
########################################################################
# Create 2nd level analysis
#%% Gourp analysis
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
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005', 'con_0006','ess_0007', 'ess_0008']

# Threshold - thresholds contrasts
level2thresh = Node(spm.Threshold(contrast_index=1,
                              use_topo_fdr=False,
                              use_fwe_correction=False,
                              extent_threshold=10,
                              height_threshold= 0.005,
                              #extent_fdr_p_threshold = 0.1,
                              height_threshold_type='p-value'),
                              
                                   name="level2thresh")
 #Infosource - a function free node to iterate over the list of subject names
infosource = Node(util.IdentityInterface(fields=['contrast_id']),
                  name="infosource")
infosource.iterables = [('contrast_id', contrast_list)]

# SelectFiles - to grab the data (alternative to DataGrabber)
templates = {'cons': opj('/media/Data/work/datasink/1stLevel/_sub*/', 
                         '{contrast_id}.nii')}
selectfiles = Node(SelectFiles(templates,
                               base_directory='/media/Data/work',
                               sort_filelist=True),
                   name="selectfiles")



l2analysis = Workflow(name='spm_l2analysisWorking')
l2analysis.base_dir = opj(data_dir, '/media/Data/work/')

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
                                                       
l2analysis.run('MultiProc', plugin_args={'n_procs': 4})

#########################################################################
# look at results:
#%% Group analysis graphs
anatimg = '/home/or/Downloads/spm12/canonical/avg152T1.nii'
plot_stat_map(
    '/media/Data/work/datasink/2ndLevel/_contrast_id_con_0001/con_0001.nii', title='Gain vs Loss', dim=1,
    bg_img=anatimg, threshold=3.6, vmax=8, display_mode='x', cut_coords=(-45, -30, -15, 0, 15), cmap='viridis');



plot_stat_map(
    '/media/Data/work/spm_l2analysis/_contrast_id_con_0001/level2conestimate/con_0001.nii', title='Risk Vs. Baseline', dim=1,
    bg_img=anatimg, threshold=3.6, vmax=8, display_mode='x', cut_coords=(-45, -30, -15, 0, 15), cmap='viridis');

from nilearn.plotting import plot_glass_brain
plot_glass_brain(nilearn.image.smooth_img('/media/Data/work/datasink/2ndLevel/_contrast_id_con_0006/con_0001.nii', 3),colorbar=True,
     threshold=2.6, display_mode='lyrz', black_bg=True)#, vmax=10);       
        
        
plot_stat_map(nilearn.image.smooth_img(
    '/media/Data/work/datasink/2ndLevel/_contrast_id_con_0006/con_0001.nii',6),
    threshold=2, display_mode='ortho');    