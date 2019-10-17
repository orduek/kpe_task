#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:44:56 2019

@author: Or Duek
Run seed based analysis
"""
import nilearn
amygdala_coords = [(31,4,-22)] #(-26, 2, -18)] #, (31,4,-22)]
hippocampus_coors = [(24,-28,-10)]
# set the coords to run
coords = amygdala_coords#hippocampus_coors # choose coords
#%% method to create seed based conenctivity
def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    import numpy as np
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['csf', 'white_matter', 'framewise_displacement',
                          'a_comp_cor_00', 'a_comp_cor_01',	'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']] # can add 'global_signal' also
     # change NaN of FD to zero
    finalConf = np.array(finalConf)
    finalConf[0,2] = 0 # if removing FD than should remove this one also
    return finalConf

def stratifyTimeseries (events_file, subject_seed_timeseries, subject_brain_timeseries, subject_id, scriptName):
    #trial_line is a parameter - if 0 then will create each line as file. If 1 then each task
    # grab subject events file
    
    events = pd.read_csv(events_file, sep=r'\s+')
  #  timeSeries = np.array(np.load(subject_timeseries, allow_pickle = True))
    # create a subject folder
    try:
        # check if already there
        os.makedirs('subject_%s' %(subject_id))
    except:
        print ("Dir already preset")
    
    # read line  by line and create matrix per line
    timeOnset = []
    timeDuration = []
    for line in events.iterrows():
        print (line)
        if line[1]['trial_type'].find(scriptName)!= -1:
            timeOnset.append(round(line[1].onset))
            timeDuration.append(round(line[1].duration))
        else:
            print('not trauma')
        
    spec_seedTimeline = np.concatenate([subject_seed_timeseries[timeOnset[0]:timeOnset[0] + timeDuration[0],:], subject_seed_timeseries[timeOnset[1]:timeOnset[1] + timeDuration[1],:], subject_seed_timeseries[timeOnset[2]:timeOnset[2] + timeDuration[2],:]]) 
    spec_brainTimeline = np.concatenate([subject_brain_timeseries[timeOnset[0]:timeOnset[0] + timeDuration[0],:], subject_brain_timeseries[timeOnset[1]:timeOnset[1]+ timeDuration[1],:], subject_brain_timeseries[timeOnset[2]:timeOnset[2]+timeDuration[2],:]])
          #  np.save('subject_%s/speficTrial_%s_%s' %(subject_id,numberRow, trial_type), specTimeline)
    # or read by trial type and create matrix per trial type
    return spec_seedTimeline , spec_brainTimeline
  
        
def createSeedVoxelSeries(coords, func_filename, confound_filename, mask_file, subject):
    from nilearn import input_data
    seed_masker = input_data.NiftiSpheresMasker(
        coords, radius=6,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=1.,
        memory="/media/Data/nilearn", memory_level=1, verbose=2)
    
    brain_masker = input_data.NiftiMasker(mask_img = mask_file,
        smoothing_fwhm=6,
        detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=1.,
        memory='/media/Data/nilearn', memory_level=1, verbose=2)

    seed_time_series = seed_masker.fit_transform(func_filename,
                                                 confounds=removeVars(confound_filename))
    
    brain_time_series = brain_masker.fit_transform(func_filename,
                                                   confounds=removeVars(confound_filename))
    
    
    return seed_time_series, brain_time_series, brain_masker
    

# stratify to events
def seedVoxelCor(spec_seed_timeseries, spec_brain_timeseries, scriptName, subject, brain_masker, session, seedName):    
    import numpy as np
    
    seed_to_voxel_correlations = (np.dot(spec_brain_timeseries.T, spec_seed_timeseries) /
                                  spec_seed_timeseries.shape[0]
                                  )

    
    from nilearn import plotting
    
    seed_to_voxel_correlations_img = brain_masker.inverse_transform(
        seed_to_voxel_correlations.T)


    seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
    print("Seed-to-voxel correlation Fisher-z transformed: min = %.3f; max = %.3f"
          % (seed_to_voxel_correlations_fisher_z.min(),
             seed_to_voxel_correlations_fisher_z.max()
             )
          )

    # Finally, we can tranform the correlation array back to a Nifti image
    # object, that we can save.
    seed_to_voxel_correlations_fisher_z_img = brain_masker.inverse_transform(
        seed_to_voxel_correlations_fisher_z.T)
    seed_to_voxel_correlations_fisher_z_img.to_filename(
        '/home/or/kpe_conn/%s_seed_%s_sub-%s_ses-%s_z.nii.gz' %(scriptName,seedName,subject, session))
    
    return seed_to_voxel_correlations, seed_to_voxel_correlations_fisher_z
# this function returns the regular correlation matrix, the fishr-z transformation of correlation matrix and the brain masker (for inverse transform the brain back to nifti image)
#%%
subject_list =['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464', '1468','1480','1499'] 
def fileList(subjects, session):
    func_files = ['/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-%s/ses-%s/func/sub-%s_ses-%s_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' % (sub,session,sub,session) for sub in subjects]
    return func_files

def maskList(subjects, session):
    mask_files = ['/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-%s/ses-%s/func/sub-%s_ses-%s_task-Memory_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz' % (sub,session,sub,session) for sub in subjects]
    return mask_files


def confList(subjects, session):
    confound_files = ['/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-%s/ses-%s/func/sub-%s_ses-%s_task-Memory_desc-confounds_regressors.tsv' % (sub,session,sub,session) for sub in subjects]
    return confound_files

def eventfList(subjects, session):
    confound_files = ['/media/Data/PTSD_KPE/condition_files/sub-%s_ses-%s.csv' % (sub,session) for sub in subjects]
    return confound_files

#%% run seed based on coords for all subjects
#######################################################
session = '1' # set the session number
script = 'trauma' # set the specific script
nameOfSeed = 'amg_right' # set name of seed

# now run the following
import os

func_files = fileList(subject_list, session)
conf_files = confList(subject_list, session)
mask_files = maskList(subject_list, session)
event_files = eventfList(subject_list, session)
#create mean mask - we eventually set one mask file to fit all
mean_mask =  '/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-008/ses-1/func/sub-008_ses-1_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'#nilearn.image.mean_img(mask_files)
amg_coords = []
amg_coorsZmat = []
rightHippo_cor_mat = []
rightHippo_cor_zMat = []
#%matplotlib inline
for func,conf, sub, event in zip(func_files, conf_files, subject_list, event_files):
    print (func)
    print (conf)
    print(sub)
    # create timeline
    seed_time_series ,brain_time_series, brainMasker = createSeedVoxelSeries(coords, func, conf, mask_files[0], sub)
    # take only trauma timeseries
    spec_seedTimeline, spec_brainTimeline = stratifyTimeseries(event, seed_time_series, brain_time_series, sub, script)
    # create seed2voxel matrices
    cor_mat, corz  = seedVoxelCor(spec_seedTimeline, spec_brainTimeline, script, sub, brainMasker, session, nameOfSeed)
    amg_coords.append(cor_mat)
    amg_coorsZmat.append(corz)
###############################################################################################################################################################################
###############################################################################################################################################################################
    
#%% if we already have the nifti files - we can try load the images
from nipype.interfaces.matlab import MatlabCommand
#mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
MatlabCommand.set_default_paths('/home/or/Downloads/spm12/') # set default SPM12 path in my computer. 
from nipype.interfaces import spm
import glob
from nipype import Node, Workflow, MapNode
from nipype.algorithms.misc import Gunzip

img_files = glob.glob('/home/or/kpe_conn/fsl/maths/hippRight_ses2-1_*.nii.gz')

gunzip = MapNode(Gunzip(), name='gunzip',
                 iterfield=['in_file'])
gunzip.inputs.in_file = img_files

# use if comparing same session:
onesamplettestdes = Node(spm.OneSampleTTestDesign(),
                         name="onesampttestdes")


tTest= Workflow(name='oneSampleTtest_total')
tTest.base_dir = ('/media/Data/work/amg_coordsseedBased')

# EstimateModel - estimates the model
level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

# EstimateContrast - estimates group contrast
level2conestimate = Node(spm.EstimateContrast(group_contrast=True),
                         name="level2conestimate")
cont1 = ['Group', 'T', ['mean'], [1]]
level2conestimate.inputs.contrasts = [cont1]

# Threshold - thresholds contrasts
level2thresh = Node(spm.Threshold(contrast_index=1,
                              use_topo_fdr=True,
                              #use_fwe_correction=True, # here we can use fwe or fdr
                              extent_threshold=10,
                              height_threshold= 0.05,
                              extent_fdr_p_threshold = 0.05,
                              height_threshold_type='p-value'),
                              
                                   name="level2thresh")

tTest.connect([
        (gunzip, onesamplettestdes, [('out_file', 'in_files')]),
        (onesamplettestdes, level2estimate, [('spm_mat_file','spm_mat_file')]),
        (level2estimate, level2conestimate, [('spm_mat_file','spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')]),
                                              
                    (level2conestimate, level2thresh, [('spm_mat_file',
                                                        'spm_mat_file'),
                                                       ('spmT_images',
                                                        'stat_image'),
                                                       ]),
        ])

tTest.run()


#%% run second level analysis (one sample t test and then treshold)
import scipy
import glob
import numpy as np
# start with simple t test and tresholding
test = scipy.stats.ttest_1samp(amg_coorsZmat,0,0)
# mean across subjects
mean_zcor = np.mean(amg_coorsZmat2,0)
# before using that we need to run and fit brain masker at least on one subject

mean_zcor_img2 = brainMasker.inverse_transform(mean_zcor.T)
from nilearn import plotting
%matplotlib qt
display = plotting.plot_stat_map(mean_zcor_img,
                                     threshold=0.2, vmax=1,
                                     
                                     title="Seed-to-voxel correlation (Left Amg seed)", display_mode = 'x',
                                     )
display.add_markers(marker_coords=amygdala_coords, marker_color='g',
                    marker_size=200)
np.savetxt(leftAmg_cor_zMat, 'zmat')
# treshold
#FWE
FWE_thr = 0.05/len(mean_zcor)
# create new matrix for correction
corr_mat_thr = np.array(mean_zcor)
# now I can treshold the mean matrix
corr_mat_thr[test[1]>FWE_thr] = 0 # everything that has p value larger than treshold becomes zero 
numNonZero = np.count_nonzero(corr_mat_thr)
print (f'Number of voxels crossed the FWE thr is {numNonZero}')
# transofrm it back to nifti
thr_nifti = brainMasker.inverse_transform(corr_mat_thr.T)
%matplotlib qt
display = plotting.plot_stat_map(thr_nifti,
                                      vmax=1,
                                     threshold = 0.2,
                                     title="Seed-to-voxel correlation (Left Amg seed)", display_mode = 'x',
                                     )
display.add_markers(marker_coords=amygdala_coords, marker_color='g',
                    marker_size=200)


# now lets try FDR
from statsmodels.stats import multitest
# we need to reshape the test p-values array to create 1D array
b = np.reshape(np.array(test[1]), -1)
fdr_mat = multitest.multipletests(b, alpha=0.01, method='fdr_tsbh', is_sorted=False, returnsorted=False)

corr_mat_thrFDR = np.array(mean_zcor)
corr_mat_thrFDR = np.reshape(corr_mat_thrFDR, -1)
corr_mat_thrFDR[fdr_mat[0]==False] = 0
nifti_fdr_thr = brainMasker.inverse_transform(corr_mat_thrFDR.T)
#%% Compare between first and second sessions
# we will use fsl maths - first step will be to create a difference nifti for each subject. 
# We then can run one sample t-test for this difference (or use randomize) - concatenating all subjects diff images to one etc.
img_list_ses1 = glob.glob('/home/or/kpe_conn/trauma_seed_hippocampus_right_sub-*_ses-1_z.nii.gz')
img_list_ses2 = glob.glob('/home/or/kpe_conn/trauma_seed_hippocampus_right_sub-*_ses-2_z.nii.gz')

os.chdir('/home/or/kpe_conn/fsl/maths')
from  nipype.interfaces import fsl
import nipype.pipeline.engine as pe  # pypeline engine
for file1, file2, sub in zip(img_list_ses1, img_list_ses2, subject_list):
    print (file1)
    print (file2)
    !fslmaths {file2} -sub {file1} {'hippRight_ses2-1_' + sub}

# now we can run the same SPM procedure as before - but with the diff images instead of actual ones
#%%
# lets try fsl randomise
import nilearn
func_list = glob.glob('/home/or/kpe_conn/leftAmg_seed_sub-*_z.nii.gz')
func_concat = nilearn.image.concat_imgs(func_list, auto_resample=True) # create a 4d image with subjects as the 4th dimension
# save to file
func_concat.to_filename("leftAmg_seed_concat.nii.gz")

from  nipype.interfaces import fsl
import nipype.pipeline.engine as pe  # pypeline engine

randomize = pe.Node(interface = fsl.Randomise(), base_dir = '/home/or/kpe_conn/fsl',
                    name = 'randomize')
randomize.inputs.in_file = '/home/or/kpe_conn/leftAmg_seed_concat.nii.gz' # choose which file to run permutation test on
#randomize.inputs.mask = '/media/Data/work/KPE_SPM/fslRandomize/group_mask.nii.gz' # group mask file (was created earlier)
randomize.inputs.one_sample_group_mean = True
randomize.inputs.tfce = True
randomize.inputs.vox_p_values = True
randomize.inputs.num_perm = 200

#randomize.inputs.var_smooth = 5

randomize.run()

#show graph
display = plotting.plot_stat_map('/home/or/kpe_conn/fsl/maths/hippRight_ses2-1_008.nii.gz',
                                     vmax=1,
                                     threshold = 0.2,
                                     cut_coords=hippocampus_coors[0],
                                     title="Seed-to-voxel correlation (Left Amg seed)", display_mode = 'x',
                                     )
display.add_markers(marker_coords=hippocampus_coors, marker_color='g',
                    marker_size=200)

# doesn't work so well here
#%% Graph it
%matplotlib qt
fig = nilearn.plotting.plot_stat_map('/home/or/kpe_conn/fsl/randomize/randomise_tstat1.nii.gz')#, alpha=0.7 , cut_coords=(0, 45, -7))
fig.add_contours('/home/or/kpe_conn/fsl/randomize/randomise_tfce_corrp_tstat1.nii.gz', levels=[0.95], colors='w')

# create the mask
group_mask = nilearn.image.resample_to_img(group_mask, func_concat, interpolation='nearest')
group_mask.to_filename(os.path.join("/media/Data/work/KPE_SPM/fslRandomize",  "group_mask.nii.gz"))