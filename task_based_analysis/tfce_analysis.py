#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:58:47 2020

@author: Or Duek
run tfce analysis for KPE study
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy
from nilearn import plotting
import nilearn
import nibabel as nib
import subprocess
#%% define 
work_dir = '/media/Data/work/fslRandomise'
try:
    os.chdir(work_dir)
except:
    print ("Dir not found")
    os.mkdir(work_dir)
    os.chdir(work_dir)


#%%
medication_cond = pd.read_csv('/home/or/kpe_task_analysis/task_based_analysis/kpe_sub_condition.csv')

ketamine_list = list(medication_cond['scr_id'][medication_cond['med_cond']==1])
ket_list = []
for subject in ketamine_list:
    print(subject)
    sub = subject.split('KPE')[1]
    ket_list.append(sub)


midazolam_list = list(medication_cond['scr_id'][medication_cond['med_cond']==0])
mid_list = []
for subject in midazolam_list:
    print(subject)
    sub = subject.split('KPE')[1]
    mid_list.append(sub)
mid_list.remove('1480')
#%% 
# build grouped mask
mask_img_temp = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-*/ses-[1,2]/func/sub-*_ses-[1,2]_task-Memory_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
mask_files = glob.glob(mask_img_temp)
mean_mask = nilearn.image.mean_img(mask_files, n_jobs=5)
plotting.plot_anat(mean_mask)

group_mask = nilearn.image.math_img("a>=0.95", a=mean_mask)
nilearn.plotting.plot_roi(group_mask)
#%%
# grab con files for all relevant 
contrast = '02' # set number of contrast
ket_func_ses1 = ['/media/Data/KPE_results/work/kpeTask_ses1/1stLevel/_subject_id_%s/con_00%s.nii' % (sub, contrast) for sub in ket_list]
ket_func_ses2 = ['/media/Data/KPE_results/work/kpeTask_ses2/1stLevel/_subject_id_%s/con_00%s.nii' % (sub, contrast) for sub in ket_list]
mid_func_ses1 = ['/media/Data/KPE_results/work/kpeTask_ses1/1stLevel/_subject_id_%s/con_00%s.nii' % (sub, contrast) for sub in mid_list]
mid_func_ses2 = ['/media/Data/KPE_results/work/kpeTask_ses2/1stLevel/_subject_id_%s/con_00%s.nii' % (sub, contrast) for sub in mid_list]


#%% First compare ketamine group
# create diff image 
group = 'mid'
for ses1,ses2 in zip(mid_func_ses1,mid_func_ses2):
    print (ses1)
    print (ses2)
    sub = ses1.split('id_')
    sub = sub[1].split('/')[0]
    print(sub)
    diff_file = 'kpe' + sub + 'diff' + group + 'con' + contrast
    cmd = '!fslmaths' + ses1 + '-sub' + ses2 + diff_file
    cmd = ['fslmaths', str(ses1), '-sub', str(ses2), str(diff_file)]
    subprocess.call(cmd)
    
    
diff_list_con = glob.glob('/media/Data/work/fslRandomise/kpe*diff%scon%s.nii.gz' %(group,contrast))
len(diff_list_con)


#%% Creating concatenated contrast (across subjects) and group mask
copes_concat = nilearn.image.concat_imgs(diff_list_con, auto_resample=True)
copes_concat.to_filename(os.path.join(work_dir, "con%s_%s.nii.gz" %(contrast, group)))


group_mask = nilearn.image.resample_to_img(group_mask, copes_concat, interpolation='nearest')
group_mask.to_filename(os.path.join(work_dir,  "group_mask.nii.gz"))

#%% Running randomization
from  nipype.interfaces import fsl
import nipype.pipeline.engine as pe  # pypeline engine
randomize = pe.Node(interface = fsl.Randomise(), base_dir = work_dir,
                    name = 'randomize')
randomize.inputs.in_file = os.path.join(work_dir,  "con%s_%s.nii.gz" %(contrast,group)) # choose which file to run permutation test on
randomize.inputs.mask = os.path.join(work_dir, 'group_mask.nii.gz') # group mask file (was created earlier)
randomize.inputs.one_sample_group_mean = True
randomize.inputs.tfce = True
#randomize.inputs.vox_p_values = True
randomize.inputs.num_perm = 500
#randomize.inputs.var_smooth = 5

randomize.run()
