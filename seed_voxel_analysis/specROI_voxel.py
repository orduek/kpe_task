#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:12:46 2020

@author: Or Duek

In this script we will use a specific ROI (using FreeSurfer lookup table) to create a mask for each subject that will serve as seed.
We then mask entire voxels of brain's gray matter (regular mask) and then do seed2voxel connectivity analysis

# In order to do that we need two main files - the original (non processed) bold file and the aparcaseg file (can be taken from freesurfer folder in fmriPrep output)
in order to run use the following:
    python specROI_voxel.py subject_no session work_dir output_dir roi_number roi_name script ('trauma','sad','relax')
"""


# First grab the files

#%%
import os
os.chdir('/home/or/kpe_task_analysis')
import nilearn
import pandas as pd
import numpy as np
import sys
import subprocess
from connUtils import timeSeriesSingle, removeVars, createSeedVoxelSeries, seedVoxelCor

#%% specific function for seed2voxel
def stratifyTimeseriesSeed (events_file, subject_seed_timeseries, subject_brain_timeseries, subject_id, scriptName):
    #trial_line is a parameter - if 0 then will create each line as file. If 1 then each task
    # grab subject events file
    
    events = pd.read_csv(events_file, sep=r'\s+')
  #  timeSeries = np.array(np.load(subject_timeseries, allow_pickle = True))
       
    # read line  by line and create matrix per script
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

#%%
sub = str(sys.argv[1])
session = str(sys.argv[2])
work_dir = str(sys.argv[3])
output_dir = str(sys.argv[4])
roi_num = sys.argv[5]
roi_name = str(sys.argv[6])
script = str(sys.argv[7])

os.chdir(work_dir)     

func_template = '/media/Data/KPE_BIDS/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-Memory_bold.nii.gz'
mask_template = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_desc-brain_mask.nii.gz'
lookup_file = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/anat/sub-{sub}_desc-aparcaseg_dseg.nii.gz'
confound_template = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-Memory_desc-confounds_regressors.tsv'
event_file_template = '/media/Data/PTSD_KPE/condition_files/sub-{sub}_ses-{session}.csv'


# take the apacaseg file and create a ROI mask
mask_file = lookup_file.format(sub=sub, session = session) 
output_file = os.path.join(work_dir, sub+roi_name)

cmd = ['fslmaths', str(mask_file), '-thr', str(roi_num), '-uthr', str(roi_num), str(output_file)]

# if subprocess.call(cmd)==0:
#     print(f'Successesfuly ran sub {sub}')
# else:
#     print(f'failure to run subject {sub}')

func_file = func_template.format(sub=sub, session=session)
confound_file = confound_template.format(sub=sub, session=session)
mask_file = mask_template.format(sub=sub)
event_file = event_file_template.format(sub=sub, session = session)
seed_mask = output_file + '.nii.gz'
#%% Now we should use 
if __name__ == "__main__": 
   
    if subprocess.call(cmd)==0:
        print(f'Successesfuly ran sub {sub}')
    else:
         print(f'failure to run subject {sub}')
    
    seed_time_series, brain_time_series, brainMasker = createSeedVoxelSeries(seed_mask, func_file, confound_file, mask_file, sub)
    spec_seedTimeline, spec_brainTimeline = stratifyTimeseriesSeed(event_file, seed_time_series, brain_time_series, sub, script)
    # create seed2voxel matrices
    try:
        os.chdir(output_dir)
    except:
        print('creating folder')
        os.makedirs(output_dir)
        os.chdir(output_dir)
    cor_mat, corz  = seedVoxelCor(np.mean(spec_seedTimeline, axis = 1), spec_brainTimeline, script, sub, brainMasker,func_file, session, roi_name)
else: 
    print ("Error!!")

#%% Transform all seed2voxel stat files to MNI space using a different file transform2MNI.py
    
    
