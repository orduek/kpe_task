#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:35:57 2020

@author: Or Duek
Building timeseries only for Resting state - save it as npy

This script should be run outsidr of Spyder, for convenienvce

Here I use sys.argv to get arguments from terminal. The first location (zero) is the script name. Next location will contain the session number (a string). Could add more later if we want. 
cli code should look like that:
    python createTSeries_cli.py session atlas ('aal','shen','yeo') output_dir
"""

work_dir = '/media/Data/work'
#%% [import stuff]
import os
os.chdir('/home/or/kpe_task_analysis')
import nilearn
import pandas as pd
import numpy as np
import sys
from connUtils import timeSeriesSingle, removeVars, stratifyTimeseries

#%% [Subject list]
subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464', '1468', '1480', '1499']
subject_list3 =['008','1223','1263','1293','1307','1322','1339','1343','1356','1364','1369','1387','1390','1403','1464', '1499'] # subjects who has 3 scans '1351',

func_template = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confound_template = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-rest_desc-confounds_regressors.tsv'


# take each subject - create timeline - stratify to events - save. Move to next sub


if sys.argv[2] == 'aal':
    # We want to run the analysis using two different atlases: 1. Shen and 2. AAL (for now)
    print("Loading AAL")
    atlas = nilearn.datasets.fetch_atlas_aal(data_dir = work_dir)
    atlas_filename = atlas.maps
elif sys.argv[2]== 'shen':
    print("loading Shen Parcellation")
    atlas_filename = os.path.join(os.getcwd(),'shen_atlas/shen_1mm_268_parcellation.nii.gz')
    atlas_labes = pd.read_csv(os.path.join(os.getcwd(),'shen_atlas/shen_268_parcellation_networklabels.csv'))
elif sys.argv[2]=='yeo':
    atlas = nilearn.datasets.fetch_atlas_yeo_2011()
    atlas_filename = atlas.thick_17


os.chdir(sys.argv[3])

def create_time(session, subject_list):
    # create a session folder
    try:
        # check if already there
        os.makedirs('session_%s' %(session))
    except:
        print ("Dir already preset")
    os.chdir('session_%s' %(session))
    for sub in subject_list:
        func_file = func_template.format(sub = sub,session = session) # use template and load specific files
        print(func_file)
        confound_file = confound_template.format(sub = sub, session = session)
        
        try:
            timeline = timeSeriesSingle(func_file, confound_file, atlas_filename)
            np.save('subject_%s/rest_ses-%s' %(subject_id, sessopm), timeline)
            stratifyTimeseries(event_file, timeline, sub, 1)
        except:
            print (f'Subject {sub} has no data files')

if __name__ == "__main__": 
    print ("Creating time series")
    createTask_time(sys.argv[1],subject_list) # create a task timeseries for all subjects takes session from CLI
else: 
    print ("Error!!")
