#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:28:24 2020

@author: Or Duek
Simple renaming script
"""

import os
import glob

template = '/media/Data/KPE_BIDS/derivatives/fmriprep/sub-008/ses-1/func/sub-008_ses-1_task-INF_*'

file_list = glob.glob(template)

for file in file_list:
    print(file)
    new_fileName = file.split('INF')
    new_fileName = new_fileName[0] + 'Memory' + new_fileName[1]
    print(new_fileName)
    os.rename(file, new_fileName)
    
    

# Now lets run through all fmriPrep KPE results and seperate between memory of scripting and of general 2.5 min script

os.stat