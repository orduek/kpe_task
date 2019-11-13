#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:45:25 2019

@author: Or Duek
Reading GSR files from Ledalab and stacking together
This is for the KPE experiment
"""

#%%
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%%
# first grab just subjects with actual valid data

subject = ['1223', '1293', '1315', '1339', '1343','1356', '1364', '1387','1390', '1464' , '1480' ]
med_cond_group = [1,1,1,1,1,0,0,1,0,1,0]
files = glob.glob('/media/Data/PTSD_KPE/physio_data/raw/kpe*/*/*.txt')

# create empty dataframe
df = pd.DataFrame()
# create a string to add to the scripts
count = []
for i in range(1,10):
    count.append(str(i))
        
for file in files:
    
    print(file)
    # get subject number
    sub = file.split('kpe')[1].split('/')[0]
    # get scan number
    scan = file.split('.')[1].split('_')[0]
    
    # check what is this scan
    readScript = pd.read_csv('/media/Data/PTSD_KPE/condition_files/sub-%s_ses-%s.csv' %(sub,scan), sep = '\t')
    # take trial name (type)
    trial_type = readScript['trial_type']
    df1= pd.read_csv(file, sep = "\t")
    df1['scr_id'] = str(sub)
    df1['scan'] = scan
    df1['trial_type'] = trial_type + count
    df = df.append(df1) #, ignore_index=True)


# select only valid subjects for the analysis (using the valid subject list)
dfValid = df.loc[df['scr_id'].isin(subject)] 

dfValid.columns # get column names

df_cond = pd.DataFrame({'scr_id': subject, 'med_cond': med_cond_group})

# merging datasets
dfValid_cond = dfValid.merge(df_cond, left_on='scr_id', right_on='scr_id', how='outer')

%matplotlib qt
sns.boxplot(y = dfValid_cond['CDA.PhasicMax'], x = dfValid_cond['scan'], hue=dfValid_cond['trial_type'], groupby = dfValid_cond['med_cond'])
# show graph with scans and groups
g = sns.FacetGrid(dfValid_cond, col='med_cond', row = 'scan')
g.map(sns.boxplot, 'CDA.PhasicMax', 'trial_type')


# save dataFrame
dfValid_cond.to_csv('testing.csv')

