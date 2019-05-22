#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:16:09 2019

@author: Or Duek
In this script I will first take data for each subject for each run. 

"""
#%% Specific subject data
data_dir = '/media/Data/FromHPC/output/fmriprep'
subject_id = 1063

def extractInfo (subject_id, data_dir):
    
    # function should recieve subject id. Take subject runs (nifti files) and mat files
    # then it will create a seperate Bunsh and contrasts for each subject.
    # it needs to return filename, bunch of the run and contrasts for each run
    import os
    import pandas as pd
    from nipype.interfaces.base import Bunch
    os.chdir('/media/Data/work')
    from readConditionFiles_r_aPTSD import loadmat, readConditions, organizeBlocks 
    
    eventsTotal = organizeBlocks(subject_id)  # creates array of 8 arrays. One for each run
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
    
    
    from bids.grabbids import BIDSLayout
    layout = BIDSLayout(data_dir)
    tasks = ['3','4','5','6'] # number of task (i.e. block. corresponding to file name)
    source_epi = layout.get(type="bold", session="1", extensions="nii.gz", subject = subject_id) # get file list of subject
    
    # grab confounds for the specific file
    model_spec = []
    contrastList =[]
    filenames = []
    condition = []
    
    for r in range(len(tasks)):
        print (source_epi[r])
        confounds = pd.read_csv(os.path.join(data_dir, 
                                    "sub-%s"%subject_id, "ses-%s"%source_epi[r].session, "func", 
                                    "sub-%s_ses-%s_task-%s_desc-confounds_regressors.tsv"%(source_epi[r].subject, source_epi[r].session, tasks[r])),
                                       sep="\t", na_values="n/a")
    
        if eventsTotal[r].condition[r+1]=='Loss':
                
           
            model_spec.insert(r, 
                          Bunch(conditions = ['LossRisk', 'LossAmb'],
                                onsets = [list(eventsTotal[r][eventsTotal[r].condName=='LossRisk'].onset),
                                         list(eventsTotal[r][eventsTotal[r].condName=='LossAmb'].onset)],
                                durations = [ list(eventsTotal[r][eventsTotal[r].condName=='LossRisk'].duration),
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
            condition_names = ['LossRisk', 'LossAmb']
       #     GainRisk_cond = ['GainRisk','T', condition_names,[1,0,0,0]]
        #    GainAmb_cond = ['GainAmb','T',condition_names,[0,1,0,0]]
            LossRisk_cond = ['LossRisk','T', condition_names,[1,0]]
            LossAmb_cond = ['LossAmb','T',condition_names,[0,1]]
           # Gain_all = ['Gain', 'F', [GainRisk_cond, GainAmb_cond]]
            Loss_all = ['Loss', 'F', [LossRisk_cond, LossAmb_cond]]       
            contrasts = [LossRisk_cond, LossAmb_cond, Loss_all] 

            contrastList.insert(r, list(contrasts))
            condition.insert(r, 'Loss')
            
        elif eventsTotal[r].condition[r+1]=='Gain':
            print ('Gain')
            
            model_spec.insert(r,
                          Bunch(conditions = ['GainRisk','GainAmb'],
                                onsets = [list(eventsTotal[r][eventsTotal[r].condName=='GainRisk'].onset),
                                         list(eventsTotal[r][eventsTotal[r].condName=='GainAmb'].onset)],
                                durations = [list(eventsTotal[r][eventsTotal[r].condName=='GainRisk'].duration),
                                             list(eventsTotal[r][eventsTotal[r].condName=='GainAmb'].duration)],
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
            condition_names = ['GainRisk','GainAmb']
            #LossRisk_cond = ['LossRisk','T', condition_names,[0,0,1,0]]
            #LossAmb_cond = ['LossAmb','T',condition_names,[0,0,0,1]]
            GainRisk_cond = ['GainRisk','T', condition_names,[1,0]]
            GainAmb_cond = ['GainAmb','T',condition_names,[0,1]]
            Gain_all = ['Gain', 'F', [GainRisk_cond, GainAmb_cond]]
            #Loss_all = ['Loss', 'F', [LossRisk_cond, LossAmb_cond]]                     
            contrasts = [GainRisk_cond, GainAmb_cond, Gain_all] #, LossRisk_cond, LossAmb_cond, Loss_all] 
#    
            contrastList.insert(r, list(contrasts))
            condition.insert(r, 'Gain')
        filenames.insert(r, source_epi[r].filename)
#    condition_names = ['GainRisk', 'GainAmb' ,'LossRisk', 'LossAmb']                          
#
#    GainRisk_cond = ['GainRisk','T', condition_names ,[1,0,0,0]]
#    GainAmb_cond = ['GainAmb','T', condition_names ,[0,1,0,0]]
#    LossRisk_cond = ['LossRisk','T', condition_names,[0,0,1,0]]
#    LossAmb_cond = ['LossAmb','T',condition_names,[0,0,0,1]]
#    Risk_vs_Amb = ["Risk vs. Amb",'T', condition_names ,[0.5, -0.5, 0.5, -0.5]]
#    Gain_vs_Loss = ["Gain vs. Loss",'T', condition_names ,[0.5, 0.5, -0.5, -0.5]]
#    
   # gain_total = ["Gain", 'F', [GainRisk_cond, GainAmb_cond]]
   #loss_total = ["Loss", 'F', [LossAmb_cond, LossRisk_cond]]
   
#    contrasts=[GainRisk_cond, GainAmb_cond, LossRisk_cond, LossAmb_cond, Risk_vs_Amb, Gain_vs_Loss] #, gain_total]#, loss_total]
    
  
    return (model_spec, contrasts, filenames, condition)