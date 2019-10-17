#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:59:04 2019

@author: Or Duek
Running task based connectivity analysis for the KPE study
1. General (whole brain/NBS)
2. Look specifically at DMN and Salience networks (seed based)
3. Amygdala and Hippocampus (Seed based?)
"""

#%% set working variables
work_dir = '/media/Data/work/connectivity'

# %%load parcellation (Yeo? / Shen?)
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import plotting
import scipy
# yeo
#atlas_filename = '/home/or/Downloads/1000subjects_reference_Yeo/Yeo_JNeurophysiol11_SplitLabels/MNI152/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz'
#atlas_labes = pd.read_csv('/home/or/Downloads/1000subjects_reference_Yeo/Yeo_JNeurophysiol11_SplitLabels/Yeo2011_17networks_N1000.split_components.glossary.csv')
#coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)
# shen
atlas_filename = '/home/or/Downloads/shenPar/shen_1mm_268_parcellation.nii.gz'
atlas_labes = pd.read_csv('/home/or/Downloads/shenPar/shen_268_parcellation_networklabels.csv')
colors = pd.read_csv('/home/or/Downloads/shenPar/shen_268_parcellation_networklabels_colors.csv')
coords = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)

atlas_labes = np.array(atlas_labes)
atlas_labes.shape

#%% Methods for analysis
def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['csf', 'white_matter', 'framewise_displacement',
                          'a_comp_cor_00', 'a_comp_cor_01',	'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 
                        'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']] # can add 'global_signal' also
     # change NaN of FD to zero
    finalConf = np.array(finalConf)
    finalConf[0,2] = 0 # if removing FD than should remove this one also
    return finalConf

# build method for creating time series for subjects
def timeSeries(func_files, confound_files):
    total_subjects = [] # creating an empty array that will hold all subjects matrix 
    # This function needs a masker object that will be defined outside the function
    for func_file, confound_file in zip(func_files, confound_files):
        print(f"proccessing file {func_file}") # print file name
        confoundClean = removeVars(confound_file)
        confoundArray = confoundClean#confoundClean.values
        time_series = masker.fit_transform(func_file, confounds=confoundArray)
        #time_series = extractor.fit_transform(func_file, confounds=confoundArray)
        #masker.fit_transform(func_file, confoundArray)
        total_subjects.append(time_series)
    return total_subjects

# contrasting two timePoints
def contFuncs(time_series1, time_series2):
    twoMinusOneMat = []
    for scanMatrix, scanMatrix2 in zip(time_series1, time_series2):
        a = scanMatrix2 - scanMatrix
        twoMinusOneMat.append(a)
    return np.array(twoMinusOneMat)

# create correlation matrix per subject
def createCorMat(time_series):
    # create correlation matrix for each subject
    fullMatrix = []
    for time_s in time_series:
        correlation_matrix = correlation_measure.fit_transform([time_s])[0]
        fullMatrix.append(correlation_matrix)
    return fullMatrix


#%% Set masker object
# Here you set the specific methods for masking and correlation. Please see Nilearn website for more info.
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiLabelsMasker
# in this mask we standardize the values, so mean is 0 and between -1 to 1
# masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, smoothing_fwhm = 6,
#                          memory="/home/oad4/scratch60/shenPar_nilearn",high_pass=.01 , low_pass = .1, t_r=1, verbose=5)

# use different masker when using Yeo atlas. 
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, smoothing_fwhm = 6,
                        memory="/media/Data/nilearn",t_r=1, verbose=5, high_pass=.01 , low_pass = .1) # As it is task based we dont' bandpassing high_pass=.01 , low_pass = .1)
                           
from nilearn.connectome import ConnectivityMeasure
#correlation_measure = ConnectivityMeasure(kind='partial correlation') # can choose partial - it might be better
correlation_measure = ConnectivityMeasure(kind='correlation') # can choose partial - it might be better        

#%% load files (image, task events)
subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464', '1468', '1480', '1499']
subject_list3 =['008''1223','1263','1293','1307','1322','1339','1343','1356','1364','1369','1387','1390','1403','1464'] # subjects who has 3 scans '1351',

def fileList(subjects, session):
    func_files = ['/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-%s/ses-%s/func/sub-%s_ses-%s_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' % (sub,session,sub,session) for sub in subjects]
    return func_files

def confList(subjects, session):
    confound_files = ['/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-%s/ses-%s/func/sub-%s_ses-%s_task-Memory_desc-confounds_regressors.tsv' % (sub,session,sub,session) for sub in subjects]
    return confound_files

def eventfList(subjects, session):
    confound_files = ['/media/Data/PTSD_KPE/condition_files/sub-%s_ses-%s.csv' % (sub,session) for sub in subjects]
    return confound_files

#%% create time series (with condounders)
session1 = timeSeries(func_files=fileList(subject_list,'1'), confound_files=confList(subject_list, '1'))
session2 = timeSeries(func_files=fileList(subject_list,'2'), confound_files=confList(subject_list, '2'))
session3 = timeSeries(func_files=fileList(subject_list3,'3'), confound_files=confList(subject_list3, '3'))
#%% Save timeseries arrays as npy files
os.chdir('/home/or/kpe_conn/ShenParc')
np.save("session_1Timeseries_Shen",session1) # saving array
np.save("session_2TimeseriesShen",session2)
np.save("session_3TimeseriesShen", session3)
#%% load sessions
session1 = list(np.load('session_1Timeseries_Shen.npy', allow_pickle = True)) # loading the saved array
session2 = list(np.load('session_2TimeseriesShen.npy', allow_pickle = True)) # loading the saved array
session3 = list(np.load('session_3TimeseriesShen.npy', allow_pickle = True))
#%% Build and save each subject's timeseries in a file
def saveSubjectSeries(subject_list, session, timeSerArray):
    # First we create a new directory for the session
    try:
        # check if already there
        os.makedirs('session_%s' %(session))
    except:
        print ("Dir already preset")
    # this short function takes subject list, session number and the full array of subjects timeseries that matches the subject list. It saves a timeseries per subject as .npy file
    for sub,arr in zip(subject_list, timeSerArray):
        print (f'Saving subject No {sub}')
        np.save(('session_%s/sub-%s_session-%s_timeseries'%(session,sub,session)), arr)

#%% first session
saveSubjectSeries(subject_list, '1', session1)
saveSubjectSeries(subject_list, '2', session2)
saveSubjectSeries(subject_list3, '3', session3)
#%% stratify to tasks

# need to build a function that will read the event file - take onset and duration of each line and stratify the timeseries accordingly
def stratifyTimeseries (events_file, subject_timeseries, subject_id, trial_line):
    #trial_line is a parameter - if 0 then will create each line as file. If 1 then each task
    # grab subject events file
    events = pd.read_csv(events_file, sep=r'\s+')
    timeSeries = np.array(np.load(subject_timeseries, allow_pickle = True))
    # create a subject folder
    try:
        # check if already there
        os.makedirs('subject_%s' %(subject_id))
    except:
        print ("Dir already preset")
    
    # read line  by line and create matrix per line
    if trial_line==0:
        for line in events.iterrows():
            print (f' Proccessing line {line}')
            numberRow = line[0] # take row number to add to matrix name later
            onset = round(line[1].onset) # take onset and round it
            duration = round(line[1].duration)
            trial_type = line[1].trial_type
            specTimeline = timeSeries[onset:(onset+duration),:]
            np.save('subject_%s/speficTrial_%s_%s' %(subject_id,numberRow, trial_type), specTimeline)
    elif trial_line==1: # read by trial type and create specific timeline for each script
        traumaOnset = []
        sadOnset = []
        relaxOnset = []
        traumaDuration = []
        sadDuration = []
        relaxDuration = []
        for line in events.iterrows(): # runs trhough the events file, takes the specific files and create timeseries per each
            print (line)
            if line[1]['trial_type'].find('trauma')!= -1:
                print('trauma')
                traumaOnset.append(round(line[1].onset))
                traumaDuration.append(round(line[1].duration))
            elif line[1]['trial_type'].find('sad')!= -1:
                print('sad')
                sadOnset.append(round(line[1].onset))
                sadDuration.append(round(line[1].duration))
            elif line[1]['trial_type'].find('relax')!= -1:
                print('relax')
                relaxOnset.append(round(line[1].onset))
                relaxDuration.append(round(line[1].duration))
        trauma_timeline = np.concatenate([timeSeries[traumaOnset[0]:traumaOnset[0] + traumaDuration[0],:], timeSeries[traumaOnset[1]:traumaOnset[1]+ traumaDuration[1],:], timeSeries[traumaOnset[2]:traumaOnset[2]+traumaDuration[2],:]])
        sad_timeline = np.concatenate([timeSeries[sadOnset[0]:sadOnset[0] + sadDuration[0],:], timeSeries[traumaOnset[1]:sadOnset[1]+ sadDuration[1],:], timeSeries[sadOnset[2]:sadOnset[2]+sadDuration[2],:]])
        relax_timeline = np.concatenate([timeSeries[relaxOnset[0]:relaxOnset[0] + relaxDuration[0],:], timeSeries[relaxOnset[1]:relaxOnset[1]+ relaxDuration[1],:], timeSeries[relaxOnset[2]:relaxOnset[2]+relaxDuration[2],:]])
        np.save('subject_%s/traumaTrials' %(subject_id), trauma_timeline)
        np.save('subject_%s/sadTrials' %(subject_id), sad_timeline)
        np.save('subject_%s/relaxTrials' %(subject_id), relax_timeline)
        
    # or read by trial type and create matrix per trial type
    else:
        print ("Need to run by task")
    # extract subject specific timeseries from 3D array

#%% Create subject's files per line
# first - move to relevant folder
os.chdir('/home/or/kpe_conn/ShenParc/session_4')
# run loop
misDat = [] # array that will hold subjects with missing data
for sub in subject_list:
    try:
        events_files = '/media/Data/PTSD_KPE/condition_files/sub-%s_ses-4.csv' %sub
        timeseries = '/home/or/kpe_conn/ShenParc/session_4/sub-%s_session-4_timeseries.npy' %sub
        subject_id = sub
        stratifyTimeseries(events_files,timeseries,sub,1) # set 0 to do by row (i.e. each script) or 1 to do by task (i.e. 3 scripts per condition)
    except:
        print(f'Subject {sub} does not have a valid file')
        misDat.append(sub)
        
#%% just plot timeseries of one subject
sub1 = session1[0]
sub2 = session1[1]
plt.plot(sub1[:,(227,228)]) # plotting Amg  (according to Shen parcelation)
#plt.plot(sub2[:,(227)])
#%% create correlation matrix
# iterate through subject list - take each timeseries and create correlation matrix per subject
correlation_allSubs = []
for sub in subject_list:
    timeseries = np.load('/home/or/kpe_conn/ShenParc/session_1/subject_%s/speficTrial_0_trauma.npy'%sub )
    correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
    correlation_allSubs.append(correlation_matrix)
correlation_Trauma_0_Mean = correlation_measure.mean_

corr_matt_mean = np.mean(np.array(correlation_allSubs), axis = 0)
corr_matt_mean = corr_matt_mean + corr_matt_mean.T
corr_matt_mean *= .5

%matplotlib qt
plotting.plot_matrix(correlation_Trauma_0_Mean, labels=atlas_labes[:,0], colorbar=True, tri = 'diag', grid = 'color') #,
# Create network

correlation_Trauma_0_Mean_thr = np.array(corr_matt_mean) # creates a different array instead of just a pointer
correlation_negative = np.array(correlation_Trauma_0_Mean_thr)# np.zeros((114,114))
correlation_positive = np.array(correlation_Trauma_0_Mean_thr)
correlation_negative[correlation_negative > 0] = 0
correlation_negative[correlation_negative < 0] = 1
correlation_positive[correlation_positive<0] = 0
correlation_positive[correlation_positive>0] =1
np.savetxt("negative_corr.csv", correlation_negative, delimiter = ',')
np.savetxt("positive_corr.csv", correlation_positive, delimiter = ',')
#%% Method to create array of correlation matrices for subjects
def creatSubCor(subject_list, session, fileName):
    # takes subject list, session number and filename and return 3D array of subject (0 axis) correlation matrix
    correlation_allSubs = []
    for sub in subject_list:
        print(sub)
        try:
            timeseries = np.load('/home/or/kpe_conn/ShenParc/session_%s/subject_%s/%s'%(session,sub,fileName) )
            correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
            correlation_allSubs.append(correlation_matrix)
        except: 
            print (f'Subject {sub} does not have a valid file')
        
        # add a line to save correlation matrix as csv file
       # np.savetxt('sub-%s_ses-%s_corrMat.csv'%(sub, session), correlation_matrix, delimiter=',')
        
    return correlation_allSubs
#%% running t-test on all subjects to set treshold of edges
# create a method to treshold a matrix
def tresholMat(corr_mat, p_thr):
    # takes a 3D array (subject, Nodes,Nodes) of correlation matrices for all subjects
    # takes p value to correct later on (p_thr)
    # create a mean matrix from subject one
    corr_mat_mean = np.mean(np.array(corr_mat), axis = 0)
    import scipy
    test = scipy.stats.ttest_1samp(corr_mat, 0) # run a simple one-sample t test on correlation matrix across all subjects. 
    a = test[1] # takes array of p-values
    # treshold using FWE correction
    FWE_thr = p_thr/(len(a)*(len(a)-1)/2)
    # create new matrix for correction
    corr_mat_thr = np.array(corr_mat_mean)
    # now I can treshold the mean matrix
    corr_mat_thr[a>FWE_thr] = 0 # everything that has p value larger than treshold becomes zero 
    numNonZero = np.count_nonzero(corr_mat_thr)
    print (f'Number of edges crossed the FWE thr is {numNonZero}')
    return corr_mat_thr


# create binarized positive and negative matrices also
# (for use in BioImageWeb)
def pos_neg (corr_thr):
    # takes the tresholded mean matrix and binarize to positive and negative
    positive_corr = np.array(corr_thr)
    positive_corr[positive_corr > 0] = 1
    positive_corr[positive_corr <0] = 0
    negative_corr = np.array(corr_thr)
    negative_corr[negative_corr > 0] = 0
    negative_corr[negative_corr <0] = 1
    return positive_corr , negative_corr


#%% do Trauma first script - 1st session
trauma_1st_ses = creatSubCor(subject_list, '1', 'speficTrial_0_trauma.npy')
trauma_1st_thr = tresholMat(trauma_1st_ses, 0.01)
pos_corr , neg_corr = pos_neg(trauma_1st_thr)
np.savetxt("pos_cor_Traum1.csv" , pos_corr, delimiter= ",")
np.savetxt("neg_cor_Traum1.csv" , neg_corr, delimiter= ",")

#% second session
trauma_2_1st_ses = creatSubCor(subject_list, '1', 'speficTrial_4_trauma.npy')
trauma_2_1st_ses_thr = tresholMat(trauma_2_1st_ses, 0.01)
pos_corr , neg_corr = pos_neg(trauma_2_1st_ses_thr)
np.savetxt("pos_cor_Traum2_ses1.csv" , pos_corr, delimiter= ",")
np.savetxt("neg_cor_Traum2_ses1.csv" , neg_corr, delimiter= ",")
#%% do trauma on second session
trauma_2nd_ses2 = creatSubCor(subject_list, '2', 'speficTrial_0_trauma.npy')
trauma_2ndSes2_corr = tresholMat(trauma_2nd_ses2, 0.01)
pos_corr , neg_corr = pos_neg(trauma_2ndSes2_corr)
np.savetxt("pos_cor_TraumSes2.csv" , pos_corr, delimiter= ",")
np.savetxt("neg_cor_TraumSes2.csv" , neg_corr, delimiter= ",")

# check difference between first and second script:
deltaMatrix = np.mean(np.array(trauma_2_1st_ses), axis = 0) - np.mean(np.array(trauma_1st_ses), axis = 0) # contrast the two matrices
deltaMatrix_each = np.array(trauma_2nd_ses2) - np.array(trauma_1st_ses)
for mat, sub in zip(deltaMatrix_each, subject_list):
    print (sub)
    np.savetxt('sub-%s_deltaMat1_2.csv' %sub, mat, delimiter=',')

# run -ttest
tr_1_2Ttest = scipy.stats.ttest_rel(trauma_1st_ses, trauma_2_1st_ses, axis = 0)
p_vals = tr_1_2Ttest[1]
delta_mat_thr = np.array(deltaMatrix)
# now I can treshold the mean matrix
fwe_thr = 0.05/(len(p_vals)*(len(p_vals)-1)/2)
delta_mat_thr[p_vals>fwe_thr] = 0 # everything that has p value larger than treshold becomes zero 
numNonZero = np.count_nonzero(delta_mat_thr)
print (f'Number of edges crossed the FWE thr is {numNonZero}')

#%% Do sad first
sad_corr_subs = creatSubCor(subject_list, '1', 'speficTrial_2_sad.npy') # take sad
sad_corr_thr = tresholMat(sad_corr_subs, 0.01)

#%% compare sad to trauma
sadVsTrauma = scipy.stats.ttest_rel(correlation_allSubs, sad_corr_subs, axis = 0)

#%%  
# need to find a nice way to eliminate al zerros from the graph (because we also have negative values that we want to keep)
plotting.plot_matrix(trauma_0_corr, labels=atlas_labes[:,0], colorbar=True, tri = 'diag', grid = 'color')
plotting.plot_matrix(sad_corr_thr, labels=atlas_labes[:,0], colorbar=True, tri = 'diag', grid = 'color')
# check how many edges crossed that thr
# plotting only edges that holds FWE correction 
plotting.plot_matrix(correlation_Trauma_0_Mean_thr, labels=atlas_labes[:,0], colorbar=True, tri = 'diag', grid = 'color')

# plot in connectome
plotting.plot_connectome(trauma_1st_thr, coords, edge_threshold='95%', colorbar=True, black_bg = True, annotate = True)

##
plotting.plot_matrix(trauma_2_1st_ses_thr, labels=atlas_labes[:,0], colorbar=True, tri = 'diag', grid = 'color')
#%% Graph theory calculations (Degree centrality etc)
import networkx as nx
G = nx.from_numpy_array(adjTrt1_2)
%matplotlib qt
nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='gray', linewidths=1, font_size=15)
list(nx.connected_components(G))
sorted(d for n, d in G.degree())
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
nx.clustering(G)
l = nx.degree_centrality(G)
centralityDat = pd.DataFrame(list(l.items()), columns = ['Node','Centrality'])
sorted(l,key = l.get, reverse = True)

plotting.plot_connectome(correlation_matrix, coords,
                             edge_threshold=0.7, colorbar=True, black_bg = True, annotate = True)
#%% Compare matrices between sessions
deltaMatrix = trauma_2ndSes2_corr - trauma_1st_thr # contrast the two matrices
# run -ttest
tr_1_2Ttest = scipy.stats.ttest_rel(trauma_1st_ses, trauma_2nd_ses2, axis = 0)
p_vals = tr_1_2Ttest[1]
delta_mat_thr = np.array(deltaMatrix)
# now I can treshold the mean matrix
fwe_thr = 0.05/(len(p_vals)*(len(p_vals)-1)/2)
delta_mat_thr[tr_1_2Ttest[1]>fwe_thr] = 0 # everything that has p value larger than treshold becomes zero 
numNonZero = np.count_nonzero(delta_mat_thr)
print (f'Number of edges crossed the FWE thr is {numNonZero}')
# maybe we should run network based statistics to compare the two networks

#%% plotting stuff
# take colors
colors = pd.read_csv('/home/or/Downloads/shenPar/shen_268_parcellation_networklabels_colors.csv')
color_node = list(colors["color"])
# plot in specific region as seed
# create 268 x 268 array
empty = np.zeros((268,268))
# take only left amygdala connections
empty[227,:] = trauma_1st_thr[227,:]
empty[:,227] = trauma_1st_thr[:,227]
# plot
plotting.plot_connectome(empty, coords, edge_threshold='95%', colorbar=True, black_bg = True, annotate = True, node_color = color_node)

# plot in browser
view = plotting.view_connectome(empty, coords, threshold='90%') 
view.open_in_browser() 
view_color.open_in_browser()


#%% Run Network Based Analysis
# first reshape the matrix dimensions (right now its [subs,x,y]) to [x,y,subs]
trt1Reshape = np.moveaxis(np.array(trauma_1st_ses),0,-1)
trt2Reshape = np.moveaxis(np.array(trauma_2_1st_ses),0,-1)

from bct import nbs
# we compare ket1 and ket3
pval, adj, _ = nbs.nbs_bct(trt1Reshape, trt2Reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
# one network is different

#%% compare sad to trauma 1
sad1Reshape = np.moveaxis(np.array(sad_corr_subs),0,-1)
pvalSadTr, adjSadTr, _ = nbs.nbs_bct(trt1Reshape, sad1Reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
# sig difference between those two networks. 
# now contrast between the mean matrices of each condition
contMat = np.mean(trauma_1st_ses, axis = 0) - np.mean(sad_corr_subs, axis = 0)  
# then multiply by the adjacency matrix created by NBS.
adjCor = contMat * adjSadTr
np.max(adjCor)
# now we can differentiate to two positive and negative matrices
pos_cor = np.array(adjCor)
pos_cor[pos_cor<=0] = 0 # zero for everything lower than zero
np.max(pos_cor)

neg_cor = np.array(adjCor)
neg_cor[neg_cor>=0] = 0 # zero everything more than zero
np.min(neg_cor)

# save to csv
np.savetxt("pos_cor_NBS_adj.csv" , pos_cor, delimiter= ",")
np.savetxt("neg_cor_NBS_adj.csv" , neg_cor, delimiter= ",")

#%% lets check trauma 2 with sad2
 
#%%now lets check if there's a difference between sad and trauma at the second session
# load the second session first sad script
sad_corr_subs_ses2 = creatSubCor(subject_list, '2', 'speficTrial_2_sad.npy') # take sad
sad2_1stSes = creatSubCor(subject_list, '1', 'speficTrial_5_sad.npy')
# problem with 1403 - run timeseries for this one again
#time_series_1403 = masker.fit_transform('/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-1403/ses-2/func/sub-1403_ses-2_task-Memoryb_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', confounds=removeVars('/media/Data/KPE_fmriPrep_preproc/kpeOutput/derivatives/fmriprep/sub-1403/ses-2/func/sub-1403_ses-2_task-Memoryb_desc-confounds_regressors.tsv'))
## save the new fixed timeseries:
#session = '2'
#sub = '1403'
#np.save(('session_%s/sub-%s_session-%s_timeseries'%(session,sub,session)), time_series_1403)
# extract task specific
# now run 1st trauma on 2nd session
trt1_ses2Reshape = np.moveaxis(np.array(trauma_2nd_ses2), 0 , -1)
# back to comparing sad2 and trauma 2
# reshape sad2
sad2Reshape = np.moveaxis(np.array(sad2_1stSes),0,-1)
pvalSad2trt2, adjSad2trt2, _ = nbs.nbs_bct(trt2Reshape, sad2Reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
# no significant difference
#%% Compare trauma 2 and sad 2
sad2_1stSesReshape = np.moveaxis(np.array(sad2_1stSes), 0 ,-1)
pvalSad2, adjSad2, _ = nbs.nbs_bct(trt1_ses2Reshape, sad2_1stSesReshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
#%% Lets check difference between trauma 1 and trauma on second session
pvalTrt1_2 , adjTrt1_2, _ = nbs.nbs_bct(trt1_ses2Reshape, trt1Reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
# almost sig. difference between trauma on first session and on second (0.052) - should check again with more subjects.

trt1_vs2_meanDistance = np.mean(np.array(trauma_2nd_ses2), axis = 0) - np.mean(np.array(trauma_1st_ses), axis = 0) # create delta between two matrices
trt1_vs2_meanDistance.min()
# treshold according to adjacancy matrix

trt1_vs2_tresh = np.array(trt1_vs2_meanDistance) * adjTrt1_2
trt1_vs2_tresh.max()
# creating positive matrix
trt1_vs2_pos = np.array(trt1_vs2_tresh)
trt1_vs2_pos[trt1_vs2_pos<0] = 0
trt1_vs2_pos.min()
sumPosTrt1vs2.sum()

trt1_vs2_neg = np.array(trt1_vs2_tresh)
trt1_vs2_neg[trt1_vs2_neg >= 0] = 0
trt1_vs2_neg.min()
#%% Function that takes original matrices and returns sum of positive and negative correlations and delta matrix (if needed)
# need to do it for each subject
def sumPosEdges(matrix1, matrix2, adj):
    # delta
    deltaMat = np.array(matrix2) - np.array(matrix1)
    deltaMat_thr = np.array(deltaMat) * adj
    deltaMat_thr_pos = np.array(deltaMat_thr)
    deltaMat_thr_pos[deltaMat_thr_pos<0] = 0
    subjectSum_pos = deltaMat_thr_pos.sum()
    deltaMat_thr_neg = np.array(deltaMat_thr)
    deltaMat_thr_neg[deltaMat_thr_neg>0] = 0 # create negarive matrix
    subjectSum_neg = deltaMat_thr_neg.sum()
    return subjectSum_pos , subjectSum_neg , deltaMat_thr
#%%
subjectPosEdges = []
for i,n in enumerate(subject_list):
    print(i)
    print(n)
    subjectPosEdges.append(sumPosEdges(trauma_1st_ses[i], trauma_2nd_ses2[i], adjTrt1_2)[0]) # take only positive
    
subjectPosEdges_Neg = []
for i,n in enumerate(subject_list):
    print(i)
    print(n)
    subjectPosEdges_Neg.append(sumPosEdges(trauma_1st_ses[i], trauma_2nd_ses2[i], adjTrt1_2)[1]) # take only negative

#%% Compare different groups - take group label from database
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('/home/or/Documents/kpe_analyses/KPEIHR0009_data_all_scored.xlsx') #, sheetname='KPE DATA')
# take only what we want
df_clinical = df.filter(['scr_id','med_cond','caps5_totals','pcl5_total_screen','pcl5_total_visit1','pcl5_total_visit7','pcl5_total_followup1','pcl5_total_followup2','cadss_total', 'bdi_total_screen', 'bdi_total_visit1', 'bdi_total_visit7', 'bdi_total_followup1', 'bdi_total_followup2']) #, (like='bdi_total')]
#df_clinical_no008 = df_clinical[df_clinical.scr_id != 'KPE008']
# so simple linear regression - take tresholded positive edges and sum them up
scipy.stats.pearsonr(subjectPosEdges, df_clinical['pcl5_total_visit1'])
# check the difference
diffPCL = df_clinical['pcl5_total_followup1'] - df_clinical['pcl5_total_visit7']
diffPCL_FU_screen = df_clinical['pcl5_total_followup1'] - df_clinical['pcl5_total_visit1']
diffPCL[2] = 0
#np.savetxt('diffPCL_FU1.csv', diffPCL, delimiter=',')
# check differences in pcl from FU1 to Visit7 - compare with trauma positive edges
scipy.stats.pearsonr(subjectPosEdges[mask], diffPCL[mask])

mask = ~np.isnan(subjectPosEdges[0:18]) & ~np.isnan(diffPCL) & ~np.isnan(df_clinical['med_cond'])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(subjectPosEdges[0:18])[mask],diffPCL[mask])
import matplotlib.pyplot as plt
line = slope*np.array(subjectPosEdges)[mask]+intercept

# plot the regression model
plt.title('Positive sum of correlations - 16 valid subjects')
plt.xlabel('Sum of connectivity')
plt.ylabel('Change in PCL (end of trt to 30 days)')
plt.plot(subjectPosEdges, diffPCL, 'o', np.array(subjectPosEdges)[mask], line)
plt.show()

# lets check the negative matrix
mask = ~np.isnan(subjectPosEdges_Neg) & ~np.isnan(diffPCL) & np.isnan(df_clinical['med_cond'])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.array(subjectPosEdges_Neg)[mask],diffPCL_FU_screen[mask])
import matplotlib.pyplot as plt
line = slope*np.array(subjectPosEdges_Neg)[mask]+intercept
plt.plot(subjectPosEdges_Neg, diffPCL_FU_screen, 'o', np.array(subjectPosEdges_Neg)[mask], line)
#%% comparing sad1 to sad2
pvalSad1vs2, adjSad1_vs2, _ = nbs.nbs_bct(sad1Reshape, sad2Reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
# no difference

#%%
############################# Run same analysis on the whole task (i.e. all three trauma and sad scripts no just the first one) ###########################################################
trt_ses1 = creatSubCor(subject_list, '1', 'traumaTrials.npy') 
np.save('trauma_ses_1', trt_ses1)
trt_ses2 = creatSubCor(subject_list, '2', 'traumaTrials.npy')
np.save('trauma_ses_2', trt_ses2)
trt_ses3 = creatSubCor(subject_list, '3', 'traumaTrials.npy')
trt_ses4 = creatSubCor(subject_list, '4', 'traumaTrials.npy')
sad_ses1 = creatSubCor(subject_list, '1', 'sadTrials.npy')
sad_ses2 = creatSubCor(subject_list, '2', 'sadTrials.npy')
sad_ses3 = creatSubCor(subject_list, '3', 'sadTrials.npy')

# comparing the two networks
# reshaping the arrays
trt_ses1_reshape = np.moveaxis(np.array(trt_ses1),0,-1)
trt_ses2_reshape = np.moveaxis(np.array(trt_ses2),0,-1)
trt_ses3_reshape = np.moveaxis(np.array(trt_ses3),0,-1)
sad_ses1_reshape = np.moveaxis(np.array(sad_ses1),0,-1)
# run nbs
pvals_sad_trt_ses1, adj_sad_trt_ses1, _ = nbs.nbs_bct(trt_ses1_reshape, sad_ses1_reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)

# compare the two trauma networks
pvals_trt1_trt2, adj_trt1_trt2, _ = nbs.nbs_bct(trt_ses1_reshape, trt_ses2_reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)
# significant difference between trauma in session 1 and 2. 

# check session 3 also
pvals_trt1_trt3, adj_trt1_trt3, _ = nbs.nbs_bct(trt_ses1_reshape, trt_ses3_reshape, thresh=2.5, tail='both',k=500, paired=True, verbose = True)


#%% Now we should look at amygdala and hippocampus as seed and analyze connectivity before and after treatment for these two
# this is done in seedTovoxel.py script

#%% Run linear regression using sckit learn - check preditction
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df_clinical.describe()
X = np.array(subjectPosEdges)[mask].reshape(-1,1)
y = np.array(diffPCL)[mask].reshape(-1,1)
# set 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression(fit_intercept=True, normalize=True)  
model = regressor.fit(X_train, y_train) #training the algorithm
predictions = regressor.predict(X_test)


## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print (f'Score: {model.score(X_test, y_test)}')

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Positive connectome and change in PCL (train set)')
plt.xlabel('Positive connectome score')
plt.ylabel('PCL change (end of trt to F/U1) ')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Positive connectome and change in PCL (test set)')
plt.xlabel('Positive connectome score')
plt.ylabel('PCL change (end of trt to F/U1) ')
plt.show()

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print("Mean squared error: %.2f" %mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter(X_train, y_train,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
# prediction is low for now
#%% Using statsmodel to do linear regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
df_reg = pd.DataFrame({'pos_edges': np.array(subjectPosEdges)[mask], 'med_cond': df_clinical['med_cond'][mask], 'diffPCL': np.array(diffPCL)[mask]})
X = df_reg[['pos_edges', 'med_cond']]
X = sm.add_constant(X)
y = df_reg['diffPCL']
df_reg.to_csv('df_reg.csv') # save as csv to run analysis on R --> analysis turned out to be the same
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

#%% Ridge
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


lin_reg = LinearRegression(normalize=True)

MSEs = cross_val_score(lin_reg, X, y, scoring='r2', cv=16)

MSEs.mean()
MSEs.std()
mean_MSE = np.mean(MSEs)

print(mean_MSE)
#
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='explained_variance', cv=16)

ridge_regressor.fit(X, y)
ridge_regressor.best_params_
ridge_regressor.best_score_
ridge_regressor.cv_results_['mean_test_score']
# worse score compared to linear

##
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 16)

lasso_regressor.fit(X, y)

lasso_regressor.best_params_
lasso_regressor.best_score_
# performs worse compared to linear regression
##

#%% try with leave one out
import numpy as np
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

#loo.get_n_splits(X)
loo.get_n_splits(np.array(subject_list[0:18])[mask])
# chose subject number
t1 = creatSubCor(subject_list, '1', 'speficTrial_0_trauma.npy')

df_clinical.describe()
X = np.array(subjectPosEdges[0:18])[mask].reshape(-1,1)
y = np.array(diffPCL)[mask]#.reshape(-1,1)
print(loo)

for train_index, test_index in loo.split(np.array(subject_list[0:18])[mask]):
   # should insert matrx tresholding for all subjects picked and calculating (each iteration) the Sum of Positive edges
   print(np.array(subject_list)[train_index])
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X_train,y_train)
  # print(X_train, X_test, y_train, y_test)
   print(f'The R value is: {r_value} and the p = {p_value}')
   # run regression
   
   ##
   


#%% Compare groups
med_group_df = df.filter(['scr_id', 'med_cond'])
ketSubject = med_group_df.loc[med_group_df['med_cond'] == 1.0]

midSubject = med_group_df.loc[med_group_df['med_cond'] == 0.0]
print(f'Number of Ketamine subjects is {len(ketSubject)}')
print(f'Number of midazolam subjects is {len(midSubject)}')

#%% check connectivity between amygdala and hippocampus (92 - 93-95 hippocampus | 228 - 229-232) and see if changes correlates to changes in symptoms
deltMatAmg_Hippo = deltaMatrix_each[:,[92,93,94,95,228,229,230,231,232],:]# [92,93,94,95,228,229,230,231,232]]
vecAmg_92_93 = deltMatAmg_Hippo[:,4,232]
df_clinical['amg92_93'] = vecAmg_92_93[0:18]

mask = ~np.isnan(diffPCL_FU_screen) & ~np.isnan(df_clinical['med_cond'])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_clinical['amg92_93'][mask],diffPCL_FU_screen[mask])
import matplotlib.pyplot as plt
line = slope*np.array(df_clinical['amg92_93'])[mask]+intercept

# plot the regression model
plt.title('Positive sum of correlations - 16 valid subjects')
plt.xlabel('Sum of connectivity')
plt.ylabel('Change in PCL (end of trt to 30 days)')
plt.plot(df_clinical['amg92_93'], diffPCL_FU_screen, 'o',np.array(df_clinical['amg92_93'])[mask], line)
plt.show()

#%% Run the same but now first we z-fisher the matrices before delta them
deltaMatrix_each = np.array(trauma_2nd_ses2) - np.array(trauma_1st_ses)
deltaMat_zfisher = []
for mat2, mat1 in zip(trauma_2nd_ses2, trauma_1st_ses):
    mat1z = np.arctanh(mat1)
    mat2z = np.arctanh(mat2)
    deltaMat = mat2z - mat1z
    deltaMat_zfisher.append(deltaMat)

deltaMatz = np.array(deltaMat_zfisher)


deltMatAmg_Hippo = deltaMatz[:,[92,93,94,95,228,229,230,231,232],:]# [92,93,94,95,228,229,230,231,232]]
vecAmg_92_93 = deltMatAmg_Hippo[:,4,232]
df_clinical['amg92_93'] = vecAmg_92_93[0:18]

mask = ~np.isnan(diffPCL_FU_screen) & ~np.isnan(df_clinical['med_cond'])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_clinical['amg92_93'][mask],diffPCL_FU_screen[mask])
import matplotlib.pyplot as plt
line = slope*np.array(df_clinical['amg92_93'])[mask]+intercept

# plot the regression model
plt.title('Positive sum of correlations - 16 valid subjects')
plt.xlabel('Sum of connectivity')
plt.ylabel('Change in PCL (end of trt to 30 days)')
plt.plot(df_clinical['amg92_93'], diffPCL_FU_screen, 'o',np.array(df_clinical['amg92_93'])[mask], line)
plt.show()

#%% Using statsmodel to do linear regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#df_reg = pd.DataFrame({'pos_edges': np.array(subjectPosEdges)[mask], 'med_cond': df_clinical['med_cond'][mask], 'diffPCL': np.array(diffPCL)[mask]})
X = df_clinical['amg92_93'][mask]
X = sm.add_constant(X)
y = diffPCL_FU_screen[mask]
df_reg.to_csv('df_reg.csv') # save as csv to run analysis on R --> analysis turned out to be the same
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
#%% lets try this with scikit learn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
# set variables
X = df_clinical['amg92_93'][mask]
y = diffPCL_FU_screen[mask]

# run model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression(fit_intercept=True, normalize=True)  
# fit and predict
model = regressor.fit(X_train, y_train) #training the algorithm
predictions = regressor.predict(X_test)


## plot predictions
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

# print score
print (f'Score: {model.score(X_test, y_test)}')

