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
subject_list =['1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464']
subject_list3 =['1223','1263','1293','1307','1322','1339','1343','1356','1364','1369','1387','1390','1403','1464'] # subjects who has 3 scans '1351',

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
sessoin2 = list(np.load('session_2TimeseriesShen.npy', allow_pickle = True)) # loading the saved array
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
    # or read by trial type and create matrix per trial type
    else:
        print ("Need to run by task")
    # extract subject specific timeseries from 3D array

#%% Create subject's files per line
# first - move to relevant folder
os.chdir('/home/or/kpe_conn/ShenParc/session_2')
# run loop
for sub in subject_list:
    events_files = '/media/Data/PTSD_KPE/condition_files/sub-%s_ses-2.csv' %sub
    timeseries = '/home/or/kpe_conn/ShenParc/session_2/sub-%s_session-2_timeseries.npy' %sub
    subject_id = sub
    stratifyTimeseries(events_files,timeseries,sub,0)
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
    # takes subject list, session number and filename of file name and returnt 3D array of subject (0 axis) correlation matrix
    correlation_allSubs = []
    for sub in subject_list:
        print(sub)
        timeseries = np.load('/home/or/kpe_conn/ShenParc/session_%s/subject_%s/%s'%(session,sub,fileName) )
        correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
        correlation_allSubs.append(correlation_matrix)
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
pos_corr , neg_corr = pos_neg(trauma_2ndSes_corr)
np.savetxt("pos_cor_TraumSes2.csv" , pos_corr, delimiter= ",")
np.savetxt("neg_cor_TraumSes2.csv" , neg_corr, delimiter= ",")

# check difference between first and second script:
deltaMatrix = np.mean(np.array(trauma_2_1st_ses), axis = 0) - np.mean(np.array(trauma_1st_ses), axis = 0) # contrast the two matrices
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
G = nx.from_numpy_array(trauma_0_corr)
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
session = '2'
fileName = 'speficTrial_2_sad.npy'
