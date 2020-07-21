'''
@author: Or Duek
@date: Jul 16 2020

This is a sciprt that uses the nee DiFuMo dictionary
atlas (https://www.sciencedirect.com/science/article/pii/S1053811920306121#appsec7)

In this file we will create a task based 
'''
#%% import libraries
import pandas as pd 
from nilearn.input_data import NiftiMapsMasker
from nilearn import connectome
from nilearn import datasets
import numpy as np
import nilearn.plotting
from sklearn.model_selection import StratifiedShuffleSplit
import os
import glob
from nilearn import connectome
import seaborn as sns
#%% Set output folder
output_dir = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_results/DiFuMo/'
# set session
ses= '1' # session is a string

## condition labels (ketamine , midazolam)
# read file
medication_cond = pd.read_csv('/home/or/kpe_task_analysis/task_based_analysis/kpe_sub_condition.csv')
subject_list = np.array(medication_cond.scr_id)
condition_label = np.array(medication_cond.med_cond)

group_label = list(map(int, condition_label))

#%% fetch atlas
maps_img = '/media/Data/work/DiFuMo_atlas/256/maps.nii.gz'
labels = pd.read_csv('/media/Data/work/DiFuMo_atlas/256/labels_256_dictionary.csv')
#coords = nilearn.plotting.find_parcellation_cut_coords(labels_img=maps_img)
coords = nilearn.plotting.find_probabilistic_atlas_cut_coords(maps_img)

#%% read files and stratify to relevant script
connectome = connectome.ConnectivityMeasure(
    
    kind='correlation', vectorize=False)
# set events file template - here we can choose either whole scripts (120seconds each) or just part of
event_template = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_BIDS/condition_files/withNumbers/sub-{sub}_ses-{ses}_30sec_window.csv'

duration = 120 #set duration of event in seconds 
mat = []
for sub in subject_list:
    subject = sub.split('KPE')[1]
    
    # load the npy file (timeseries)
    ts = np.load(output_dir + '/sub-' + subject + '_ses-' + ses + '.npy', allow_pickle=True)
    event = event_template.format(sub=subject, ses=ses)
    events = pd.read_csv(event, sep='\t')
    onset = int(events.onset[events.trial_type_30=='trauma1_0']) # take onset of trauma first script
    ts_script = ts[onset:onset+duration, :]
    mat.append(connectome.fit_transform([ts_script])[0])

mat = np.array(mat)
mat.shape
meanMat = np.mean(mat, axis=0)

# %%
nilearn.plotting.plot_matrix(meanMat, 
 colorbar=True)


# %%
nilearn.plotting.plot_connectome(meanMat, coords,black_bg=True, edge_threshold="99%")

# %%
# Plot stength of edges
## plot strength
nilearn.plotting.plot_connectome_strength(
    meanMat, coords, title='Connectome strength for DiFuMo atlas'
)

## just positive
from matplotlib.pyplot import cm

# plot the positive part of of the matrix
nilearn.plotting.plot_connectome_strength(
    np.clip(meanMat, 0, meanMat.max()), coords, cmap=cm.YlOrRd,
    title='Strength of the positive edges of the DiFuMo correlation matrix'
)

# plot the negative part of of the matrix
nilearn.plotting.plot_connectome_strength(
    np.clip(meanMat, meanMat.min(), 0), coords, cmap=cm.PuBu,
    title='Strength of the negative edges of the DiFuMo correlation matrix'
)
#%%
# Behaviour correlation - get indexes of Amygdala, Hippocampus and vmPFC
labels_list = list(labels.Difumo_names)
amg = labels_list.index('Amygdala')
hippo_post = labels_list.index('Hippocampus posterior')
hippo_ant = labels_list.index('Hippocampus anterior')
vmPFC_ant = labels_list.index('Ventromedial prefrontal cortex anterior')
vmPFC = labels_list.index('Ventromedial prefrontal cortex')
#%% Run through and extract correlation of each edge here
scr_id = []
amg_hippPost = []
amg_hippAnt = []
amg_vmPFC = []
amg_vmPFCant = []
for i, sub in enumerate(subject_list):
    scr_id.append(sub)
    amg_hippPost.append(mat[i,amg,hippo_post])
    amg_hippAnt.append(mat[i,amg,hippo_ant])
    amg_vmPFC.append(mat[i,amg,vmPFC])
    amg_vmPFCant.append(mat[i,amg,vmPFC_ant])
# create dataframe from that
corDF = pd.DataFrame({'scr_id':scr_id, 'group':group_label, 'amg_hippPost': amg_hippPost,
'amg_hippAnt':amg_hippAnt, 'amg_vmPFC':amg_vmPFC, 'amg_vmPFCant': amg_vmPFCant})
# %%
pclDf = pd.read_csv('/home/or/Documents/kpe_analyses/KPEIHR0009_DATA_2019-10-07_1121.csv')
# take only KPE patients
pclDf = pclDf[pclDf['scr_id'].str.startswith('KPE')]
dfP = pd.DataFrame({'subject': pclDf['scr_id']})
dfP_PCL = pclDf[['scr_id','redcap_event_name','pcl5_1', 'pcl5_2', 'pcl5_3', 'pcl5_4', 'pcl5_5', 'pcl5_6', 'pcl5_7',
 'pcl5_8', 'pcl5_9', 'pcl5_10', 'pcl5_11', 'pcl5_12', 'pcl5_13', 'pcl5_14', 'pcl5_15', 'pcl5_16', 'pcl5_17',
 'pcl5_18', 'pcl5_19', 'pcl5_20']]
# remove NAs
dfP_PCL = dfP_PCL.dropna()
# set list of columns for analysis
colList = list(dfP_PCL)
colList.remove('scr_id')
colList.remove('redcap_event_name')
# set total pcl scores 
dfP_PCL['pclTotal'] = dfP_PCL[colList].sum(axis=1)
sns.distplot(dfP_PCL.pclTotal)
# reshape it to wide
df2=dfP_PCL.pivot(index = 'scr_id',columns='redcap_event_name', values='pclTotal')
list(df2)
df2 = df2.rename(columns={"30_day_follow_up_s_arm_1": "30Days", "90_day_follow_up_s_arm_1": "90Days",
                    "screening_selfrepo_arm_1": "Screening", "visit_1_arm_1": "Visit1", 
                    "visit_7_week_follo_arm_1": "Visit7"})
#df2['scr_id'] = dfP_PCL['scr_id']
df2
# %%
# merging two data frames toghether
dfTest = pd.merge(corDF, df2, on = 'scr_id')
# create difference pcl score
dfTest['days30_1'] = dfTest['30Days'] - dfTest.Visit1
dfTest['days30_s'] = dfTest['30Days'] - dfTest.Screening
dfTest['days7_1'] = dfTest['Visit7'] - dfTest.Visit1
dfTest

# %%
import scipy
sns.lmplot(x='amg_hippPost',y='days30_s',hue='group', data=dfTest)
naMask = np.isnan(dfTest['days30_s'])
scipy.stats.pearsonr(dfTest['days30_s'][~naMask], dfTest['amg_hippPost'][~naMask])

# %%
sns.lmplot(x='amg_vmPFC',y='30Days',hue='group', data=dfTest)
naMask = np.isnan(dfTest['30Days'])
scipy.stats.pearsonr(dfTest['30Days'][~naMask], dfTest['amg_vmPFC'][~naMask])


#####
#%% Now we should check the delta in all of these association and the correlation to 
# symptoms change
