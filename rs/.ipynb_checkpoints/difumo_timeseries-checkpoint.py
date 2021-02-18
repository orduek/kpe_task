# %%
'''
@author: Or Duek
@date: Jul 16 2020

This is a sciprt that uses the nee DiFuMo dictionary
atlas (https://www.sciencedirect.com/science/article/pii/S1053811920306121#appsec7)

In this file we will create a task based 
'''
# %%
# %config Completer.use_jedi = False

# %% import libraries
import pandas as pd 
from nilearn.input_data import NiftiMapsMasker

from nilearn import datasets
import numpy as np
import nilearn.plotting
from sklearn.model_selection import StratifiedShuffleSplit
import os
import glob
from nilearn import connectome
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
# %% Set output folder
## condition labels (ketamine , midazolam)
# read file
medication_cond = pd.read_csv('/home/oad4/kpe_task/task_based_analysis/kpe_sub_condition.csv')
# remove 1315 (OCD), also remove for 2nd session remove 1578
medication_cond1 = medication_cond[medication_cond.scr_id!='KPE1315'] # for 1st session
medication_cond2 = medication_cond[(medication_cond.scr_id!='KPE1315') & 
                                   (medication_cond.scr_id!='KPE1578')] # for 2nd session
medication_cond3 = medication_cond[(medication_cond.scr_id!='KPE1315') &
                                  (medication_cond.scr_id!='KPE1253') & (medication_cond.scr_id!='KPE1468') &
                                  (medication_cond.scr_id!='KPE1480') & (medication_cond.scr_id!='KPE1587')
                                  ] # for 3rd session

subject_list1 = np.array(medication_cond1.scr_id)
subject_list2 = np.array(medication_cond2.scr_id)
subject_list3 = np.array(medication_cond3.scr_id)
condition_label1 = np.array(medication_cond1.med_cond)
condition_label2 = np.array(medication_cond2.med_cond)
condition_label3 = np.array(medication_cond3.med_cond)
#group_label1 = list(map(int, condition_label))

# %%
medication_cond3

# %%
len(subject_list3)
#print(condition_label2)

# %% fetch atlas
maps_img = '/gpfs/gibbs/pi/levy_ifat/Or/DiFuMo_atlas/256/maps.nii.gz'
labels = pd.read_csv('/gpfs/gibbs/pi/levy_ifat/Or/DiFuMo_atlas/256/labels_256_dictionary.csv')
#coords = nilearn.plotting.find_parcellation_cut_coords(labels_img=maps_img)
coords = nilearn.plotting.find_probabilistic_atlas_cut_coords(maps_img)
# plot atlas (only if we want)
nilearn.plotting.plot_prob_atlas(maps_img, draw_cross=False)
# %% read files and stratify to relevant script
# method to generate subject array of timeseries
def pooledTS(subject_list, ses):
    rs_template = '/gpfs/gibbs/pi/levy_ifat/Or/kpe/results/restingsub-{sub}_ses-{ses}.npy'
    sub_ts = []
    for sub in subject_list:
        subject = sub.split('KPE')[1]
        
        # load the npy file (timeseries)
        ts = np.load(rs_template.format(sub=subject, ses=ses), allow_pickle=True)
        ts_script = ts[5:,:] # remove first five trs
        sub_ts.append(ts_script)
    return sub_ts
# %%
from nilearn import connectome
connectome = connectome.ConnectivityMeasure(
    kind='partial correlation', vectorize=False)

mat_ses1 = connectome.fit_transform(pooledTS(subject_list1, '1'))


# %%
mat_ses1.shape

# %% plot mean matrix
# %matplotlib inline
nilearn.plotting.plot_matrix(connectome.mean_, 
 colorbar=True, labels= range(256), reorder='average')

# %% lets run ses 2
mat_ses2 = connectome.fit_transform(pooledTS(subject_list2, '2'))
# %matplotlib inline
nilearn.plotting.plot_matrix(connectome.mean_, 
 colorbar=True, labels= range(256), reorder='average')

# %%
mat_ses3 = connectome.fit_transform(pooledTS(subject_list3, '3'))

# %%
# fisher-z transformation
mat_ses1 = np.arctan(mat_ses1)
mat_ses2 = np.arctan(mat_ses2)
mat_ses3 = np.arctan(mat_ses3)

# %%
## Generate matrix of just ROIs (amygdala, hippocampus, vmpfc and caudate)
# get index of each ROI

labels_list = list(labels.Difumo_names)
amg = labels_list.index('Amygdala')
hippo_post = labels_list.index('Hippocampus posterior')
hippo_ant = labels_list.index('Hippocampus anterior')
vmPFC_ant = labels_list.index('Ventromedial prefrontal cortex anterior')
vmPFC = labels_list.index('Ventromedial prefrontal cortex')
index_list = np.array([amg, hippo_post, hippo_ant, vmPFC_ant, vmPFC])#, caudate_ant, caudate_inf, caudate_sup])

# set matrix for specific ROIs (as defined above)
mat1ROI =  mat_ses1[: ,index_list,:]
mat1ROI = mat1ROI[:,:,index_list]

mat2ROI = mat_ses2[: ,index_list,:]
mat2ROI = mat2ROI[:,:,index_list]

mat3ROI = mat_ses3[: ,index_list,:]
mat3ROI = mat3ROI[:,:,index_list]



# %%
mat2ROI.shape
labels = ['amygdala','hippoPost','hippoAnt','vmPFCAnt','vmPFC']#,'Ca_Ant','Ca_In','ca_sup']
nilearn.plotting.plot_matrix((np.mean(mat2ROI, axis=0)), 
 colorbar=True, labels= labels, reorder='average')

nilearn.plotting.plot_matrix((np.mean(mat1ROI, axis=0)), 
 colorbar=True, labels= labels, reorder='average')

# %%
# show groups
ketSes1 = mat1ROI[condition_label1==1]
midSes1 = mat1ROI[condition_label1==0]

ketSes2 = mat2ROI[condition_label2==1]
midSes2 = mat2ROI[condition_label2==0]

ketSes3 = mat3ROI[condition_label3==1]
midSes3 = mat3ROI[condition_label3==0]



# %%
midSes3.shape

# %%
## First session
sns.heatmap(np.mean(ketSes1, axis=0), annot=True, 
           xticklabels = labels, yticklabels = labels,
           vmin = -1, vmax = 1, cmap='coolwarm')
plt.title("Ketamine")
plt.show()

sns.heatmap(np.mean(midSes1, axis=0), annot=True, 
           xticklabels = labels, yticklabels = labels,
           vmin = -1, vmax = 1, cmap='coolwarm')
plt.title("Midazolam")

# %%
sns.heatmap(np.mean(ketSes2, axis=0), annot=True, 
           xticklabels = labels, yticklabels = labels,
           vmin = -1, vmax = 1, cmap='coolwarm')
plt.title("Ketamine")
plt.show()

sns.heatmap(np.mean(midSes2, axis=0), annot=True, 
           xticklabels = labels, yticklabels = labels,
           vmin = -1, vmax = 1, cmap='coolwarm')
plt.title("Midazolam")


# %%
sns.heatmap(np.mean(ketSes3, axis=0), annot=True, 
           xticklabels = labels, yticklabels = labels,
           vmin = -1, vmax = 1, cmap='coolwarm')
plt.title("Ketamine")
plt.show()

sns.heatmap(np.mean(midSes3, axis=0), annot=True, 
           xticklabels = labels, yticklabels = labels,
           vmin = -1, vmax = 1, cmap='coolwarm')
plt.title("Midazolam")

# %%
import scipy
t, p = scipy.stats.ttest_ind(ketSes3, midSes3)
tArr = np.array(t)

thr = 0.05
tArr[p>thr] = 0
sns.heatmap(tArr,xticklabels = labels, yticklabels = labels,  cmap='coolwarm', annot=True)

# %%
tArr

# %%
# run simple t test to show whats going on
t, p = scipy.stats.ttest_ind(ketSes2, midSes2)
tArr = np.array(t)
tArr[p>thr] = 0
sns.heatmap(tArr, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)

# %%
# create delta arrays
ketDelta = np.subtract(ketSes2, ketSes1)
# removing 1578 (last one) from midazolam group 
midDelta = np.subtract(midSes2, midSes1[0:11])
sns.heatmap(np.mean(ketDelta, axis=0),
            cmap='coolwarm', xticklabels=labels, 
            yticklabels=labels, annot=True, vmin = -1, vmax = 1)
plt.show()
sns.heatmap(np.mean(midDelta, axis=0), cmap='coolwarm', 
            xticklabels=labels, yticklabels=labels, annot=True, vmin=-1, vmax=1)
plt.show()


# %%
# create delta arrays 3 and one - so need to make everything like 3 for the midazolam group (they have only 9)
mat_sesM1 = connectome.fit_transform(pooledTS(subject_list3, '1'))
mat_sesM1 = np.arctan(mat_sesM1)
mat1ROIM =  mat_sesM1[: ,index_list,:]
mat1ROIM = mat1ROIM[:,:,index_list]
midSes1M = mat1ROIM[condition_label3==0]


ketDelta = np.subtract(ketSes3, ketSes1)
# removing 1578 (last one) from midazolam group 
midDelta = np.subtract(midSes3, midSes1M)
sns.heatmap(np.mean(ketDelta, axis=0),
            cmap='coolwarm', xticklabels=labels, 
            yticklabels=labels, annot=True, vmin = -1, vmax = 1)
plt.show()
sns.heatmap(np.mean(midDelta, axis=0), cmap='coolwarm', 
            xticklabels=labels, yticklabels=labels, annot=True, vmin=-1, vmax=1)
plt.show()


# %%
import mne

t, p = scipy.stats.ttest_ind(ketDelta, midDelta)
tArr = np.array(t)
# vectorize the p to include only the actual
pvec = np.concatenate(np.tril(p))
print(pvec)
fdr = mne.stats.fdr_correction(pvec, alpha=0.8, method='indep')
print(fdr)
tArr[fdr[1]>.05] = 0
#print(tArr)
sns.heatmap(tArr, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)

# %%
# lets run NBS
ketDeltaReshape = np.moveaxis(np.array(ketDelta),0,-1)
midDeltaReshape = np.moveaxis(np.array(midDelta),0,-1)

ketSes2_reshape = np.moveaxis(np.array(ketSes2),0,-1)
midSes2_reshape = np.moveaxis(np.array(midSes2),0,-1)

ketSes3_reshape = np.moveaxis(np.array(ketSes3),0,-1)
midSes3_reshape = np.moveaxis(np.array(midSes3),0,-1)
print(ketDeltaReshape.shape)
print(midDeltaReshape.shape)

# difference between the sessions
mat1ROI_reshape = np.moveaxis(np.array(mat1ROI), 0, -1)
mat3ROI_reshape = np.moveaxis(np.array(mat3ROI), 0, -1)
mat2ROI_reshape = np.moveaxis(np.array(mat2ROI), 0, -1)

from bct import nbs
                              
# we compare ket1 and ket3
pval, adj, _ = nbs.nbs_bct(mat1ROI_reshape, mat2ROI_reshape, thresh=1.5, tail='both',k=1000, 
                           paired=True, verbose = False)
print(pval)

# %%
# ok lets threshold using adjacency
#tTresh = t[np.tril(adj)]
tTresh = t* adj
#tTresh[np.triu(tTresh)] = t
sns.heatmap(tTresh, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)
plt.show()
sns.heatmap(np.mean(ketDelta, axis=0), xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)
plt.show()
sns.heatmap(np.mean(midDelta, axis=0), xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)

# %%
labels

# %%
# built data frame for each session
df_ses1 = pd.DataFrame(
    {'subject': subject_list1, 'group': condition_label1, 'amg_hippPost1': mat1ROI[:,0,1], 'amg_vmPFC1': mat1ROI[:,0,4], 'amg_hippAnt1': mat1ROI[:,0,2],
    'amg_vmPFCant1': mat1ROI[:,0,3], 'hippAnt_vmPFC1': mat1ROI[:,2,4], 'hippPost_vmPFC1': mat1ROI[:,1,4],
    'hippAnt_vmPFCant1': mat1ROI[:,2,3], 'hippPost_vmPFCant1': mat1ROI[:,1,3]}
)

df_ses1.to_csv('RS_conn_ses1.csv', index=False)

# %%
df_ses1

# %%
# built data frame for each session
df_ses2 = pd.DataFrame(
    {'subject': subject_list2, 'group': condition_label2, 'amg_hippPost2': mat2ROI[:,0,1], 
     'amg_vmPFC2': mat2ROI[:,0,4], 'amg_hippAnt2': mat2ROI[:,0,2],
    'amg_vmPFCant2': mat2ROI[:,0,3], 'hippAnt_vmPFC2': mat2ROI[:,2,4], 'hippPost_vmPFC2': mat2ROI[:,1,4],
    'hippAnt_vmPFCant2': mat2ROI[:,2,3], 'hippPost_vmPFCant2': mat2ROI[:,1,3]}
)

df_ses2.to_csv('RS_conn_ses2.csv', index=False)

# %%
# built data frame for each session
df_ses3 = pd.DataFrame(
    {'subject': subject_list3, 'group': condition_label3, 'amg_hippPost3': mat3ROI[:,0,1], 
     'amg_vmPFC3': mat3ROI[:,0,4], 'amg_hippAnt3': mat3ROI[:,0,2],
    'amg_vmPFCant3': mat3ROI[:,0,3], 'hippAnt_vmPFC3': mat3ROI[:,2,4], 'hippPost_vmPFC3': mat3ROI[:,1,4],
    'hippAnt_vmPFCant3': mat3ROI[:,2,3], 'hippPost_vmPFCant3': mat3ROI[:,1,3]}
)

df_ses3.to_csv('RS_conn_ses3.csv', index=False)

# %%
# combine all three
dfAll = pd.merge(df_ses1, df_ses2, how='left')
dfAll = pd.merge(dfAll, df_ses3, how= 'left')
dfAll.to_csv('RS_conn_3Sessions.csv', index=False)
