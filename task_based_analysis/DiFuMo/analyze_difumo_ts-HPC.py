# %%
'''
@author: Or Duek
@date: Jul 16 2020

This is a sciprt that uses the nee DiFuMo dictionary
atlas (https://www.sciencedirect.com/science/article/pii/S1053811920306121#appsec7)

In this file we will create a task based 
'''
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
# %% Set output folder
output_dir = '/gpfs/gibbs/pi/levy_ifat/Or/kpe/DiFuMo/'
# set session

## condition labels (ketamine , midazolam)
# read file
medication_cond = pd.read_csv('/home/oad4/kpe_task/task_based_analysis/kpe_sub_condition.csv')
subject_list = np.array(medication_cond.scr_id)
condition_label = np.array(medication_cond.med_cond)

group_label = list(map(int, condition_label))

# %%
subject_list = subject_list[0:24] # removing 1578

# %% fetch atlas
maps_img = '/media/Data/work/DiFuMo_atlas/256/maps.nii.gz'
labels = pd.read_csv('/media/Data/work/DiFuMo_atlas/256/labels_256_dictionary.csv')
#coords = nilearn.plotting.find_parcellation_cut_coords(labels_img=maps_img)
coords = nilearn.plotting.find_probabilistic_atlas_cut_coords(maps_img)
# plot atlas (only if we want)
nilearn.plotting.plot_prob_atlas(maps_img, draw_cross=False)
# %% read files and stratify to relevant script
# method to generate subject array of timeseries
def pooledTS(subject_list, ses):
    event_template = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_BIDS/condition_files/withNumbers/sub-{sub}_ses-{ses}_30sec_window.csv'
    duration = 60 #set duration of event in seconds 
    sub_ts = []
    for sub in subject_list:
        subject = sub.split('KPE')[1]
        
        # load the npy file (timeseries)
        ts = np.load(output_dir + '/sub-' + subject + '_ses-' + ses + '.npy', allow_pickle=True)
        event = event_template.format(sub=subject, ses=ses)
        events = pd.read_csv(event, sep='\t')
        onset = int(events.onset[events.trial_type_30=='trauma1_0']) # take onset of trauma first script
        ts_script = ts[onset:onset+duration, :]
        sub_ts.append(ts_script)
    return sub_ts
# %%
from nilearn import connectome
connectome = connectome.ConnectivityMeasure(
    kind='correlation', vectorize=False)

mat_ses1 = connectome.fit_transform(pooledTS(subject_list, '1'))


# %% plot mean matrix
nilearn.plotting.plot_matrix(connectome.mean_, 
 colorbar=True, labels= range(256), reorder='average')

# %% lets run ses 2
mat_ses2 = connectome.fit_transform(pooledTS(subject_list, '2'))

nilearn.plotting.plot_matrix(connectome.mean_, 
 colorbar=True, labels= range(256), reorder='average')

# %%
nilearn.plotting.plot_connectome(connectome.mean_, coords,black_bg=False, edge_threshold="99.5%")

# %%
# Plot stength of edges
## plot strength
nilearn.plotting.plot_connectome_strength(
    connectome.mean_, coords, title='Connectome strength for DiFuMo atlas'
)

## just positive
from matplotlib.pyplot import cm

# plot the positive part of of the matrix
nilearn.plotting.plot_connectome_strength(
    np.clip(connectome.mean_, 0, connectome.mean_.max()), coords, cmap=cm.YlOrRd,
    title='Strength of the positive edges of the DiFuMo correlation matrix'
)

# plot the negative part of of the matrix
nilearn.plotting.plot_connectome_strength(
    np.clip(connectome.mean_, connectome.mean_.min(), 0), coords, cmap=cm.PuBu,
    title='Strength of the negative edges of the DiFuMo correlation matrix'
)

# %%
# fisher-z transformation
mat_ses1 = np.arctan(mat_ses1)
mat_ses2 = np.arctan(mat_ses2)

# %%
## Generate matrix of just ROIs (amygdala, hippocampus, vmpfc and caudate)
# get index of each ROI
labels = pd.read_csv('/media/Data/work/DiFuMo_atlas/256/labels_256_dictionary.csv')
labels_list = list(labels.Difumo_names)
amg = labels_list.index('Amygdala')
hippo_post = labels_list.index('Hippocampus posterior')
hippo_ant = labels_list.index('Hippocampus anterior')
vmPFC_ant = labels_list.index('Ventromedial prefrontal cortex anterior')
vmPFC = labels_list.index('Ventromedial prefrontal cortex')
caudate_inf = labels_list.index('Caudate inferior')
caudate_ant = labels_list.index('Caudate anterior')
caudate_sup = labels_list.index('Caudate superior')
index_list = np.array([amg, hippo_post, hippo_ant, vmPFC_ant, vmPFC])#, caudate_ant, caudate_inf, caudate_sup])

mat2ROI = mat_ses2[: ,index_list,:]
mat2ROI = mat2ROI[:,:,index_list]

mat1ROI =  mat_ses1[: ,index_list,:]
mat1ROI = mat1ROI[:,:,index_list]

# %%
mat2ROI.shape
labels = ['amygdala','hippoPost','hippoAnt','vmPFCAnt','vmPFC']#,'Ca_Ant','Ca_In','ca_sup']
nilearn.plotting.plot_matrix((np.mean(mat2ROI, axis=0)), 
 colorbar=True, labels= labels, reorder='average')

nilearn.plotting.plot_matrix((np.mean(mat1ROI, axis=0)), 
 colorbar=True, labels= labels, reorder='average')

# %%
# show groups
group_label = np.array(group_label[0:24])
ketSes2 = mat2ROI[group_label==1]
midSes2 = mat2ROI[group_label==0]

ketSes1 = mat1ROI[group_label==1]
midSes1 = mat1ROI[group_label==0]

# %%
group_label

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
np.mean(ketSes1, axis=0)
# get 5-95% percentiles for amg-hippPost
amgH = ketSes2[:,3,2]

np.percentile(amgH,[.05,95],  axis=0)
#amgH

# %%
amgH = midSes2[:,3,2]

np.percentile(amgH,[.05,95],  axis=0)
#amgH

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
import scipy
t, p = scipy.stats.ttest_ind(ketSes2, midSes2)
tArr = np.array(t)
thr = 0.05
tArr[p>thr] = 0
sns.heatmap(tArr, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)

# %%
p

# %%
# now compare difference between ses1 and 2 for those groups
# divide matrix of 1ses to groups
#group_label = group_label[0:24]
ketSes1 = mat1ROI[group_label==1]
midSes1 = mat1ROI[group_label==0]

# run simple t test to show whats going on
t, p = scipy.stats.ttest_ind(ketSes1, midSes1)
tArr = np.array(t)
tArr[p>thr] = 0
sns.heatmap(tArr, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)

# %%
# create delta arrays
ketDelta = np.subtract(ketSes2, ketSes1)
midDelta = np.subtract(midSes2, midSes1)
sns.heatmap(np.mean(ketDelta, axis=0),
            cmap='coolwarm', xticklabels=labels, 
            yticklabels=labels, annot=True, vmin = -1, vmax = 1)
plt.show()
sns.heatmap(np.mean(midDelta, axis=0), cmap='coolwarm', 
            xticklabels=labels, yticklabels=labels, annot=True, vmin=-1, vmax=1)
plt.show()


# %%
t, p = scipy.stats.ttest_ind(ketDelta, midDelta)
tArr = np.array(t)
fdr = sm.multitest.fdrcorrection(p, alpha=0.05, method='indep', is_sorted=True)
#fdr = sm.multitest.multipletests(pvec, alpha=thr, method='fdr_bh')
print(fdr[0])
tArr[fdr[1]>.05] = 0
print(tArr)
sns.heatmap(tArr, xticklabels=labels, yticklabels=labels, cmap='coolwarm', annot=True)

# %%
print(1/10*0.30)
p

# %%
# lets run NBS
ketDeltaReshape = np.moveaxis(np.array(ketDelta),0,-1)
midDeltaReshape = np.moveaxis(np.array(midDelta),0,-1)

ketSes2_reshape = np.moveaxis(np.array(ketSes2),0,-1)
midSes2_reshape = np.moveaxis(np.array(midSes2),0,-1)
print(ketDeltaReshape.shape)
print(midDeltaReshape.shape)
from bct import nbs
                              
# we compare ket1 and ket3
pval, adj, _ = nbs.nbs_bct(ketDeltaReshape, midDeltaReshape, thresh=2.3, tail='both',k=1000, 
                           paired=False, verbose = False)
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
sns.barplot()

# %% [markdown]
# ## Create a dataframe to test correlation between amygdala, hippocampus (vmPFC) in groups and session (1,2)

# %%
dfCors = pd.DataFrame({'amg_hippPost2': mat2ROI[:,0,1], 'amg_vmPFC2': mat2ROI[:,0,4],
                         'amg_hippPost1': mat1ROI[:,0,1], 'amg_vmPFC1': mat1ROI[:,0,4],
                       'amg_hippAnt2': mat2ROI[:,0,2], 'amg_HippAnt1': mat1ROI[:,0,2],
                       'amg_vmPFCAnt2': mat2ROI[:,0,3], 'amg_vmPFCAnt1': mat1ROI[:,0,3],
                       'hippAnt_vmPFCAnt2' : mat2ROI[:,2,3], 
                       'hippAnt_vmPFCAnt1' : mat1ROI[:,2,3], 
                       'hippAnt_hippPost1': mat1ROI[:, 1,2], 
                       'hippAnt_hippPost2': mat2ROI[:, 1,2]
                      })
dfCors['groupIdx'] = group_label[0:24]
dfCors['amg_hipp_change'] = dfCors.amg_hippPost2 - dfCors.amg_hippPost1
dfCors['amg_hippAnt_change'] = dfCors.amg_hippAnt2 - dfCors.amg_HippAnt1
dfCors['amg_vmpfcAnt_change'] = dfCors.amg_vmPFCAnt2 - dfCors.amg_vmPFCAnt1
dfCors['amg_vmpfc_change'] = dfCors.amg_vmPFC2 - dfCors.amg_vmPFC1
dfCors['hippoAnt_vmpfcAnt_change'] = dfCors.hippAnt_vmPFCAnt2 - dfCors.hippAnt_vmPFCAnt1
dfCors['hippoAnt_hippPost_change'] = dfCors.hippAnt_hippPost2 - dfCors.hippAnt_hippPost1

dfCors

# %%
# add group condition
group = {1:'ketamine', 0:'midazolam'} 
dfCors['group'] =[group[item] for item in dfCors.groupIdx] 

# %%
## create plot for publication
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5),gridspec_kw={'wspace': .05})
g1 = sns.boxplot(y = 'amg_hipp_change', x= 'group', data=dfCors, ax=ax1,
                boxprops=dict(alpha=.4))
sns.stripplot(y = 'amg_hipp_change', x= 'group', data=dfCors,size=8, ax=ax1)
ax1.text(-.4,0.8, "Amygdala-Posterior Hippocampus")
g2 = sns.boxplot(y = 'hippoAnt_vmpfcAnt_change', x= 'group', data=dfCors, ax=ax2,
                boxprops=dict(alpha=.4))
sns.stripplot(y = 'hippoAnt_vmpfcAnt_change', x= 'group', data=dfCors, size=8, ax=ax2)
ax2.text(-.4,0.8, "Anterior Hippocampus - vmPFC")
# g3 = sns.boxplot(y = 'hippoAnt_hippPost_change', x= 'group', data=dfCors, ax=ax3,
#                 boxprops=dict(alpha=.4))
# sns.stripplot(y = 'hippoAnt_hippPost_change', x= 'group', data=dfCors,size=8, ax=ax3)
# ax3.text(-.4,0.8, "Anterior-Posterior Hippocampus")
ylow = -0.9
yhigh=0.9
ax1.set_ylim(ylow,yhigh)
ax2.set_ylim(ylow,yhigh)
#ax3.set_ylim(ylow,yhigh)
ax1.set_xlabel("")
ax2.set_xlabel("")
#ax3.set_xlabel("")
ax2.set_yticks([])
#ax3.yaxis.tick_right()
ax1.set_ylabel("Difference before/after treatment", fontsize=14)
ax2.set_ylabel("")
#ax3.set_ylabel("")
ax1.set_xticklabels(['Ketamine', 'Midazolam'], fontsize=14)
ax2.set_xticklabels(['Ketamine', 'Midazolam'], fontsize=14)
#ax3.set_xticklabels(['Ketamine', 'Midazolam'], fontsize=14)
fig.savefig("changeCorrelation.png", dpi=300,  bbox_inches='tight')

# %% [markdown]
# ## Use PyMC3 to compare the difference in correlation

# %%
# Using Pymc3
import pymc3 as pm
from pymc3.glm import GLM

with pm.Model() as model_glm:
    GLM.from_formula('amg_hipp_change ~ groupIdx', dfCors)
    trace = pm.sample(draws=4000, tune=3000)

# %%
pm.summary(trace, credible_interval=.95).round(2)

# %%
# Using Pymc3 - compare antrior hippo and antvmpfc
with pm.Model() as model_glm2:
    GLM.from_formula('hippoAnt_vmpfcAnt_change ~ groupIdx', dfCors)
    trace2 = pm.sample(draws=4000, tune=2000)

# %%
pm.summary(trace2, credible_interval=.95).round(2)

# %%

# %%
sns.boxplot(y = 'amg_hipp_change', x= 'groupIdx', data=dfCors)
scipy.stats.ttest_ind(dfCors.amg_hipp_change[dfCors.groupIdx==1],dfCors.amg_hipp_change[dfCors.groupIdx==0])

# %%
sns.boxplot(y = 'hippoAnt_vmpfcAnt_change', x= 'groupIdx', data=dfCors)
scipy.stats.ttest_ind(dfCors.hippoAnt_vmpfcAnt_change[dfCors.groupIdx==1],
                      dfCors.hippoAnt_vmpfcAnt_change[dfCors.groupIdx==0])

# %%
sns.boxplot(y = 'hippoAnt_hippPost_change', x= 'groupIdx', data=dfCors)
scipy.stats.ttest_ind(dfCors.hippoAnt_hippPost_change[dfCors.groupIdx==1],dfCors.hippoAnt_hippPost_change[dfCors.groupIdx==0])

# %%
# lets plot that on the brain
coordsROI = coords[index_list, :]

#nilearn.plotting.plot_connectome(tTresh,coordsROI)
nilearn.plotting.view_connectome(tTresh,coordsROI)

# %% [markdown]
# ## calculating correlations with amygdala for behavioral correlations

# %% Run through and extract correlation of each edge here
labels = pd.read_csv('/media/Data/work/DiFuMo_atlas/256/labels_256_dictionary.csv')
def makeConnDF(mat, subject_list, labels_list = list(labels.Difumo_names)):
    # takes array (Nsubject X Nodes X Nodes) and returns a dataframe of connectivity between 
    # inputs: mat = array (subjectXNodesXNodes)
    # subject list
    # labels list (Difumo atlas)
    # Amygdala, hippocampus (posterior/anterior), vmPFC (and antrior), caudate (inferior, superior and anterior)
    # 
    # Behaviour correlation - get indexes of Amygdala, Hippocampus and vmPFC
    
    amg = labels_list.index('Amygdala')
    hippo_post = labels_list.index('Hippocampus posterior')
    hippo_ant = labels_list.index('Hippocampus anterior')
    vmPFC_ant = labels_list.index('Ventromedial prefrontal cortex anterior')
    vmPFC = labels_list.index('Ventromedial prefrontal cortex')
    caudate_inf = labels_list.index('Caudate inferior')
    caudate_ant = labels_list.index('Caudate anterior')
    caudate_sup = labels_list.index('Caudate superior')
    scr_id = []
    amg_hippPost = []
    amg_hippAnt = []
    amg_vmPFC = []
    amg_vmPFCant = []
    amg_caudInf = []
    amg_caudAnt = []
    amg_caudSup = []
    for i, sub in enumerate(subject_list):
        scr_id.append(sub)
        amg_hippPost.append(mat[i,amg,hippo_post])
        amg_hippAnt.append(mat[i,amg,hippo_ant])
        amg_vmPFC.append(mat[i,amg,vmPFC])
        amg_vmPFCant.append(mat[i,amg,vmPFC_ant])
        amg_caudInf.append(mat[i, amg, caudate_inf])
        amg_caudAnt.append(mat[i, amg, caudate_ant])
        amg_caudSup.append(mat[i, amg, caudate_sup])
    # create dataframe from that
    corDF = pd.DataFrame({'scr_id':scr_id, 'group':group_label, 'amg_hippPost': amg_hippPost,
    'amg_hippAnt':amg_hippAnt, 'amg_vmPFC':amg_vmPFC, 'amg_vmPFCant': amg_vmPFCant,
    'amg_caudAnt': amg_caudAnt, 'amg_caudInf': amg_caudInf, 'amg_caudSup': amg_caudSup})
    return corDF
# %%
labels

# %%
pclDf = pd.read_csv('/home/or/Documents/kpe_analyses/KPEIHR0009_DATA_2020-08-31_1301.csv')
# take only KPE patients
pclDf['scr_id'] = pclDf['scr_id'].str.replace(" ","")
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
df2 = df2.rename(columns={"30_day_follow_up_s_arm_1": "Days30", "90_day_follow_up_s_arm_1": "90Days",
                    "screening_selfrepo_arm_1": "Screening", "visit_1_arm_1": "Visit1", 
                    "visit_7_week_follo_arm_1": "Visit7"})
#df2['scr_id'] = dfP_PCL['scr_id']
df2


# %% Call makeDF function on session 1 and 2
dfSes1 = makeConnDF(mat_ses1, subject_list)

# %% ses 2
dfSes2 = makeConnDF(mat_ses2, subject_list)




# %%
# merging two data frames toghether
dfTest = pd.merge(dfSes2, df2, on = 'scr_id')
# create difference pcl score
dfTest['days30_1'] = dfTest['Days30'] - dfTest.Visit1
dfTest['days30_s'] = dfTest['Days30'] - dfTest.Screening
dfTest['days7_1'] = dfTest['Visit7'] - dfTest.Visit1
dfTest

# %%
import scipy
sns.lmplot(x='amg_hippAnt',y='Days30',hue='group', data=dfTest)
naMask = np.isnan(dfTest['Days30'])
scipy.stats.pearsonr(dfTest['Days30'][~naMask], dfTest['amg_hippAnt'][~naMask])

# %%
#compare correlations of only ketamine and only midazolam
ketCorr = scipy.stats.pearsonr(dfTest['Visit7'][~naMask][dfTest.group==1], dfTest['amg_hippPost'][~naMask][dfTest.group==1])
midCorr = scipy.stats.pearsonr(dfTest['Visit7'][~naMask][dfTest.group==0], dfTest['amg_hippPost'][~naMask][dfTest.group==0])

from corrstats import independent_corr
checkCorr = independent_corr(ketCorr[0], midCorr[0], n=11, n2 = 10, twotailed=True, conf_level=0.95, method='fisher')
print(f'Correlation difference between CC and PTSD with anhedonia is {checkCorr}')


# %%
sns.lmplot(x='amg_vmPFC',y='Visit7',hue='group', data=dfTest)
naMask = np.isnan(dfTest['Visit7'])
scipy.stats.pearsonr(dfTest['Visit7'][~naMask], dfTest['amg_vmPFC'][~naMask])

# %% Caudate?
sns.lmplot(x='amg_caudSup',y='Days30',hue='group', data=dfTest)
naMask = np.isnan(dfTest['Days30'])
scipy.stats.pearsonr(dfTest['Days30'][~naMask], dfTest['amg_caudSup'][~naMask])

#####
# %% [markdown]
# ### So it seems like there's a general positive correlation between symptoms at 30 days and connectivity between amgygdala and 
# ### hippocampus. While we see a general negative correlation between amg-vmPFC connectivity. 
# #### But - it seems like we have group differences - lets check the interaction of group and each of them

# %%
# amg and hippocampus
import statsmodels.formula.api as smf

model = smf.ols(formula='Visit7 ~ group * amg_hippPost', data=dfTest)
res = model.fit()
print(res.summary())

# %% [markdown]
# ### Indeed we see an interaction between the group and the correlation. 

# %% Now for vmPFC
modelvmPFC = smf.ols(formula='days30_scale ~ group * amg_vmPFC', data=dfTest)
resVMpfc = modelvmPFC.fit()
print(resVMpfc.summary())

# %% Now we should check the delta in all of these association and the correlation to
# Using Pymc3
import pymc3 as pm
from pymc3.glm import GLM

with pm.Model() as model_glm:
    GLM.from_formula('days30_scale ~ group * scaleamgHipp', dfTest)
    trace = pm.sample(draws=4000, tune=3000)

# %%
pm.summary(trace, credible_interval=.95).round(2)

# %%
# lets scale everything
dfTest['scaleamgHipp'] = (dfTest.amg_hippPost - dfTest.amg_hippPost.mean()) / dfTest.amg_hippPost.std()


# %%
dfTest['days30_scale'] = (dfTest.Days30 - dfTest.Days30.mean()) / dfTest.Days30.std()
dfTest['Visit7_scale'] = (dfTest.Visit7 - dfTest.Visit7.mean()) / dfTest.Visit7.std()

# %%
# Creating a delta between connectivity of second - first session
dfTest['amg_HippPost_Change'] = dfSes2.amg_hippPost - dfSes1.amg_hippPost

# %%
sns.lmplot(x='amg_HippPost_Change',y='days30_1',hue='group', data=dfTest)
naMask = np.isnan(dfTest['days30_1'])
scipy.stats.pearsonr(dfTest['days30_1'][~naMask], dfTest['amg_HippPost_Change'][~naMask])

# %%
model_delta = smf.ols(formula='days30_1 ~ group *amg_HippPost_Change', data=dfTest)
resdelta = model_delta.fit()
print(resdelta.summary())


# %%
# check changes in connectivity between amygdala and hippocampus
sns.stripplot(y='amg_HippPost_Change', x='group', data=dfTest)
scipy.stats.ttest_ind(dfTest['amg_hippPost'][~naMask][dfTest.group==0], dfTest['amg_hippPost'][~naMask][dfTest.group==1])

# %% [markdown]
# ## Check correlation with SCR

# %%
scr = pd.read_csv('/home/or/kpe_task_analysis/scr_deltas.csv')
scr1 = scr.drop(columns = ['med_cond', 'groupIdx'])
scr1

# %%
dfMerge = pd.merge(dfTest, scr1)
dfMerge


# %%
sns.lmplot('amg_hippPost', 'Trauma_2vs1',hue='group',data=dfMerge)
#scipy.stats.pearsonr(dfMerge.days7_1, dfMerge.Trauma_2vs1)
