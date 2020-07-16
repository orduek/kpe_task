```
@author: Or Duek
@date: Jul 16 2020

This is a sciprt that uses the nee DiFuMo dictionary atlas (https://www.sciencedirect.com/science/article/pii/S1053811920306121#appsec7)

```
#%% import libraries
import pandas as pd 
from nilearn.input_data import NiftiMapsMasker
from nilearn import connectome
from nilearn import datasets
import numpy as np
import nilearn.plotting
from sklearn.model_selection import StratifiedShuffleSplit
#%% Functions
# extract RS data and create vector for each subject
def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['csf','white_matter', 'framewise_displacement', 'dvars', 'std_dvars',
                          'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                        ]] # can add 'global_signal' also ,
     # change NaN of FD to zero
    finalConf = np.array(finalConf.fillna(0.0))
    #finalConf[0,2] = 0 # if removing FD than should remove this one also
    return finalConf


#%% functional files
# subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369'
#                 ,'1387','1390','1403','1464', '1468', '1480', '1499']

func_template = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confound_template = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-rest_desc-confounds_regressors.tsv'

## condition labels (ketamine , midazolam)
# read file
medication_cond = pd.read_csv('/home/or/kpe_task_analysis/task_based_analysis/kpe_sub_condition.csv')
subject_list = np.array(medication_cond.scr_id)
condition_label = np.array(medication_cond.med_cond)

#%%
# Fetch grey matter mask from nilearn shipped with 
# ICBM templates - should consider changing it to mean image
# of our dataset
#gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)

# condition (medication condition)


#%% fetch atlas
maps_img = '/media/Data/work/DiFuMo_atlas/256/maps.nii.gz'
labes = pd.read_csv('/media/Data/work/DiFuMo_atlas/256/labels_256_dictionary.csv')
#coords = nilearn.plotting.find_parcellation_cut_coords(labels_img=maps_img)

# generate time series
#
mask_params = { 'mask_img': gm_mask,
               'detrend': True, 'standardize': True,
               'high_pass': 0.01, 'low_pass': 0.1, 't_r': 1,
               'smoothing_fwhm': 6., 'verbose': 5}

masker = NiftiMapsMasker(maps_img=maps_img, **mask_params)

#%%
subject_ts = []
ses = '2'
for sub in subject_list:
    print(f' Analysing subject {sub}')
    subject = sub.split('KPE')[1]
    func = func_template.format(sub=subject, session=ses)
    confound = confound_template.format(sub=subject, session=ses)
    signals = masker.fit_transform(imgs=func, confounds=removeVars(confound))
    subject_ts.append(signals)

#%% generate connectivity matrix
from sklearn.covariance import LedoitWolf
connectome_measure = connectome.ConnectivityMeasure(
    cov_estimator=LedoitWolf(assume_centered=True),
    kind='partial correlation', vectorize=True)

# Vectorized connectomes across subject-specific timeseries
vec = connectome_measure.fit_transform(subject_ts)




 # %%
group_label = list(map(str, condition_label))
len(group_label)
# %% XGboost

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

model = XGBClassifier(n_jobs=8, 
                     verbose = 9, random_state=None)


#%% Run cross validation
cv = StratifiedShuffleSplit(n_splits=20, test_size=0.25,
                            random_state=0)
scores = cross_val_score(model, vec, group_label,
                         scoring='roc_auc', cv=cv)

#%% permutation scores
from sklearn.model_selection import permutation_test_score
score, permutation_scores, pvalue = permutation_test_score(
    model, vec, group_label, scoring="accuracy", cv=cv, n_permutations=500,
     n_jobs=8, verbose=5)

print("Classification score %s (pvalue : %s)" % (score, pvalue))

#%%
score