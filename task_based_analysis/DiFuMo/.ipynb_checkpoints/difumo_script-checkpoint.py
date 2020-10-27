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
from nilearn import connectome
from nilearn import datasets
import numpy as np
import nilearn.plotting
from sklearn.model_selection import StratifiedShuffleSplit
import os
import glob
from nilearn import connectome
import seaborn as sns
import dask
# %% Set output folder
output_dir = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_results/DiFuMo/'
# set session
ses= '1' # session is a string
# %% Functions
# extract RS data and create vector for each subject
def removeVars (confoundFile):
    # this method takes the csv regressors file (from fmriPrep) and chooses a few to confound. You can change those few
    import pandas as pd
    confound = pd.read_csv(confoundFile,sep="\t", na_values="n/a")
    finalConf = confound[['csf', 'white_matter', 'framewise_displacement', 
                          'a_comp_cor_00', 'a_comp_cor_01',	'a_comp_cor_02', 'a_comp_cor_03', 
                          'a_comp_cor_04', 'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 
                          'rot_x', 'rot_y', 'rot_z']] # can add 'global_signal' also ,,
                          # 
     # change NaN of FD to zero
    finalConf = np.array(finalConf.fillna(0.0))
    return finalConf


# %% functional files
subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369'
                 ,'1387','1390','1403','1464', '1468', '1480', '1499']

func_template = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-Memory_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confound_template = '/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_BIDS/derivatives/fmriprep/sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-Memory_desc-confounds_regressors.tsv'

## condition labels (ketamine , midazolam)
# read file
medication_cond = pd.read_csv('/home/or/kpe_task_analysis/task_based_analysis/kpe_sub_condition.csv')
subject_list = np.array(medication_cond.scr_id)
condition_label = np.array(medication_cond.med_cond)

group_label = list(map(int, condition_label))
# %%
# create a mean mask of all subjects
# load mask of brain


brainmasks = glob.glob('/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_BIDS/derivatives/fmriprep/sub-*/ses-%s/func/sub-*_ses-%s_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz' %(ses,ses))
print(brainmasks)
# %matplotlib inline
#for mask in brainmasks:
 #   nilearn.plotting.plot_roi(mask)
    
mean_mask = nilearn.image.mean_img(brainmasks)
#nilearn.plotting.plot_stat_map(mean_mask)
group_mask = nilearn.image.math_img("a>=0.95", a=mean_mask)
#nilearn.plotting.plot_roi(group_mask)

# %% fetch atlas
maps_img = '/media/Data/work/DiFuMo_atlas/256/maps.nii.gz'
labels = pd.read_csv('/media/Data/work/DiFuMo_atlas/256/labels_256_dictionary.csv')
#coords = nilearn.plotting.find_parcellation_cut_coords(labels_img=maps_img)
coords = nilearn.plotting.find_probabilistic_atlas_cut_coords(maps_img)
# generate time series
#
mask_params = { 'mask_img': group_mask,
               'detrend': True, 'standardize': True,
               'high_pass': 0.01, 'low_pass': 0.1, 't_r': 1,
               'smoothing_fwhm': 6.,
                'verbose': 5}

masker = NiftiMapsMasker(maps_img=maps_img, **mask_params)

# %% Generate npy files of timeseries for each subject per session
# we will use it later on, stratify to scripts etc.
# build a specific folder
try:
    os.makedirs(output_dir)
except:
    print('Folder already exist')

subject_ts = []
for sub in subject_list:
    print(f' Analysing subject {sub}')
    subject = sub.split('KPE')[1]
    func = func_template.format(sub=subject, session=ses)
    confound = confound_template.format(sub=subject, session=ses)
    signals = masker.fit_transform(imgs=func, confounds=removeVars(confound))
    save = np.save(output_dir + 'sub-' + subject + '_ses-' + ses, signals)
    subject_ts.append(signals)



