#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:10:24 2019

@author: Or DUek
analyzing resutls from SPM 1st level analysis using FSL randomize
It uses TFCE for correction which is a recommended approach today
Variation from:
    https://github.com/poldrack/fmri-analysis-vm/blob/master/analysis/postFMRIPREPmodelling/First%20and%20Second%20Level%20Modeling%20(SPM).ipynb
"""

import os
from nilearn.plotting import plot_glass_brain
from nilearn.plotting import plot_stat_map
import nilearn.plotting
import glob


# grab all spm T images
spmTimages = glob.glob('/media/Data/work/datasink/1stLevel/_subject_id_*/spmT_000*.nii')
print(spmTimages)

# show all results (all T images per subject)
for con_image in spmTimages:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(con_image, 8),
                                      display_mode='lyrz', colorbar=True, plot_abs=False, threshold=2.3)


# grab all contrast images
con_images = glob.glob('/media/Data/work/datasink/1stLevel/_subject_id_*/con_0001.nii')
for con_image in con_images:
    nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(con_image, 8),
                                      display_mode='lyrz', colorbar="w", plot_abs=False)



#%% Now we grab al the contrasts per subject
subjectList = ['1063' , '1072', '1206', '1244','1273' ,'1291', '1305', '1340', '1345', '1346']

copes = {}
ess = {}
for i in subjectList:
    con_images = glob.glob('/media/Data/work/datasink/1stLevel/_subject_id_' + i + '/con_000*.nii')
    ess_images = glob.glob('/media/Data/work/datasink/1stLevel/_subject_id_' + i + '/ess_000*.nii')
    copes[i] = list(con_images)
    ess[i] = list(ess_images)
    print(copes)
    
#%% Smoothing the specific contrast we want (v[3] meaning the 4th contrast)
smooth_copes = []
for k,v in copes.items():
    smooth_cope = nilearn.image.smooth_img(v[2], 8)
    print(v[2])
    smooth_copes.append(smooth_cope)
    nilearn.plotting.plot_glass_brain(smooth_cope,
                                      display_mode='lyrz', 
                                      colorbar=True, 
                                      plot_abs=False)
    
#%% plotting the smoothed contrasts
nilearn.plotting.plot_glass_brain(nilearn.image.mean_img(smooth_copes),
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False)

#%% Grabing brain mask for analysis
brainmasks = glob.glob('/media/Data/FromHPC/output/fmriprep/sub-*/ses-1/func/sub-*_ses-1_task-3_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
print(brainmasks)
for mask in brainmasks:
    nilearn.plotting.plot_roi(mask)
    
mean_mask = nilearn.image.mean_img(brainmasks)
nilearn.plotting.plot_stat_map(mean_mask)
group_mask = nilearn.image.math_img("a>=0.95", a=mean_mask)
nilearn.plotting.plot_roi(group_mask)


#%% Creating concatenated contrast (across subjects) and group mask
copes_concat = nilearn.image.concat_imgs(smooth_copes, auto_resample=True)
copes_concat.to_filename("/media/Data/work/custom_modelling_spm/negGainRisk_cope.nii.gz")

#group_mask = nilearn.image.resample_to_img(group_mask, copes_concat, interpolation='nearest')
#group_mask.to_filename(os.path.join("/media/Data/work/", "custom_modelling_spm", "group_mask.nii.gz"))

#%% Running randomization
from  nipype.interfaces import fsl
import nipype.pipeline.engine as pe  # pypeline engine
randomize = pe.Node(interface = fsl.Randomise(), base_dir = '/media/Data/work/custom_modelling_spm/neg',
                    name = 'randomize')
randomize.inputs.in_file = '/media/Data/work/custom_modelling_spm/oppnegGainRisk_cope.nii.gz' # choose which file to run permutation test on
randomize.inputs.mask = '/media/Data/work/custom_modelling_spm/group_mask.nii.gz' # group mask file (was created earlier)
randomize.inputs.one_sample_group_mean = True
randomize.inputs.tfce = True
randomize.inputs.vox_p_values = True
randomize.inputs.num_perm = 1000
#randomize.inputs.var_smooth = 5

randomize.run()
#%% Graph it
fig = nilearn.plotting.plot_stat_map('/media/Data/work/custom_modelling_spm/randomize/randomise_tstat1.nii.gz', alpha=0.7 , cut_coords=(0, 45, -7))
fig.add_contours('/media/Data/work/custom_modelling_spm/randomize/randomise_tfce_corrp_tstat1.nii.gz', levels=[0.95], colors='w')
#%% opposite image run
fig = nilearn.plotting.plot_stat_map('/media/Data/work/custom_modelling_spm/neg/randomize/randomise_tstat1.nii.gz', alpha=0.7 , cut_coords=(0, 45, -7))
fig.add_contours('/media/Data/work/custom_modelling_spm/neg/randomize/randomise_tfce_corrp_tstat1.nii.gz', levels=[0.95], colors='w')
#%%
from nipype.caching import Memory
datadir = "/media/Data/work/"
mem = Memory(base_dir='/media/Data/work/custom_modelling_spm')
randomise = mem.cache(fsl.Randomise)
randomise_results = randomise(in_file=os.path.join(datadir, "custom_modelling_spm", "GainvsAmb_cope.nii.gz"),
                              mask=os.path.join(datadir, "custom_modelling_spm", "group_mask.nii.gz"),
                              one_sample_group_mean=True,
                              tfce=True,
                              vox_p_values=True,
                              num_perm=500)
randomise_results.outputs

#%% Look at results
fig = nilearn.plotting.plot_stat_map(randomise_results.outputs.tstat_files[0], alpha=0.7)# , cut_coords=(-20, -80, 18))
fig.add_contours(randomise_results.outputs.t_corrected_p_files[0], levels=[0.95], colors='w')


#%% F contrasts
smooth_es = []
for k,v in ess.items():
    smooth_ess = nilearn.image.smooth_img(v[0], 8)
    print(v[0])
    smooth_es.append(smooth_ess)
    nilearn.plotting.plot_glass_brain(smooth_ess,
                                      display_mode='lyrz', 
                                      colorbar=True, 
                                      plot_abs=False)
    
#%% plotting the smoothed contrasts
nilearn.plotting.plot_glass_brain(nilearn.image.mean_img(smooth_es),
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False)
#%%
ess_concat = nilearn.image.concat_imgs(smooth_es, auto_resample=True)
ess_concat.to_filename("/media/Data/work/custom_modelling_spm/lossTotal.nii.gz")

#%%
randomize.inputs.in_file = '/media/Data/work/custom_modelling_spm/lossTotal.nii.gz'
randomize.inputs.f_only = True
fig = nilearn.plotting.plot_stat_map('/media/Data/work/custom_modelling_spm/randomize/randomise_tstat1.nii.gz', alpha=0.7, cut_coords=(-20, -80, 18))
fig.add_contours('/media/Data/work/custom_modelling_spm/randomize/randomise_tfce_corrp_tstat1.nii.gz', levels=[0.95], colors='w')

#%% Fliping to see the negative
from nipype.interfaces.fsl import MultiImageMaths
maths = MultiImageMaths()
maths.inputs.in_file = "/media/Data/work/custom_modelling_spm/GainRisk_cope.nii.gz"
maths.inputs.op_string = "-add %s -mul -1 -div %s"
!fslmaths "/media/Data/work/custom_modelling_spm/negGainRisk_cope.nii.gz" -mul -1 "/media/Data/work/custom_modelling_spm/oppnegGainRisk_cope.nii.gz"

