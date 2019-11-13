#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:35:31 2019

@author: Or Duek
A set of functions to run seed based analyses
"""

def createDelta(func_files1, func_files2, mask_img):
    from nilearn.input_data import NiftiMasker
    
    # here I use a masked image so all will have same size
    nifti_masker = NiftiMasker(
        mask_img= mask_img,
        smoothing_fwhm=6,
        memory='nilearn_cache', memory_level=1, verbose=2)  # cache options
    fmri_masked_ses1 = nifti_masker.fit_transform(func_files1)
    fmri_masked_ses2 = nifti_masker.fit_transform(func_files2)
    ###
    from nilearn import input_data
    brainMasker = input_data.NiftiMasker(
            smoothing_fwhm=6,
            detrend=True, standardize=True,
            t_r=1.,
            memory='/media/Data/nilearn', memory_level=1, verbose=2)
    brainMasker.fit(func_files1)

    ####
    deltaCor_a = fmri_masked_ses2 - fmri_masked_ses1
    print (f'Shape is: {deltaCor.shape}')

    # run paired t-test 
    testDelta = scipy.stats.ttest_rel(fmri_masked_ses1, fmri_masked_ses2) 
    print (f'Sum of p values < 0.005 is {np.sum(testDelta[1]<0.005)}')
    
    
    return deltaCor_a, testDelta # return the delta correlation and the t-test array

def createZimg(deltaCor, scriptName, seedName):
    # mean across subjects
    mean_zcor_Delta = np.mean(deltaCor,0)
    mean_zcor_img_delta = brain_masker.inverse_transform(mean_zcor_Delta.T)
    # save it as file
    mean_zcor_img_delta.to_filename(
        '/home/or/kpe_conn/%s_seed_%s_delta_z.nii.gz' %(scriptName,seedName))
    
    return mean_zcor_img_delta, mean_zcor_Delta # returns the image and the array 

## now create a function to do FDR thresholding
def fdrThr(testDelta, mean_zcor_Delta, alpha, brain_masker):
    from statsmodels.stats import multitest
    # we need to reshape the test p-values array to create 1D array
    #b = np.reshape(np.array(testDelta[1]), -1)
    fdr_mat = multitest.multipletests(testDelta[1], alpha=alpha, method='fdr_bh', is_sorted=False, returnsorted=False)
    #fdr_mat = multitest.fdrcorrection(testDelta[1], alpha=0.7, method='indep', is_sorted=False)
    np.sum(fdr_mat[1]<0.05)
    corr_mat_thrFDR = np.array(mean_zcor_Delta)
    corr_mat_thrFDR = np.reshape(corr_mat_thrFDR, -1)
    corr_mat_thrFDR[fdr_mat[0]==False] = 0
   
    # now I can treshold the mean matrix
    numNonZeroDelta = np.count_nonzero(corr_mat_thrFDR)
    print (f'Number of voxels crossed the FDR thr is {numNonZeroDelta}')
    # transofrm it back to nifti
    nifti_fdr_thr = brain_masker.inverse_transform(corr_mat_thrFDR.T)
    return corr_mat_thrFDR, nifti_fdr_thr # return matrix after FDR and nifti file
                                         