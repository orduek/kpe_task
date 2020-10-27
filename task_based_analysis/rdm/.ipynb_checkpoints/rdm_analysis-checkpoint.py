#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Wed Mar 11 12:19:37 2020

@author: Ruonan Jia, Or Duek

Using Ruonan's code to generate dissimilarity matrix
"""

# %% Compute ROI RDM
    
def compute_roi_rdm(in_file,
                    stims,
                    all_masks):
    
    from pathlib import Path
    from nilearn.input_data import NiftiMasker
    import numpy as np
    import nibabel as nib
    
    rdm_out = Path('roi_rdm.npy').resolve()
    stim_num = len(stims)
    
    # dictionary to store rdms for all rois
    rdm_dict = {}
    
    # loop over all rois
    for mask_name in all_masks.keys():
        mask = all_masks[mask_name]
        masker = NiftiMasker(mask_img=mask)
        
        # initiate matrix
        spmt_allstims_roi= np.zeros((stim_num, np.sum(mask.get_data())))
            
        for (stim_idx, spmt_file) in enumerate(in_file):
            spmt = nib.load(spmt_file)
          
            # get each condition's beta
            spmt_roi = masker.fit_transform(spmt)
            spmt_allstims_roi[stim_idx, :] = spmt_roi
        
        # create rdm
        rdm_roi = 1 - np.corrcoef(spmt_allstims_roi)
        
        rdm_dict[mask_name] = rdm_roi
        
    # save    
    np.save(rdm_out, rdm_dict)
    
    return str(rdm_out)



get_roi_rdm = Node(util.Function(
    input_names=['in_file', 'stims', 'all_masks'],
    function=compute_roi_rdm, 
    output_names=['rdm_out']),
    name='get_roi_rdm',
    )    
    
get_roi_rdm.inputs.stims = {'01': 'Med_amb_0', '02': 'Med_amb_1', '03': 'Med_amb_2', '04': 'Med_amb_3',
                            '05': 'Med_risk_0', '06': 'Med_risk_1', '07': 'Med_risk_2', '08': 'Med_risk_3', 
                            '09': 'Mon_amb_0', '10': 'Mon_amb_1', '11': 'Mon_amb_2', '12': 'Mon_amb_3',
                            '13': 'Mon_risk_0', '14': 'Mon_risk_1', '15': 'Mon_risk_2', '16': 'Mon_risk_3'}

# Masker files
maskfile_vmpfc = os.path.join(output_dir, 'binConjunc_PvNxDECxRECxMONxPRI_vmpfc.nii.gz')
maskfile_vstr = os.path.join(output_dir, 'binConjunc_PvNxDECxRECxMONxPRI_striatum.nii.gz')
maskfile_roi1 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p005_roi1.nii.gz')
maskfile_roi2 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p005_roi2.nii.gz')
maskfile_roi3 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p005_roi3.nii.gz')

maskfiles = {'vmpfc': maskfile_vmpfc, 
             'vstr': maskfile_vstr, 
             'med_mon_1': maskfile_roi1, 
             'med_mon_2': maskfile_roi2, 
             'med_mon_3': maskfile_roi3}

# roi inputs are loaded images
get_roi_rdm.inputs.all_masks = {key_name: nib.load(maskfiles[key_name]) for key_name in maskfiles.keys()}


wfSPM_rsa.connect([
        (contrastestimate, get_roi_rdm, [('spmT_images', 'in_file')]),
        ])

# %% data sink rdm
# Datasink
datasink_rdm = Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_resp_rsa_nosmooth')),
                                         name="datasink_rdm")
                       

wfSPM_rsa.connect([
        (get_roi_rdm, datasink_rdm, [('rdm_out', 'rdm.@rdm')]),
        ])

# %%
wfSPM_rsa.write_graph(graph2use = 'flat')

# # wfSPM.write_graph("workflow_graph.dot", graph2use='colored', format='png', simple_form=True)
# # wfSPM.write_graph(graph2use='orig', dotfilename='./graph_orig.dot')
# %matplotlib inline
# from IPython.display import Image
# %matplotlib qt
# Image(filename = '/home/rj299/project/mdm_analysis/work/l1spm/graph.png')

# %% run
wfSPM_rsa.run('MultiProc', plugin_args={'n_procs': 4})
