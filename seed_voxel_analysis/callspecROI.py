#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:37:09 2020

@author: Or Duek

simple file that will run a whole bunch of subject's seed2voxel analysis'
calling specROI.py

"""
import nipype.interfaces.io as nio  # Data i/o
from nipype.pipeline import engine as pe
from nipype.interfaces.ants import ApplyTransforms
import nipype.interfaces.utility as util  # utility
from nipype.interfaces.base import CommandLine
from nipype.interfaces.base import CommandLineInputSpec
from traits.api import Int, Str
import os

#%% define basic vars
ses = '1' #session
roi_num = 54 #ROI according to freesurfer lookup table
roi_name = 'leftAmg' #name of seed
script_type = 'trauma' # can be 'sad' or 'relax' or 'trauma'

work_dir = '/media/Data/work'
output_dir = '/media/Data/KPE_results'

#%%


#subject_list = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464', '1468','1480','1499'] 
subject_list = ['1364','1369','1387','1390','1464', '1468','1480','1499']

infosource = pe.Node(util.IdentityInterface(fields=['subject_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list)]


# def callSpecROI(subject_num, session, work_dir, output_dir, roi_number, roi_name, script_type):
# #    import subprocess
#  #   cmd = ['python','home/or/kpe_task_analysis/specROI_voxel.py', str(subject_id), str(session), str(work_dir), str(output_dir), str(roi_number), str(roi_name), str(script_type)]
#     !python /home/or/kpe_task_analysis/specROI_voxel.py subject_num session work_dir output_dir roi_number roi_name script_type
    #return subprocess.call(cmd)

# runSpecROI= pe.Node(util.Function(
#     input_names=['subject_num', 'session', 'work_dir', 'output_dir', 'roi_number','roi_name','script_type'],
#     function=callSpecROI),
#     name='runSpecROI')
 
                   
class TransformInfoInputSpec(CommandLineInputSpec):
    subject_id = Str(exists=True, mandatory=True, argstr='%s',
                   position=0, desc='subject id')
    session = Str(exists=True, mandatory=True, argstr='%s',
                   position=1, desc='session')
    work_dir = Str(exists=True, mandatory=True, argstr='%s',
                   position=2, desc='working folder')
    output_dir = Str(exists=True, mandatory=True, argstr='%s',
                   position=3, desc='output folder')
    roi_num = Int(exists=True, mandatory=True, argstr='%i',
                   position=4, desc='roi number according to freesurfer lookuptable')
    roi_name = Str(exists=True, mandatory=True, argstr='%s',
                   position=5, desc='name of seed')
    script_type = Str(exists=True, mandatory=True, argstr='%s',
                   position=6, desc="type of script, ('trauma','sad','relax')")

class extractSeed(CommandLine):
    _cmd = 'python /home/or/kpe_task_analysis/specROI_voxel.py'
    input_spec = TransformInfoInputSpec


#my_info_interface = TransformInfo(subject_id = '1223',session = '1')

runSpecROI = pe.Node(extractSeed(), name = 'runSpecROI')

runSpecROI.inputs.session = ses
runSpecROI.inputs.work_dir = work_dir
runSpecROI.inputs.output_dir = output_dir
runSpecROI.inputs.roi_num = roi_num
runSpecROI.inputs.roi_name = roi_name
runSpecROI.inputs.script_type = script_type



wfSpecRoi = pe.Workflow(name = 'runSpecROI', base_dir = '/media/Data/work')

wfSpecRoi.connect([
    (infosource, runSpecROI, [('subject_id','subject_id')])
    ])

wfSpecRoi.run('MultiProc', plugin_args={'n_procs': 4})


#!python /home/or/kpe_task_analysis/specROI_voxel.py subject_id session work_dir output_dir roi_number roi_name script_type
