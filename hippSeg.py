#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:29:21 2019

@author: Or Duek
Building Hippocampus and Amygdala subfields. Taken instructions from here: https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala

In order to run we need FreeSurfer development version.

For interface I've used this tutorial: https://miykael.github.io/nipype_tutorial/notebooks/advanced_create_interfaces.html
"""

from nipype.interfaces.base import CommandLine
from traits.api import Directory, String
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec

class CustomHippoSegInputSpec(CommandLineInputSpec):
   # in_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='the input image')
    subject_dir = Directory(exists=True, mandatory=True, argstr='%s', position=1, desc='the location of data')
    subject_id = String(exists=True, mandatory=True, argstr='%s', position=0, desc='subject ID (No.)')

class CustomHippoSeg(CommandLine):
    _cmd = 'segmentHA_T1.sh'
    input_spec = CustomHippoSegInputSpec
   # output_spec = CustomHippoSegOutputSpec
#
#    def _list_outputs(self):
#
#        # Get the attribute saved during _run_interface
#        return {'out_file': self.inputs.out_file,
#                'mask_file': self.inputs.out_file.replace('brain')}
#        
