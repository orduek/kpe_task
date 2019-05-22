#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:28:29 2019

@author: or
"""

from biosppy import storage
from biosppy.signals import eda, ecg

import bioread

a= bioread.read('/media/Drobo/Levy_Lab/Projects/PTSD_KPE/physio_data/raw/kpe1387/scan_1/kpe1387.1_scripts_2018-09-10T08_39_24.acq')

# choose scripts channel
b = a.named_channels["GSR100C"].raw_data

c = b[28700:154159]
out = eda.eda(signal=c, sampling_rate=1000., show=True)


k["onsetCor"] = k["onset"]+ diff


j = a.named_channels["Heart Rate"].raw_data

n = j[287:154159]

ecgOut = ecg.ecg(signal = n, show=True)
