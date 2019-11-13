#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:40:13 2019

@author: Or Duek
Analyzing GSR data
"""

#%%
events_file = '/media/Data/PTSD_KPE/condition_files/sub-1223_ses-1.csv'
#%% Read event file

import numpy as np
import pandas as pd
from scipy import signal 
events = pd.read_csv(events_file, sep=r'\s+')

from biosppy import storage
from biosppy.signals import eda, ecg

import bioread

a= bioread.read('/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_1/kpe1223.1_scripts_2017-01-30T08_17_08.acq')

# choose scripts channel
b = a.named_channels["GSR100C"].raw_data

a_resample = signal.decimate(b, 40)
plt.plot(b)
plt.plot(a_resample)
plt.show()
c = a_resample[int(7000/40):int(130000/40)]
out = eda.eda(signal=b, sampling_rate=1000., show=True)
len(out['filtered'])

plt.plot(out['ts'])
d = b[130000:252799]
out = eda.eda(signal=b, sampling_rate=1000., show=True)


#%% second scan
events_file2 = '/media/Data/PTSD_KPE/condition_files/sub-1223_ses-2.csv'
events = pd.read_csv(events_file2, sep=r'\s+')

a= bioread.read('/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_2/kpe1223.2_scripts_2017-02-06T08_39_32.acq')

# choose scripts channel
b = a.named_channels["GSR100C"].raw_data

import matplotlib.pyplot as plt

plt.plot(b)


#%% analyze data using NeuroKit
import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns


# loading BioPack file
df, sampling_rate = nk.read_acqknowledge('/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_2/kpe1223.2_scripts_2017-02-06T08_39_32.acq', return_sampling_rate=True)
df[['GSR100C', 'Script']].plot()
# Process the signals
bio = nk.bio_process(eda=df["GSR100C"], add=df["Script"], sampling_rate=1000.)
# Plot the processed dataframe, normalizing all variables for viewing purpose
#%%
condition_list = ["Trauma", "Relax", "Sad","Relax","Trauma", "Sad","Relax", "Trauma", "Sad"]
#%%
def analyzeSCR(event_file, acq_file):
    import neurokit as nk
    import pandas as pd
    import numpy as np
    import seaborn as sns
    ## loading acq file
    df, sampling_rate = nk.read_acqknowledge(acq_file, return_sampling_rate=True)
    bio = nk.bio_process(eda=df["GSR100C"], add=df["Script"], sampling_rate=sampling_rate)
    
    # adding conditions
    events = pd.read_csv(event_file, sep=r'\s+')
    condition_list = events['trial_type']
    events = {'onset':np.array(events["onset"]*1000),'duration':np.array(events["duration"]*1000)}
    epochs = nk.create_epochs(bio["df"], events["onset"], duration=120000, onset=0) # create epoch file with 120sec duration and -1sec begins

    data = {}  # Initialize an empty dict
    for epoch_index in epochs:
        data[epoch_index] = {}  # Initialize an empty dict for the current epoch
        epoch = epochs[epoch_index]
    
        # ECG
    #    baseline = epoch["ECG_RR_Interval"].ix[-100:0].mean()  # Baseline
    #    rr_max = epoch["ECG_RR_Interval"].ix[0:400].max()  # Maximum RR interval
    #    data[epoch_index]["HRV_MaxRR"] = rr_max - baseline  # Corrected for baseline
    
        # EDA - SCR
        scr_max = epoch["SCR_Peaks"].ix[100:15000].max()  # Maximum SCR peak - now its 30sec after initiation of script
        if np.isnan(scr_max):
            scr_max = 0  # If no SCR, consider the magnitude, i.e.  that the value is 0
        data[epoch_index]["SCR_Magnitude"] = scr_max
    
    data = pd.DataFrame.from_dict(data, orient="index")  # Convert to a dataframe
    data["Condition"] = condition_list  # Add the conditions
    data  # Print 
    return data
    
#%%
scan1_dat = analyzeSCR('/media/Data/PTSD_KPE/condition_files/sub-1223_ses-1.csv', '/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_1/kpe1223.1_scripts_2017-01-30T08_17_08.acq')
scan2_dat = analyzeSCR('/media/Data/PTSD_KPE/condition_files/sub-1223_ses-2.csv','/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_2/kpe1223.2_scripts_2017-02-06T08_39_32.acq')
scan3_dat = analyzeSCR('/media/Data/PTSD_KPE/condition_files/sub-1223_ses-3.csv','/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_3/kpe1223.3_scripts_2017-03-11T10_41_05.acq')
scan4_dat = analyzeSCR('/media/Data/PTSD_KPE/condition_files/sub-1223_ses-4.csv','/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_4/kpe1223.4_scripts_2017-05-12T10_39_28.acq')

#%% plot data
data_all = {'scan1':scan1_dat, 'scan2':scan2_dat, 'scan3':scan3_dat, 'scan4':scan4_dat}
for i in data_all:
    plt.figure() #this creates a new figure on which your plot will appear
    plt.title(i)
    sns.boxplot(x="Condition", y="SCR_Magnitude", data=data_all[i])

sns.lineplot(y="SCR_Magnitude", data = scan1_dat)
plt.plot(scan3_dat["SCR_Magnitude"])

#%% GSR analysis with cvxEDA
import os
os.chdir('/home/or/kpe_task_analysis')
import cvxEDA
y = b[7000:130000]
yn = (y - y.mean()) / y.std()
Fs = 1000.
[r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1./Fs)
import pylab as pl
tm = pl.arange(1., len(y)+1.) / Fs
pl.hold(True)
pl.plot(tm, yn)
pl.plot(tm, r)
pl.plot(tm, p)
pl.plot(tm, t)
pl.show()
#%% using pypsy
import Pypsy as ps
ps.signal.analysis.interimpulse_fit()


#%% Using ledapay
import ledapy
import scipy.io as sio
from numpy import array as npa
import bioread
a= bioread.read('/media/Data/PTSD_KPE/physio_data/raw/kpe1223/scan_1/kpe1223.1_scripts_2017-01-30T08_17_08.acq')
sampling_rate = 1000
# choose scripts channel
rawdata  = a.named_channels["GSR100C"].raw_data

phasicdata = ledapy.runner.getResult(rawdata, 'phasicdriver', sampling_rate, downsample=40, optimisation=2)
import matplotlib.pyplot as plt  # note: requires matplotlib, not installed by default
plt.plot(phasicdata)
plt.show()