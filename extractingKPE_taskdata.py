#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:02:52 2019

@author: Or Duek
Extracting KPE physio data from biopac using bioread package
https://github.com/uwmadison-chm/bioread

This script eventually create a csv file for each session for each subject - with condition (Trauma, Relax, Sad), onset and duration.
It takes the scanSheet excel file to get the order of scripts per session.
To run the script you should call all functions and then run a loop (at bottom of file) with subject numbers
"""

import bioread
import os

# loading file
#a= bioread.read('/media/Drobo/Levy_Lab/Projects/PTSD_KPE/physio_data/raw/kpe1387/scan_1/kpe1387.1_scripts_2018-09-10T08_39_24.acq')

# choose scripts channel
#b = a.channels[7].raw_data
#%%
# create loop that will look for changes between zero and back to zero as on and off set time points. 
def lookZero(b, offSet): # take Channel data and if we need to adjust timings
    time_onset = []
    time_offset = []
# this function takes a raw data from bioread channel and return two arrays of on and off sets.
    look_for_zero = b[0] != 0
    for i, v in enumerate(b[1:]):
        if look_for_zero and v == 0:
            look_for_zero = False
            time_offset.append(i/1000 - offSet)
        elif not look_for_zero and v != 0:
            look_for_zero = True
            time_onset.append(i/1000 - offSet)
    return (time_onset, time_offset)
        
#%% Function to extract actual data from subjects
def kpeTaskDat(filename):
    # takes filename and returns data frame of onsets and duration. Needs to attach condition and subject number
    import pandas as pd
    a = bioread.read(filename)
   ## Take the first ready screen
    readyScreen = a.named_channels["Ready Screen"].raw_data
    readyOn = lookZero(readyScreen,0)[0]     
    # set difference between first appereance and TRs. 
    # Setting to first Ready screen at 6 seconds
    diff = readyOn[0] - 6
    # Choose Script channel by its name
    b = a.named_channels["Script"].raw_data
    scriptTime = lookZero(b, diff)
    duration = []
    #condition = []
    for i in range(len(scriptTime[0])): # run through the set
        duration.append(scriptTime[1][i] - scriptTime[0][i]) # create duration
    events= pd.DataFrame({'onset':scriptTime[0], 'duration':duration})
    return events

def orderSize(folder):
    # this simple function will return the highest file size, in order to get the largest acknowledgment file 
    # The folder containing files.
    directory = folder
    # Get all files.
    list = os.listdir(directory)
    
    # Loop and add files to list.
    pairs = []
    for file in list:
        # Use join to get full file path.
        location = os.path.join(directory, file)
        # Get size and add to list of tuples.
        size = os.path.getsize(location)
        pairs.append((size, file))
    # Sort list of tuples by the first element, size.
    pairs.sort(key=lambda s: s[0], reverse = True)
    # Display pairs.
    return pairs[0][1] # return only file name
#%% This part takes the scan sheet and create a data frame with condition and sessions. 
totalScanData = pd.read_excel('/media/Data/PTSD_KPE/kpe_scan_table.xls', sheet_name = 'kpe_scan_table')        
# short loop to fill in subject numebrs and sessions
totalScanData["subject_id"] = totalScanData["subject_id"].fillna('noSub') # filling all NaNs with noSub. 
# create a session column
   
for index,rows  in totalScanData.iterrows():
    print(index)
    print (rows.subject_id)
    if rows.subject_id != 'noSub':
        subject = rows.subject_id
 
    else:
        totalScanData["subject_id"][index] = subject
 
trialOrder = pd.DataFrame({'subject_id': totalScanData["subject_id"], "scriptOrder":totalScanData["Script Order"], "session":totalScanData["scan_num"]})
 # read subject id and pick the right line from the data frame
#%%
def getCondition(subNum):
    
    # use scanSheet (from top lines), subjectNumber and session to return a list of condition by order of appereance    
    subjectId = "kpe" + str(subNum)
    subjectData = trialOrder[trialOrder.subject_id==subjectId]
    subjectData = subjectData.dropna(subset=['scriptOrder', 'session']) # removing NaN rows (with no session or scriptOrder)
    # rnu through all session of subject and create conditions with onset and duration. 
    conditionDat = pd.DataFrame(columns=['session', 'condition'])
    for s , r in subjectData.iterrows(): # s is index and r is the actual row
        print (r.scriptOrder)
    #    condition = []
        session = int(r.session)
        print(session)
        breakTrial = subjectData["scriptOrder"][s].split() # now its the first line but should be with subject id accordinaly. 
        for n in breakTrial:
            print (n)
            if 'Sad;' in n:
                conditionDat = conditionDat.append({'session':session,'condition':'sad'}, ignore_index = True)
                print ("Out of loop")
                break
            elif 'Trauma;' in n:
                conditionDat = conditionDat.append({'session':session,'condition':'trauma'}, ignore_index = True)
                break
            elif 'Relaxing;' in n:
                conditionDat = conditionDat.append({'session':session,'condition':'relax'}, ignore_index = True)
                break
            elif 'Sad' in n:
                #condition.append('sad')
                conditionDat = conditionDat.append({'session':session,'condition':'sad'}, ignore_index = True)
            elif 'Relax' in n:
                #condition.append('relax')
                conditionDat = conditionDat.append({'session':session,'condition':'relax'}, ignore_index = True)
            elif 'Trauma' in n:
                #condition.append('trauma')
                conditionDat = conditionDat.append({'session':session,'condition':'trauma'}, ignore_index = True)
            else:
                pass
            #conditionDat = conditionDat.append({'session':session,'condition':condition}, ignore_index = True)
        #conditionList.append(conditionDat)
    return conditionDat
        #conditionTotal.append(conditionDat)
    #events= pd.DataFrame({'onset':scriptTime[0], 'duration':duration, 'condition':condition})
    # now we should create a data frame
    

#%% 
# this function takes subject and session number and returns the specific acq file
def getFile(subNum, session):
    data_dir = '/media/Data/PTSD_KPE/physio_data/raw/'
    folder = data_dir + "kpe" + str(subNum) + "/" + "Scan_" + str(session) + "/"
    try:
        fullFile = orderSize(folder)
        return folder + fullFile
    except:
        print (f"The following folder + file doesn't exist: {folder}")    
        return 99


#%%
# now we can iterate through subjects and sessions and create subject data for each
# for now - lets create tsv files for each subject per each session
subList = ['008','1223','1253','1263','1293','1307','1315','1322','1339','1343','1351','1356','1364','1369','1387','1390','1403','1464']

sessionList = [1,2,3,4]  

for sub in subList:
    subNum = sub
    print(subNum)
    # get condition list for all sessions
    conditionList = getCondition(subNum)
    # set session
    for i in sessionList:
        session  = i
        print (session)
        # call file
        file = getFile(subNum, session)
        if file == 99:
            # if no scan then passloop
            continue
        print (file)
        # get script order
        conditionSession = conditionList[conditionList.session==session]
        onsetsDat = kpeTaskDat(file)
        # combine the two 
        onsetsDat['trial_type'] = conditionSession['condition'].tolist()
        # save as tsv file in specifi location BIDS compatible name (i.e. sub-subNum_ses_session_task_.tsv)
        # save filename in folder
        onsetsDat.to_csv(r'/media/Data/PTSD_KPE/condition_files/'+'sub-' + str(subNum)+ '_' + 'ses-' +str(session)+'.csv', index = False, sep = '\t')
    
