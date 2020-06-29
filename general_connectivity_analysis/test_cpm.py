#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Sep 27 14:11:22 2019

@author: or
"""

import numpy as np 
import scipy as sp
import pandas as pd
#from matplotlib import pyplot as plt
#import seaborn as sns
import glob
from scipy import stats
import random
import glob

# %%

def generate_csv_list(path):
    fn_list = glob.glob(path+'/*')
    return fn_list

def read_mats(fn_list):
    """
    Accepts list of csv file names where each csv contains a single subject FC matrix
    Returns stacked matrices
    """
    fns = [pd.read_csv(fn, header=None) for fn in fn_list]
    fns = [df.dropna(axis=1).values for df in fns]            
    fn_mats = np.stack(fns, axis=2)
    return fn_mats


def train_cpm(ipmat, pheno, numNodes, thr):

    """
    Accepts input matrices and pheno data
    Returns model
    @author: David O'Connor
    @documentation: Javid Dadashkarimi
    cpm: in cpm we select the most significant edges for subjects. so each subject
         have a pair set of edges with positive and negative correlation with behavioral subjects.
         It's important to keep both set in final regression task.  
    posedges: positive edges are a set of edges have positive
              correlatin with behavioral measures
    negedges: negative edges are a set of edges have negative
              correlation with behavioral measures
    numNodes: Number of nodes in the parcellation
    thr = threshold of pearson
    """
     
    cc=[stats.pearsonr(pheno,im) for im in ipmat]
    rmat=np.array([c[0] for c in cc])
    pmat=np.array([c[1] for c in cc])
    rmat=np.reshape(rmat,[numNodes,numNodes])
    pmat=np.reshape(pmat,[numNodes,numNodes])
    posedges=(rmat > 0) & (pmat < thr)
    posedges=posedges.astype(int)
    negedges=(rmat < 0) & (pmat < thr)
    negedges=negedges.astype(int)
    pe=ipmat[posedges.flatten().astype(bool),:]
    ne=ipmat[negedges.flatten().astype(bool),:]
    pe=pe.sum(axis=0)/2
    ne=ne.sum(axis=0)/2


    if np.sum(pe) != 0:
        fit_pos=np.polyfit(pe,pheno,1)
    else:
        fit_pos=[]

    if np.sum(ne) != 0:
        fit_neg=np.polyfit(ne,pheno,1)
    else:
        fit_neg=[]

    return fit_pos,fit_neg,posedges,negedges


def kfold_cpm(X,y,k):
    """
    Accepts input matrices and pheno data
    Returns model
    Use "run_validate" instead
    @author: David O'Connor
    @documentation: Javid Dadashkarimi
    X: is the input matrix in v*n which v is number of nodes and n is the number of subjects 
    y: is the gold data which is fluid intelligence
    k: is the size of folds in k-fold
    """

    numsubs = X.shape[1]
    randinds=np.arange(0,numsubs)
    random.shuffle(randinds)

    samplesize=int(np.floor(float(numsubs)/k))

    behav_pred_pos=np.zeros([k,samplesize])
    behav_pred_neg=np.zeros([k,samplesize])

    behav_actual=np.zeros([k,samplesize])

    for fold in range(0,k):
        print("Running fold:",fold+1)
        si=fold*samplesize
        fi=(fold+1)*samplesize


        if fold != k-1:
            testinds=randinds[si:fi]
        else:
            testinds=randinds[si:]

        traininds=randinds[~np.isin(randinds,testinds)]
        
        trainmats=X[:,traininds]
        trainpheno=y[traininds]
 
        testmats=X[:,testinds]
        testpheno=y[testinds]

        behav_actual[fold,:]=testpheno


        pos_fit,neg_fit,posedges,negedges=train_cpm(trainmats,trainpheno, 116, 0.01)

        pe=np.sum(testmats[posedges.flatten().astype(bool),:], axis=0)/2
        ne=np.sum(testmats[negedges.flatten().astype(bool),:], axis=0)/2


        if len(pos_fit) > 0:
            behav_pred_pos[fold,:]=pos_fit[0]*pe + pos_fit[1]
        else:
            behav_pred_pos[fold,:]='nan'

        if len(neg_fit) > 0:
            behav_pred_neg[fold,:]=neg_fit[0]*ne + neg_fit[1]
        else:
            behav_pred_neg[fold,:]='nan'

    return behav_pred_pos,behav_pred_neg,behav_actual



def run_validate(X,y,cvtype):
    
    
    """
    Accepts input matrices (X), phenotype data (y), and the type of cross-valdiation (cv_type)    
    Returns the R-values for positive model (Rpos), negative model (Rneg), and the combination
    X: the feature matrix of size (number of nodes x number of nodes x number of subjects)
    y: the phenotype vector of size (number of subjects)
    cv_type: the cross-valdiation type, takes one of the followings: 
    1) LOO: leave-one-out cross-validation
    2) 5k: 
    """
    numsubs=X.shape[2]
    numNodes = X.shape[0]
    X=np.reshape(X,[-1,numsubs])
    pheno = y
    
    if cvtype == 'LOO':
        behav_pred_pos=np.zeros([numsubs])
        behav_pred_neg=np.zeros([numsubs])
        for loo in range(0,numsubs):

          #  print("Running LOO, sub no:",loo)
      
            train_mats=np.delete(X,[loo],axis=1)
            train_pheno=np.delete(pheno,[loo],axis=0)
            
            test_mat=X[:,loo]
            test_pheno=y[loo]

            pos_fit,neg_fit,posedges,negedges=train_cpm(train_mats,train_pheno, numNodes, 0.01)

            pe=np.sum(test_mat[posedges.flatten().astype(bool)])/2
            ne=np.sum(test_mat[negedges.flatten().astype(bool)])/2

            if len(pos_fit) > 0:
                behav_pred_pos[loo]=pos_fit[0]*pe + pos_fit[1]
            else:
                behav_pred_pos[loo]='nan'

            if len(neg_fit) > 0:
               behav_pred_neg[loo]=neg_fit[0]*ne + neg_fit[1]
            else:
                behav_pred_neg[loo]='nan'
        # in case we run permutation tests and correlation returns an Nan adding try, except here
        try:
            Rpos=stats.pearsonr(behav_pred_pos,pheno)#[0]
            Rneg=stats.pearsonr(behav_pred_neg,pheno)#[0]
        except:
            Rpos = (0.0, 0.99)
            Rneg = (0.0, 0.99)
        return Rpos,Rneg, behav_pred_pos, behav_pred_neg


    elif cvtype == '5k':
        bp,bn,ba=kfold_cpm(X,y,5)



        ccp=np.array([stats.pearsonr(bp[i,:],ba[i,:]) for i in range(0,5)])
        Rpos_mean=ccp.mean(axis=0)[0]

        ccn=np.array([stats.pearsonr(bn[i,:],ba[i,:]) for i in range(0,5)])
        Rneg_mean=ccn.mean(axis=0)[0]

    elif cvtype == '7k':
        bp,bn,ba=kfold_cpm(X,y,7)

        ccp=np.array([stats.pearsonr(bp[i,:],ba[i,:]) for i in range(0,3)])
        Rpos_mean=ccp.mean(axis=0)[0]

        ccn=np.array([stats.pearsonr(bn[i,:],ba[i,:]) for i in range(0,3)])
        Rneg_mean=ccn.mean(axis=0)[0]

    elif cvtype == '10k':
        bp,bn,ba=kfold_cpm(X,y,10)


        ccp=np.array([stats.pearsonr(bp[i,:],ba[i,:]) for i in range(0,10)])
        Rpos_mean=ccp.mean(axis=0)[0]

        ccn=np.array([stats.pearsonr(bn[i,:],ba[i,:]) for i in range(0,10)])
        Rneg_mean=ccn.mean(axis=0)[0]



    elif cvtype == 'splithalf':
        bp,bn,ba=kfold_cpm(X,y,2)

        ccp=np.array([stats.pearsonr(bp[i,:],ba[i,:]) for i in range(0,2)])
        Rpos_mean=ccp.mean(axis=0)[0]

        ccn=np.array([stats.pearsonr(bn[i,:],ba[i,:]) for i in range(0,2)])
        Rneg_mean=ccn.mean(axis=0)[0]


    else:
        raise Exception('cvtype must be LOO, 5k, 10k, or splithalf')


    return Rpos_mean,Rneg_mean
    


# %%
n = np.load('/home/or/kpe_task_analysis/trauma_ses1.npy')
n.shape

# %%
first = np.load('/home/or/kpe_task_analysis/trauma_ses1.npy')
second = np.load('/home/or/kpe_task_analysis/trauma_ses2.npy')
delta = np.subtract(second, first)

# delta1 = np.delete(delta, mask, axis=2)
# delta1.shape
y = np.array([2,  -1,  30,  17,  28,   4,  30,  18,  22,  18,  -1,  11,   2,   4, -16,  32,   8,  14,  20,   3,  23])
delta.shape


# %%

# %%
## Run three fold with many iterations
rposAll = []
rnegAll = []
for i in range(1000):
    print(f'Iteration nu. {i}')
    Rpos,Rneg = run_validate(delta, y, '7k')
    rposAll.append(Rpos)
    rnegAll.append(Rneg)


# %%
sns.distplot(rposAll)
np.mean(rposAll)

# %%
rposAll = np.array(rposAll)
np.quantile(rposAll, [0.05, 0.95])

# %%
sns.distplot(rnegAll)
np.mean(rnegAll)

# %%
## use permutation test to assess the real value of this calculation
nitr = 500
rPos_perm = []
rNeg_perm = []
for i in range(nitr):
    print (f'Iteration No. {i}')
    y_rand = random.shuffle(y)
    #print(y)
    Rpos,Rneg, behav_pred_pos, behav_pred_neg = run_validate(delta, y, 'LOO')
    rPos_perm.append(Rpos)
    rNeg_perm.append(Rneg)

# %%
# plotting the permutation tests
rPos_perm = np.array(rPos_perm)
rNeg_perm = np.array(rNeg_perm)

sns.distplot(rPos_perm[:,0])
sns.distplot(rNeg_perm[:,0])

# %%
# run it one time not randomized
y = np.array([2,  -1,  30,  17,  28,   4,  30,  18,  22,  18,  -1,  11,   2,   4, -16,  32,   8,  14,  20,   3,  23])
Rpos,Rneg, behav_pred_pos, behav_pred_neg = run_validate(delta, y, 'LOO')

# %%
Rpos

# %%
# chances of receiving this result if its random is...
sum(rPos_perm[:,0]>0.41) / len(rPos_perm)


# %%
# chances of rNeg
sum(rNeg_perm[:,0]>0.24) / len(rNeg_perm)


# %% [markdown]
# ### So - positive correlation is above chance level, negative is close - but not above chance level

# %%
import matplotlib.pyplot as plt
import seaborn as sns
group_label = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
df = pd.DataFrame({'positive_pred': behav_pred_pos, 'negative_pred':behav_pred_neg, 'observed': y, 'group':group_label})
df = df.replace(to_replace={'group': {0:'midazolam', 1:'ketamine'}})
sns.set_style("whitegrid")
sns.regplot('positive_pred', 'observed', data=df)
ax = sns.scatterplot('positive_pred', 'observed', data=df, hue='group')
plt.legend(loc='upper left')
ax.set(xlabel="Prediction from Positive Edges", ylabel = "Observed delta")
print(f'Correlation between positive edges and changes in symptoms is {stats.pearsonr(df.positive_pred,df.observed)}')

# %%
df

# %%
import matplotlib.pyplot as plt
import seaborn as sns
group_label = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
df = pd.DataFrame({'positive_pred': behav_pred_pos, 'negative_pred':behav_pred_neg, 'observed': y, 'group':group_label})
df


# %%
df.replace(to_replace={'group': {0:'midazolam', 1:'ketamine'}})

# %%
sns.lmplot('positive_pred', 'observed', data=df, hue='group', scatter_kws={'alpha':0.5})

# %%
stats.pearsonr(df.negative_pred[df['group']=='midazolam'],df.observed[df['group']=='midazolam'])
