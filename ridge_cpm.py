'''
@Author: Or Duek
Using Ridge regression to run the CPM for the KPE study
'''
#%% Load libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import PredefinedSplit
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import pandas as pd
import numpy as np
import scipy.io as sio
import h5py

import time


#%% Load matrices and behaviour
# AAL
first = np.load('/home/or/kpe_task_analysis/trauma_ses1.npy')
second = np.load('/home/or/kpe_task_analysis/trauma_ses2.npy')
delta = np.subtract(second, first)

# delta1 = np.delete(delta, mask, axis=2)
# delta1.shape
y = np.array([-2,  1,  -30,  -17,  -28,   -4,  -30,  -18,  -22,  -18,  1,  -11,   -2,   -4, 16,  -32,   -8,  -14,  -20,   -3,  -23])
delta.shape

#%% turn matrix to array
vecs = []
for i in range(delta.shape[2]):
    mat = delta[:,:,i]
    flat = mat.flatten()
    vecs.append(flat) 
vecs = np.array(vecs)
print(vecs.shape)
vecs_reshape = np.moveaxis(vecs,0,-1)
#%% Set parameters of the model
pct = 0.8 # percent of edges kept in feature selection
alphas = 10**np.linspace(10,-2,100)*0.5 # specify alphas to search
#%%
rg_grid = GridSearchCV(Ridge(normalize=False), cv=10, param_grid={'alpha':alphas}, iid=False)
# using LASSO regression instead of ridge
lasso = linear_model.Lasso
lasso_grid = GridSearchCV(lasso(normalize=False), cv=10, param_grid={'alpha':alphas}, iid=False)

reg = Pipeline([
  ('feature_selection', SelectPercentile(f_regression, percentile=pct)),
  ('regression', lasso_grid)
])

cv10 = KFold(n_splits=21)#, random_state=665)
rpcv10 = RepeatedKFold(n_splits=3,n_repeats=3, random_state=665)
# %% Run model
start = time.time() # time the function
all_pred = cross_val_predict(reg, vecs_reshape.T, y, cv=cv10, n_jobs=4)
#all_score = cross_val_score(reg, vecs_reshape.T, y, cv=rpcv10, n_jobs=1) # repeated kfolds
end = time.time()
print(end - start) # print function running time

# %%
print(np.corrcoef(all_pred.T, y.T))

# %%
import scipy
scipy.stats.pearsonr(all_pred, y)
#%% plot
import seaborn as sns
import matplotlib.pyplot as plot
sns.regplot(all_pred, y)


#%% Try with shen parcellation
first = np.load('/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_results/shen/trauma_ses1_shen.npy')
second = np.load('/media/Data/Lab_Projects/KPE_PTSD_Project/neuroimaging/KPE_results/shen/trauma_ses2_shen.npy')
delta = np.subtract(second, first)


# %%
# lets plot with groups also
df = pd.DataFrame({'group': group_label, 'observed':y, 'predicted':all_pred})


sns.lmplot('predicted','observed', hue='group', data= df)
print(f'Ketamine group correlation {scipy.stats.pearsonr(df.predicted[df.group==1], df.observed[df.group==1])}')
print(f'Midazolam group correlation {scipy.stats.pearsonr(df.predicted[df.group==0], df.observed[df.group==0])}')
# %%
