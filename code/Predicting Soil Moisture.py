#!/usr/bin/env python
# coding: utf-8

# # Mini Project
# 
# # Predicting Soil Moisture using Weather Data
# 
# # CS-A (ABESEC)
# 
# # Atul Dagar (2000320120056)
# # Anurag Bhardwaj (2000320120039)
# # Aryan Tyagi (2000320120050)

# In[1]:


import numpy as np
import pandas as pd 
import os as os
from time import time, strftime
from datetime import datetime
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.Benchmark import Benchmark

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, explained_variance_score, r2_score
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')


# ## Benchmarks

# In[2]:


get_ipython().system(' python --version')


# In[3]:


import time
StopWatch.start("a")
time.sleep(3)
StopWatch.stop("a")
StopWatch.status("a", True)
StopWatch.benchmark()


# In[4]:


def b():
  Benchmark.Start()
  print ("b")
  import time
  time.sleep(3)
  Benchmark.Stop()

def c():
  Benchmark.Start()
  print ("c")
  import time
  time.sleep(1)
  Benchmark.Stop()


# In[5]:


b()
c()


# In[6]:


Benchmark.print()


# ## Data Processing

# In[7]:


class Load_Data(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        self.weather_dir = ''
        self.soil_dir = ''
        self.drop_columns = ['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'AWND_ATTRIBUTES', 'PGTM_ATTRIBUTES', 
                             'PSUN', 'PSUN_ATTRIBUTES', 'SNOW', 'SNOW_ATTRIBUTES', 'SNWD', 'SNWD_ATTRIBUTES', 'TAVG',
                             'TAVG_ATTRIBUTES', 'TMAX_ATTRIBUTES', 'TMIN_ATTRIBUTES', 'TSUN', 'TSUN_ATTRIBUTES', 'WDF2_ATTRIBUTES', 
                             'WDF5_ATTRIBUTES', 'WSF2_ATTRIBUTES','WSF5_ATTRIBUTES', 'WT01_ATTRIBUTES', 'WT02_ATTRIBUTES', 
                             'WT03_ATTRIBUTES', 'WT06_ATTRIBUTES', 'WT08_ATTRIBUTES', 'PRCP_ATTRIBUTES']
        
    def fit(self, w_dir, s_dir):
        self.weather_dir = w_dir
        self.soil_dir = s_dir
        return self
    
    def transform(self, X):
        #Aggregate all 43 files into one file
        file_list = os.listdir(self.soil_dir)
        agg_data = pd.DataFrame()
        for file in file_list:
            path = self.soil_dir + file
            curr_data = pd.read_csv(path, sep='\t')
            agg_data = agg_data.append(curr_data)
        
        #Drop rows with only NAs for measurement values
        soil = agg_data.dropna(thresh=10)
        
        #Import weather files and drop unnessecary fields
        weather = pd.read_csv(self.weather_dir)
        drop_cols = list(set(weather.columns).intersection(self.drop_columns))
        weather = weather.drop(columns = self.drop_columns)
        
        #Convert both files to use same datetime
        soil['Date'] = pd.to_datetime(soil['Date'])
        weather['DATE'] = pd.to_datetime(weather['DATE'])
        
        #Join previous 16 days weather to moisture readings
        for i in range(0, 17):
            weather_new = weather.add_suffix('_' + str(i))
            soil = soil.merge(weather_new, how = 'left', left_on = 'Date', right_on = weather['DATE'] - pd.DateOffset(i * -1))
            
        #Store the month of the reading as a feature
        soil['Month'] = pd.DatetimeIndex(soil['Date']).month
        
        date_attribs = ['Date', 'DATE_0', 'DATE_1', 'DATE_2', 'DATE_3', 'DATE_4','DATE_5', 'DATE_6', 'DATE_7', 'DATE_8', 'DATE_9', 'DATE_10',                         'DATE_11', 'DATE_12', 'DATE_13', 'DATE_14', 'DATE_15', 'DATE_16']
        
        if 'DATE_0' in list(soil.columns):
            soil.drop(columns = date_attribs, inplace = True)
        soil['Location'] = soil['Location'].astype('object')
            
        return soil


# In[8]:


class Feature_Engineer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        #Add categorical feature that simply stores if it rained that day or not
        for i in range(17):
            col_name = 'PRCP_' + str(i)
            rain_y_n_name = 'RAIN_Y_N_' + str(i)
            X[rain_y_n_name] = np.nan
            X[rain_y_n_name].loc[X[col_name] > 0] = 1
            X[rain_y_n_name].loc[X[col_name] == 0] = 0
            X[rain_y_n_name] = X[rain_y_n_name].astype('object')
        return X


# In[9]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        print(X)
        return X[self.attribute_names].values


# In[10]:


class Convert_Date(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names = None):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Date'] = pd.to_timedelta(X['Date']).dt.total_seconds().astype(int)
        return X


# ## Data Processing Pipeline

# In[11]:


get_ipython().run_cell_magic('time', '', "soil_file_dir = '../data/soil/'\nweather_file_dir = '../data/weather/weather_data.csv'\nx = 0\n\npre_work_pipeline = Pipeline([\n    ('prework', Load_Data()),\n    ('features', Feature_Engineer())\n])\n\npre_work_pipeline.fit(weather_file_dir, soil_file_dir)\nprework_df = pre_work_pipeline.transform(x)\n#Save to CSV so that we do not need to import and clean data everytime\nprework_df.to_csv('clean_data.csv')")


# ## Make Classifier Label

# In[12]:


y_cols = ['VW_30cm', 'VW_60cm', 'VW_90cm', 'VW_120cm', 'VW_150cm']

for cols in y_cols:
    name = cols[3:] + '_class'
    prework_df[name] = ''
    prework_df[name].loc[(prework_df[cols] <= 0.1)] = '0.1'
    prework_df[name].loc[(prework_df[cols] > 0.1) & (prework_df[cols] <= 0.2)] = '0.2'
    prework_df[name].loc[(prework_df[cols] > 0.2) & (prework_df[cols] <= 0.3)] = '0.3'
    prework_df[name].loc[(prework_df[cols] > 0.3) & (prework_df[cols] <= 0.4)] = '0.4'
    prework_df[name].loc[(prework_df[cols] > 0.4) & (prework_df[cols] <= 0.5)] = '0.5'
    prework_df[name].loc[(prework_df[cols] > 0.5) & (prework_df[cols] <= 0.6)] = '0.6'
    prework_df[name].loc[(prework_df[cols] > 0.6) & (prework_df[cols] <= 0.7)] = '0.7'
    prework_df[name].loc[(prework_df[cols] > 0.7) & (prework_df[cols] <= 0.8)] = '0.8'
    prework_df[name].loc[(prework_df[cols] > 0.8)] = '0.9'


# ## Make Data Frames for Each Depth

# The moisture data is taken at various depths. We want to build models seperately for different depths. So we need to make a dataframe for each depth so that we can elminate entire rows where the predictor is NA
# 

# In[13]:


# First split out y values
all_y_cols = ['VW_30cm', 'VW_60cm', 'VW_90cm', 'VW_120cm', 'VW_150cm', '30cm_class', '60cm_class', '90cm_class', '120cm_class', '150cm_class']
X_sets = {}
y_sets = {}
x_cols = [col for col in prework_df.columns if col not in y_cols]
X = prework_df.loc[:, x_cols]
#y = prework_df.loc[:, y_cols]

for cols in all_y_cols:
    if cols[:1] == 'V':
        dataset_name = cols[3:]
    else:
        dataset_name = cols
    holder = prework_df.dropna(subset = [cols])
    X_sets[dataset_name] = holder[x_cols].fillna(0)
    y_sets[dataset_name] = holder[cols]


# ## Split Train and Test

# In[14]:


# Split training and test data
# 80-20 ratio
# Trying to keep same ratios for each location using stratify
# Could have done this in the cell above, but wanted a seperate step for this
X_train_set = {}
X_test_set = {}
y_train_set = {}
y_test_set = {}

for cols in all_y_cols:
    if cols[:1] == 'V':
        dataset_name = cols[3:]
    else:
        dataset_name = cols 
    X_train_set[dataset_name], X_test_set[dataset_name], y_train_set[dataset_name], y_test_set[dataset_name] = train_test_split(X_sets[dataset_name], y_sets[dataset_name],                                                                                                                                 test_size=0.2, stratify = X_sets[dataset_name]['Location'], random_state=42)


# ## Generic Pipeline

# In[15]:


num_attribs = X_train_set['60cm'].select_dtypes(exclude=['object', 'category']).columns
cat_attribs = X_train_set['60cm'].select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value = 0)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value = '')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_attribs),
        ('cat', categorical_transformer, cat_attribs)
    ])


# ## Linear Regression Tests

# ### Simple Test

# In[16]:


pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', LinearRegression())])

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
try:
    log
except NameError:
    log = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'datetime'])
    
for cols in data_cols:
    t0 = time.time()
    pipe_with_estimator.fit(X_train_set[cols], y_train_set[cols])
    t1 = time.time()
    preds = pipe_with_estimator.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.loc[len(log)] = ['First Linear Reg', cols, t1-t0, t2-t1, r2sc, now]
    
print(log)


# Great Scores, but oddly 60 cm has a very small r2 score

# Let's try lasso

# In[17]:


pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', Ridge(alpha = 1))])

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
try:
    log
except NameError:
    log = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'datetime'])
    
for cols in data_cols:
    t0 = time.time()
    pipe_with_estimator.fit(X_train_set[cols], y_train_set[cols])
    t1 = time.time()
    preds = pipe_with_estimator.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.loc[len(log)] = ['Ridge Reg - Alpha = 1', cols, t1-t0, t2-t1, r2sc, now]
    
print(log)


# ### Results are better! Let's try Lasso

# In[18]:


pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', Lasso(alpha = 1))])

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
try:
    log
except NameError:
    log = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'datetime'])
    
for cols in data_cols:
    t0 = time.time()
    pipe_with_estimator.fit(X_train_set[cols], y_train_set[cols])
    t1 = time.time()
    preds = pipe_with_estimator.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.loc[len(log)] = ['Lasso Reg - Alpha = 1', cols, t1-t0, t2-t1, r2sc, now]
    
print(log)


# ### At least with with these parameters, Lasso Fits Poorly

# ### Ridge with a built in gridsearch cross validation

# In[19]:


pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RidgeCV(alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]))])

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
try:
    log
except NameError:
    log = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'datetime'])
    
for cols in data_cols:
    t0 = time.time()
    pipe_with_estimator.fit(X_train_set[cols], y_train_set[cols])
    t1 = time.time()
    preds = pipe_with_estimator.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.loc[len(log)] = ['Ridge Reg - GSCV', cols, t1-t0, t2-t1, r2sc, now]
    
print(log)


# Gridsearch found alpha = 1 to be the best parameter

# ## Other Regressor Tests

# Right now Ridge Regression with an alpha of 1 is winning as the best model so far. Let's see if we can beat it

# In[20]:


pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestRegressor())])

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
try:
    log_other
except NameError:
    log_other = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'datetime'])
for cols in data_cols:
    t0 = time.time()
    pipe_with_estimator.fit(X_train_set[cols], y_train_set[cols])
    t1 = time.time()
    preds = pipe_with_estimator.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_other.loc[len(log_other)] = ['Random Forest - Default', cols, t1-t0, t2-t1, r2sc, now]
    
print(log_other)


# Amazing results! Although it takes considerably longer to train, the default does rather well

# As a litmus test, lets just try a few more models.

# In[21]:


pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', SVR())])

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
try:
    log_other
except NameError:
    log_other = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'datetime'])
for cols in data_cols:
    t0 = time.time()
    pipe_with_estimator.fit(X_train_set[cols], y_train_set[cols])
    t1 = time.time()
    preds = pipe_with_estimator.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_other.loc[len(log_other)] = ['SVM - Default', cols, t1-t0, t2-t1, r2sc, now]
    
print(log_other)


# Just with the default values, SVM, did not perform well, but this could just mean that default parameters are not good

# In[22]:


pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', SGDRegressor())])

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
try:
    log_other
except NameError:
    log_other = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'datetime'])
for cols in data_cols:
    t0 = time.time()
    pipe_with_estimator.fit(X_train_set[cols], y_train_set[cols])
    t1 = time.time()
    preds = pipe_with_estimator.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_other.loc[len(log_other)] = ['SGD - Default', cols, t1-t0, t2-t1, r2sc, now]
    
print(log_other)


# ## Hyper Parameter Tuning Random Forest

# The following, will take a considerable amount of time to run. Run with caution!!

# This experiment is not included in the final report, but shows an extension of trying to get better results.

# In[ ]:


## Param grid comes from the following site:
## https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

pipe_with_estimator = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestRegressor())])

param_grid = {'classifier__bootstrap': [True, False],
              'classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
              'classifier__max_features': ['auto', 'sqrt'],
              'classifier__min_samples_leaf': [1, 2, 4],
              'classifier__min_samples_split': [2, 5, 10],
              'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

data_cols = ['30cm', '60cm', '90cm', '120cm', '150cm']
cv_res = {}
try:
    log_rf
except NameError:
    log_rf = pd.DataFrame(columns = ['Experiment', 'Depth', 'Fit_Time', 'Pred_Time', 'r2_score', 'best_params' 'datetime'])
for cols in data_cols:
    t0 = time.time()
    random_search = RandomizedSearchCV(estimator = pipe_with_estimator, param_distributions = param_grid, n_iter = 10, cv = 3, verbose=10, random_state=42, n_jobs = -1)
    random_search.fit(X_train_set[cols], y_train_set[cols])
    best = random_search.best_params_
    t1 = time.time()
    preds = random_search.predict(X_test_set[cols])
    t2 = time.time()
    r2sc = r2_score(y_test_set[cols], preds)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_rf.loc[len(log_rf)] = ['RF - random search', cols, t1-t0, t2-t1, r2sc, best, now]
    cv_res[cols] = random_search.cv_results_
    print(log_rf)
    
print(log_rf)

