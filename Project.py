#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, learning_curve, cross_val_score, train_test_split

from joblib import dump
import warnings
warnings.filterwarnings('ignore')



# In[2]:


# Loading the dataset from the uploaded file
file_path = 'D:/Student Prediction/studentper.csv'  # Update the file name accordingly
df = pd.read_csv(file_path)
# Display a sample of the data
print(df.sample(10))


# # DATA EXPLORATION

# ## statistical descriptions of data before pre-processing

# In[3]:


#Mean
#df.shape
stat_descriptions = df.describe()
print(stat_descriptions)


# ### Plot histogram,correlation & countplot for dataset

# In[4]:


#df.info()


def plot_corr(df, annot=True):
    numeric_columns = df._get_numeric_data().columns
    _, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        df[numeric_columns].corr(method='pearson'),
        annot=annot,
        cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True),
        ax=ax
    )
    _.set_facecolor("whitesmoke")
    



def plot_histplot(column):
    sns.histplot(x=column, color='#65b87b', alpha=.7) 
    
    
def plot_countplot(df, column_name, ax=None):
    _df = df[[column_name]].copy()
    if len(_df[_df[column_name].isnull()]):
        _df.fillna('NaN', inplace=True)
    
    color = '#42b0f5' if ax != None else '#7661ff'
    sns.countplot(x=column_name, data=_df, color=color, alpha=.7, ax=ax)
    del _df




# ### Pearson Correlation between all the features after using one hot encoding to convert categorical features 
The heatmap of the correlation looks blury because the datapints are large in size to be shown clearly. Categorical features are well correlated with each other and thus exclude them from our final descriptive features used for training the model and also because Linear Regression works better with numeric values
# In[5]:


categorical_cols = df.select_dtypes(include='object').columns

ohe_df = pd.get_dummies(df,columns = categorical_cols)
ohe_df.head()
plot_corr(ohe_df, annot=True)


# #### Histogram of the proposed target class G3 shows that the scores are skewed right which is a good  thing given that exams scores tend to be around and above average in the real-world

# In[6]:


plot_histplot(df['G3'])


# ### Pearson Correlation between the numeric features to be taken into consideration for training the model

# In[7]:


plot_corr(df, annot=True)

Walc has moderate positive correlation with Dalc and they both have negligible negative correlation

failure has low negative correlation with grades

studytime & grades have a low positive correlation

age has low positive correlation with failure

freetime has low positive correlation with goout

goout has low positive correlation with Walc

Medu & Fedu has moderate positive correlation & they both have low positive correlation with grades
# ### Correlation of the Multi-Class targets- G1, G2, & G3- they are highly correlated

# In[8]:


df_targets= df[['G1','G2','G3']]
plot_corr(df_targets)


# ### Above are charts of data exploration of categorical features

# In[9]:


def plot_base_categorical_relation(df, figsize=(20, 60)):
    columns = df.columns.tolist()
    _, axs = plt.subplots(len(columns), 3, figsize=figsize)
    
    for idx, column in enumerate(columns):
        try:
            # To get knowledge about outliers & distribution
            sns.boxplot(x=df[column], y=df['G3'], ax=axs[idx][0])

            # To get its realtion with Average Grades
            sns.stripplot(
                x=column, y='G3', data=df,
                color='#706dbd', alpha=.7, jitter=.1,
                ax=axs[idx][1]
            )

            # To get count plot for `column` (considering NaN, so we can know 
            # how much of data is missing)
            plot_countplot(df, column, axs[idx][2])
        except ValueError:
            # ValueError: min() arg is an empty sequence
            # 
            # The above error happens while creating plot for some columns (maybe 
            # because it has NaN value)
            print(f'{column} cannot be plotted')
        
        
plot_base_categorical_relation(
    pd.concat(
        [df.select_dtypes(include=['object']), df[['G3']]],
        axis='columns'
    )
)

Since G1, G2 & G3 are highly correlated, We take only G3 and try a novel approach of feeding G1 & G2 as descriptive features to train the model for robust predictions
# In[10]:


# Plotting base relations
def plot_base_relation(df, figsize=(20, 60)):
    columns = df.columns.tolist()
    _, axs = plt.subplots(len(columns), 3, figsize=figsize)

    for idx, column in enumerate(columns):
        # To get distribution of data
        sns.histplot(
            x=df[column],
            kde=False,
            color='#65b87b', alpha=.7,
            ax=axs[idx][0]
        )

        # To get knowledge about outliers
        sns.boxplot(
            x=df[column],
            color='#6fb9bd',
            ax=axs[idx][1]
        )

        # To get its relation with G3
        sns.scatterplot(
            x=column, y='G3', data=df,
            color='#706dbd', alpha=.7, s=80,
            ax=axs[idx][2]
        )
    plt.show()
new_df= df.select_dtypes(include=[int, float])
print(new_df.head())
plot_base_relation(df.select_dtypes(include=[int, float]))


# ### The above charts is a data exploration of the numeric features of the dataset

# ## Modelling Phase
# ## Creating a regression model that can predict student's final grades.

# In[11]:


# KFold for cross validation
kf = KFold(n_splits=10, shuffle=True)


# In[12]:


# Shuffling the dataset
numeric_df = df.select_dtypes(include='number')

numeric_df = numeric_df.sample(frac=1, random_state=5)
numeric_df.info


# ## Deriving Model without G1 and G2 as descriptive features

# In[13]:


# Selecting features by analysing which features are collinear to `G3` and collinear 
# to the selected columns
features = ['failures', 'Medu', 'studytime', 'absences']
target = 'G3'

x_train, x_test, y_train, y_test = train_test_split(
    numeric_df[features], numeric_df[target], test_size=0.3, random_state=0
)


# In[14]:


# Scaling the dataset

scaler = StandardScaler()

x_train = scaler.fit_transform(np.asanyarray(x_train))
y_train = np.asanyarray(y_train)

x_test = scaler.fit_transform(np.asanyarray(x_test))
y_test = np.asanyarray(y_test)


# In[15]:


# Cross Validation
scoring = 'r2'
score = cross_val_score(linear_model.LinearRegression(), x_train, y_train, cv=4, scoring=scoring)
score.mean()


# In[16]:


# Plotting learning curve
_sizes = [i for i in range(1, 408, 10)]
train_sizes = np.array([_sizes])  # Relative sizes
scoring = 'neg_mean_squared_error'

lr = linear_model.LinearRegression()
train_sizes_abs, train_scores, cv_scores = learning_curve(
    lr, x_train, y_train, train_sizes=train_sizes, cv=10, scoring=scoring
)


# In[17]:


train_scores_mean = []
for row in train_scores:
    _mean = row.mean()
    train_scores_mean.append(_mean)
    
cv_scores_mean = []
for row in cv_scores:
    _mean = row.mean()
    cv_scores_mean.append(_mean)    
    
train_scores_mean = -np.array(train_scores_mean)
cv_scores_mean = -np.array(cv_scores_mean)
    
print(train_scores_mean)
print()
print(cv_scores_mean)


# In[18]:


plt.plot(train_sizes_abs, train_scores_mean, label='Train')
plt.plot(train_sizes_abs, cv_scores_mean, label='Cross Validation')

plt.legend()


# In[19]:


# Fitting the model
model = lr.fit(x_train, y_train)


# In[20]:


# Optimal parameter
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients: ", coefficients)
print("Intercept: ", model.intercept_)


# ## Evaluation

# In[21]:


y_test_pred = model.predict(x_test)


# In[22]:


# To see how our model performs on data that model has NOT seen

rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
r2_score_value = r2_score(y_test, y_test_pred)

print(f"Root mean squared error: {rms_error}")
print(f"R2-score: {r2_score_value}")


# ## Creating a pipeline

# In[23]:


scaling = ('scale', StandardScaler())
model = ('model', linear_model.LinearRegression())

# Steps in the pipeline
steps = [scaling, model]

pipe = Pipeline(steps=steps)

# Fiitting the model
model = pipe.fit(x_train, y_train)

# Out-Of-Sample Forecast
y_test_pred = model.predict(x_test)

# Evaluation
rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
r2_score_value = r2_score(y_test, y_test_pred)

print(f"Root mean squared error: {rms_error}")
print(f"R2-score: {r2_score_value}")


# In[24]:


# Saving the model
dump(model, 'model.joblib')


# ## Visualizing our prediction against actual values.

# In[25]:


f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))

ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax2.plot(np.arange(len(y_test_pred)), y_test_pred, label='Prediction')

ax1.legend()
ax2.legend()

f, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(13, 5))

ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred)), y_test_pred, label='Prediction')

ax3.legend()


# ## Deriving Model with G1 and G2 as descriptive features

# In[26]:


# Selecting features by analysing which features are collinear to `G3` and collinear 
# to the selected columns
features = ['failures', 'Medu', 'studytime', 'absences','G1','G2']
target = 'G3'

x_train, x_test, y_train, y_test = train_test_split(
    numeric_df[features], numeric_df[target], test_size=0.3, random_state=0
)

# Scaling the dataset

scaler = StandardScaler()

x_train = scaler.fit_transform(np.asanyarray(x_train))
y_train = np.asanyarray(y_train)

x_test = scaler.fit_transform(np.asanyarray(x_test))
y_test = np.asanyarray(y_test)

# Cross Validation
scoring = 'r2'
score = cross_val_score(linear_model.LinearRegression(), x_train, y_train, cv=4, scoring=scoring)
score.mean()

# Plotting learning curve
_sizes = [i for i in range(1, 408, 10)]
train_sizes = np.array([_sizes])  # Relative sizes
scoring = 'neg_mean_squared_error'

lr = linear_model.LinearRegression()
train_sizes_abs, train_scores, cv_scores = learning_curve(
    lr, x_train, y_train, train_sizes=train_sizes, cv=10, scoring=scoring
)


train_scores_mean = []
for row in train_scores:
    _mean = row.mean()
    train_scores_mean.append(_mean)
    
cv_scores_mean = []
for row in cv_scores:
    _mean = row.mean()
    cv_scores_mean.append(_mean)    
    
train_scores_mean = -np.array(train_scores_mean)
cv_scores_mean = -np.array(cv_scores_mean)
    
print(train_scores_mean)
print()
print(cv_scores_mean)

plt.plot(train_sizes_abs, train_scores_mean, label='Train')
plt.plot(train_sizes_abs, cv_scores_mean, label='Cross Validation')

plt.legend()

# Fitting the model
model = lr.fit(x_train, y_train)

# Optimal parameter
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients: ", coefficients)
print("Intercept: ", model.intercept_)


# ## Evaluation

# In[27]:


y_test_pred = model.predict(x_test)
# To see how our model performs on data that model has NOT seen

rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
r2_score_value = r2_score(y_test, y_test_pred)

print(f"Root mean squared error: {rms_error}")
print(f"R2-score: {r2_score_value}")


scaling = ('scale', StandardScaler())
model = ('model', linear_model.LinearRegression())

# Steps in the pipeline
steps = [scaling, model]

pipe = Pipeline(steps=steps)

# Fiitting the model
model = pipe.fit(x_train, y_train)

# Out-Of-Sample Forecast
y_test_pred = model.predict(x_test)

# Evaluation
rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
r2_score_value = r2_score(y_test, y_test_pred)

print(f"Root mean squared error: {rms_error}")
print(f"R2-score: {r2_score_value}")

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))

ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax2.plot(np.arange(len(y_test_pred)), y_test_pred, label='Prediction')

ax1.legend()
ax2.legend()

f, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(13, 5))

ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred)), y_test_pred, label='Prediction')

ax3.legend()


# In[ ]:





# In[ ]:





# In[ ]:




