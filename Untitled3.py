#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# Loading the dataset from the uploaded file
file_path = 'D:/Student Prediction/studentper.csv'  # Update the file name accordingly
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

#Mean
#df.shape
stat_descriptions = df.describe()
print(stat_descriptions)

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



categorical_cols = df.select_dtypes(include='object').columns

ohe_df = pd.get_dummies(df,columns = categorical_cols)
ohe_df.head()
plot_corr(ohe_df, annot=True)

plot_histplot(df['G3'])


plot_corr(df, annot=True)


df_targets= df[['G1','G2','G3']]
plot_corr(df_targets)


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


# In[3]:


# KFold for cross validation
kf = KFold(n_splits=10, shuffle=True)


# Shuffling the dataset
numeric_df = df.select_dtypes(include='number')









# In[4]:


numeric_df = numeric_df.sample(frac=1, random_state=5)
numeric_df.info







# In[5]:


# Selecting features by analysing which features are collinear to `G3` and collinear 
# to the selected columns
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
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


# In[6]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:


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

# Loading the dataset from the uploaded file
file_path = 'D:/Student Prediction/studentper.csv'  # Update the file name accordingly
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Mean
# df.shape
stat_descriptions = df.describe()
print(stat_descriptions)

# df.info()


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


categorical_cols = df.select_dtypes(include='object').columns

ohe_df = pd.get_dummies(df, columns=categorical_cols)
ohe_df.head()
plot_corr(ohe_df, annot=True)

plot_histplot(df['G3'])

plot_corr(df, annot=True)

df_targets = df[['G1', 'G2', 'G3']]
plot_corr(df_targets)


def plot_base_categorical_relation(df, figsize=(20, 60)):
    columns = df.columns.tolist()
    _, axs = plt.subplots(len(columns), 3, figsize=figsize)

    for idx, column in enumerate(columns):
        try:
            # To get knowledge about outliers & distribution
            sns.boxplot(x=df[column], y=df['G3'], ax=axs[idx][0])

            # To get its relation with Average Grades
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
            # The above error happens while creating a plot for some columns (maybe
            # because it has NaN value)
            print(f'{column} cannot be plotted')


plot_base_categorical_relation(
    pd.concat(
        [df.select_dtypes(include=['object']), df[['G3']]],
        axis='columns'
    )
)


# Plotting base relations
def plot_base_relation(df, figsize=(20, 60)):
    columns = df.columns.tolist()
    _, axs = plt.subplots(len(columns), 3, figsize=figsize)

    for idx, column in enumerate(columns):
        # To get the distribution of data
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


new_df = df.select_dtypes(include=[int, float])
print(new_df.head())
plot_base_relation(df.select_dtypes(include=[int, float]))  # KFold for cross-validation
kf = KFold(n_splits=10, shuffle=True)

# Shuffling the dataset
numeric_df = df.select_dtypes(include='number')
numeric_df = numeric_df.sample(frac=1, random_state=5)
numeric_df.info

# Selecting features by analyzing which features are collinear to `G3` and collinear
# to the selected columns
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
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

# Fitting the model
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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, RepeatedKFold

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")

evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_value_base > r2_score_value_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RepeatedKFold
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Function to plot correlation heatmap
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

# Plot correlation heatmap
plot_corr(df)

# Function to plot histogram
def plot_histplot(column):
    sns.histplot(x=column, color='#65b87b', alpha=.7)

# Plot histogram for 'G3'
plot_histplot(df['G3'])

# Function to plot count plot
def plot_countplot(df, column_name, ax=None):
    _df = df[[column_name]].copy()
    if len(_df[_df[column_name].isnull()]):
        _df.fillna('NaN', inplace=True)

    color = '#42b0f5' if ax != None else '#7661ff'
    sns.countplot(x=column_name, data=_df, color=color, alpha=.7, ax=ax)
    del _df

# Plot count plot for selected columns
plot_countplot(df, 'failures')
plot_countplot(df, 'Medu')
plot_countplot(df, 'studytime')
plot_countplot(df, 'absences')
plot_countplot(df, 'G1')
plot_countplot(df, 'G2')

# Function to plot base categorical relations
def plot_base_categorical_relation(df, figsize=(20, 60)):
    columns = df.columns.tolist()
    _, axs = plt.subplots(len(columns), 3, figsize=figsize)

    for idx, column in enumerate(columns):
        try:
            sns.boxplot(x=df[column], y=df['G3'], ax=axs[idx][0])
            sns.stripplot(
                x=column, y='G3', data=df,
                color='#706dbd', alpha=.7, jitter=.1,
                ax=axs[idx][1]
            )
            plot_countplot(df, column, axs[idx][2])
        except ValueError:
            print(f'{column} cannot be plotted')

# Plot base categorical relations
plot_base_categorical_relation(
    pd.concat(
        [df.select_dtypes(include=['object']), df[['G3']]],
        axis='columns'
    )
)

# Function to plot base relations
def plot_base_relation(df, figsize=(20, 60)):
    columns = df.columns.tolist()
    _, axs = plt.subplots(len(columns), 3, figsize=figsize)

    for idx, column in enumerate(columns):
        sns.histplot(
            x=df[column],
            kde=False,
            color='#65b87b', alpha=.7,
            ax=axs[idx][0]
        )
        sns.boxplot(
            x=df[column],
            color='#6fb9bd',
            ax=axs[idx][1]
        )
        sns.scatterplot(
            x=column, y='G3', data=df,
            color='#706dbd', alpha=.7, s=80,
            ax=axs[idx][2]
        )
    plt.show()

# Plot base relations
new_df = df.select_dtypes(include=[int, float])
print(new_df.head())
plot_base_relation(df.select_dtypes(include=[int, float]))

# KFold for cross-validation
kf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# Shuffling the dataset
numeric_df = df.select_dtypes(include='number')
numeric_df = numeric_df.sample(frac=1, random_state=5)
numeric_df.info()

# Selecting features by analyzing which features are collinear to `G3` and collinear 
# to the selected columns
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

x_train, x_test, y_train, y_test = train_test_split(
    numeric_df[features], numeric_df[target], test_size=0.3, random_state=0
)

# Scaling the dataset
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(np.asanyarray(x_train))
y_train_scaled = np.asanyarray(y_train)

x_test_scaled = scaler.fit_transform(np.asanyarray(x_test))
y_test_scaled = np.asanyarray(y_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train_scaled)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, random_state=42)

# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    # Add other hyperparameters if needed
}

# Perform grid search
grid_search = GridSearchCV(
    bagging_model,
    param_grid,
    cv=kf,
    scoring='r2',
    n_jobs=-1
)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train_scaled)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train_scaled)

# Evaluate base model
y_test_pred_base = base_lr_model.predict(x_test_scaled)
rms_error_base = mean_squared_error(y_test_scaled, y_test_pred_base, squared=False)
r2_score_value_base = r2_score(y_test_scaled, y_test_pred_base)

print("Base Linear Regression Model:")
print(f"Root mean squared error: {rms_error_base}")
print(f"R2-score: {r2_score_value_base}")

# Evaluate bagging model
y_test_pred_bagging = bagging_model.predict(x_test_scaled)
rms_error_bagging = mean_squared_error(y_test_scaled, y_test_pred_bagging, squared=False)
r2_score_value_bagging = r2_score(y_test_scaled, y_test_pred_bagging)

print("Bagging Model:")
print(f"Root mean squared error: {rms_error_bagging}")
print(f"R2-score: {r2_score_value_bagging}")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_value_base > r2_score_value_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test_scaled, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test_scaled, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")

evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
r2_score_value_base = r2_score(y_test, base_lr_model.predict(x_test_scaled))
r2_score_value_bagging = r2_score(y_test, bagging_model.predict(x_test_scaled))
best_model = base_lr_model if r2_score_value_base > r2_score_value_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RepeatedKFold

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import linear_model, preprocessing

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, learning_curve, cross_val_score, train_test_split
from joblib import dump

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RepeatedKFold

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import linear_model, preprocessing

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, learning_curve, cross_val_score, train_test_split
from joblib import dump

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")

evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_base > r2_score_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")


# In[ ]:





# In[ ]:





# In[52]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")
    return r2_score_value

# Calculate R2 scores
r2_score_base = evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
r2_score_bagging = evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_base > r2_score_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")
    return r2_score_value

# Calculate R2 scores
r2_score_base = evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
r2_score_bagging = evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_base > r2_score_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")





# Displaying the plots
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Plotting difference between actual and predicted values
difference = y_test - y_test_pred_best
ax2.plot(np.arange(len(difference)), difference, label='Difference')
ax2.set_title('Difference between Actual and Predicted Values')
ax2.legend()

# Plotting actual vs predicted values in a single plot
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.plot(np.arange(len(difference)), difference, label='Difference', linestyle='--')
ax3.set_title('Actual vs Predicted Values with Difference')
ax3.legend()

# Displaying the plots
plt.show()





# Calculate the absolute difference between actual and predicted values
absolute_difference = np.abs(y_test - y_test_pred_best)

# Displaying the plots
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend()

# Displaying the plots
plt.show()





















# Displaying the plots with a secondary y-axis for the absolute difference
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference with a secondary y-axis
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with a secondary y-axis for the absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.set_ylabel('Actual/Predicted Values')
ax3_twin = ax3.twinx()
ax3_twin.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--', color='red')
ax3_twin.set_ylabel('Absolute Difference')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Displaying the plots
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")
    return r2_score_value

# Calculate R2 scores
r2_score_base = evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
r2_score_bagging = evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_base > r2_score_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")

# Displaying the plots
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Plotting difference between actual and predicted values
difference = y_test - y_test_pred_best
ax2.plot(np.arange(len(difference)), difference, label='Difference')
ax2.set_title('Difference between Actual and Predicted Values')
ax2.legend()

# Plotting actual vs predicted values in a single plot
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.plot(np.arange(len(difference)), difference, label='Difference', linestyle='--')
ax3.set_title('Actual vs Predicted Values with Difference')
ax3.legend()

# Calculate the absolute difference between actual and predicted values
absolute_difference = np.abs(y_test - y_test_pred_best)

# Displaying the plots
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend()

# Displaying the plots with a secondary y-axis for the absolute difference
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference with a secondary y-axis
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with a secondary y-axis for the absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.set_ylabel('Actual/Predicted Values')
ax3_twin = ax3.twinx()
ax3_twin.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--', color='red')
ax3_twin.set_ylabel('Absolute Difference')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Displaying the plots
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[59]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")
    return r2_score_value

# Calculate R2 scores
r2_score_base = evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
r2_score_bagging = evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_base > r2_score_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")

# Displaying the plots
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Plotting difference between actual and predicted values
difference = y_test - y_test_pred_best
ax2.plot(np.arange(len(difference)), difference, label='Difference')
ax2.set_title('Difference between Actual and Predicted Values')
ax2.legend()

# Plotting actual vs predicted values in a single plot
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.plot(np.arange(len(difference)), difference, label='Difference', linestyle='--')
ax3.set_title('Actual vs Predicted Values with Difference')
ax3.legend()

# Calculate the absolute difference between actual and predicted values
absolute_difference = np.abs(y_test - y_test_pred_best)

# Displaying the plots
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend()

# Displaying the plots with a secondary y-axis for the absolute difference
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference with a secondary y-axis
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with a secondary y-axis for the absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.set_ylabel('Actual/Predicted Values')
ax3_twin = ax3.twinx()
ax3_twin.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--', color='red')
ax3_twin.set_ylabel('Absolute Difference')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Displaying the plots
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")
    return r2_score_value

# Calculate R2 scores
r2_score_base = evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
r2_score_bagging = evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_base > r2_score_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")


# Displaying the plots with a secondary y-axis for the absolute difference
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference with a secondary y-axis
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with a secondary y-axis for the absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.set_ylabel('Actual/Predicted Values')
ax3_twin = ax3.twinx()
ax3_twin.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--', color='red')
ax3_twin.set_ylabel('Absolute Difference')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Adjust x-axis limits for better visibility in all subplots
x_range = 50  # Adjust this value based on your preference
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, x_range)

# Displaying the plots
plt.show()


# In[ ]:




