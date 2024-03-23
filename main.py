import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
for filename in filenames:
print(os.path.join(dirname, filename)
data = pd.read_csv('/content/Uncleaned_employees_final_dataset (1).csv') 
data.head()
data.isnull().sum() 
numerical_summary = data.describe()
categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel'] 
categorical_summary = data[categorical_columns].nunique().to_frame('Unique Values') 
numerical_summary, categorical_summary 
data['education'].fillna(data['education'].mode()[0], inplace=True) 
data['previous_year_rating'].fillna(data['previous_year_rating'].median(), inplace=True) 
data.isnull().sum()
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(15, 10)) 
sns.histplot(data['previous_year_rating'], ax=ax[0, 0], kde=False, bins=5) 
ax[0, 0].set_title('Distribution of Previous Year Rating') 
sns.histplot(data['KPIs_met_more_than_80'], ax=ax[0, 1], kde=False, bins=2) 
ax[0, 1].set_title('Distribution of KPIs Met More Than 80%') 
sns.histplot(data['awards_won'], ax=ax[1, 0], kde=False, bins=2)
ax[1, 0].set_title('Distribution of Awards Won') 
sns.histplot(data['avg_training_score'], ax=ax[1, 1], kde=False, bins=30) 
ax[1, 1].set_title('Distribution of Average Training Score') 
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(4, 2, figsize=(15, 20))
sns.boxplot(x='department', y='previous_year_rating', data=data, ax=ax[0, 0]) 
ax[0, 0].set_title('Previous Year Rating by Department')
ax[0, 0].tick_params(axis='x', rotation=90)
sns.boxplot(x='department', y='avg_training_score', data=data, ax=ax[0, 1]) 
ax[0, 1].set_title('Average Training Score by Department')
ax[0, 1].tick_params(axis='x', rotation=90)
sns.boxplot(x='education', y='previous_year_rating', data=data, ax=ax[1, 0]) 
ax[1, 0].set_title('Previous Year Rating by Education') 
sns.boxplot(x='education', y='avg_training_score', data=data, ax=ax[1, 1]) 
ax[1, 1].set_title('Average Training Score by Education') 
sns.boxplot(x='gender', y='previous_year_rating', data=data, ax=ax[2, 0])
ax[2, 0].set_title('Previous Year Rating by Gender') 
sns.boxplot(x='gender', y='avg_training_score', data=data, ax=ax[2, 1]) 
ax[2, 1].set_title('Average Training Score by Gender')
sns.boxplot(x='recruitment_channel', y='previous_year_rating', data=data, ax=ax[3, 0]) 
ax[3, 0].set_title('Previous Year Rating by Recruitment Channel') 
sns.boxplot(x='recruitment_channel', y='avg_training_score', data=data, ax=ax[3, 1]) 
ax[3, 1].set_title('Average Training Score by Recruitment Channel')
plt.tight_layout() 
plt.show()
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
sns.scatterplot(x='no_of_trainings', y='previous_year_rating', data=data, ax=ax[0, 0]) 
ax[0, 0].set_title('Previous Year Rating vs. Number of Trainings') 
sns.scatterplot(x='no_of_trainings', y='avg_training_score', data=data, ax=ax[0, 1]) 
ax[0, 1].set_title('Average Training Score vs. Number of Trainings') 
sns.scatterplot(x='age', y='previous_year_rating', data=data, ax=ax[1, 0])
ax[1, 0].set_title('Previous Year Rating vs. Age')
sns.scatterplot(x='age', y='avg_training_score', data=data, ax=ax[1, 1]) 
ax[1, 1].set_title('Average Training Score vs. Age')
plt.tight_layout() 
plt.show()
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
features = data.drop(['employee_id', 'previous_year_rating'], axis=1) 
target = data['previous_year_rating']
categorical_features = features.select_dtypes(include=['object']).columns 
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = pd.DataFrame(encoder.fit_transform(features[categorical_features])) 
features = features.drop(categorical_features, axis=1)
features = pd.concat([features, encoded_features], axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, 
random_state=42)
X_train.shape, X_test.shape
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
import numpy as np
model = LinearRegression() 
features.columns
X_train.columns = X_train.columns.astype(str) 
X_test.columns = X_test.columns.astype(str) 
model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions)) 
rmse