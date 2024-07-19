#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # EDA

# In[5]:


df = pd.read_csv('traffic.csv')


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df.sample(5)


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


df.duplicated().sum()


# In[13]:


df['Traffic Situation'].value_counts()


# In[14]:


# Calculate the total of the 'CarCount','Bike Count','Bus Count','Truck Count'
total_car_count = df['CarCount'].sum()
total_Bike_count = df['BikeCount'].sum()
total_Bus_count = df['BusCount'].sum()
total_Truck_count = df['TruckCount'].sum()
total_count = df['Total'].sum()

print("Total Car Count:", total_car_count)
print("Total Bike Count:", total_Bike_count)
print("Total Bus Count:", total_Bus_count)
print("Total Truck Count:", total_Truck_count)
print("Total VEHICLE COUNT:", total_count)


# In[ ]:





# In[15]:


sns.pairplot(df)
plt.show()


# In[16]:


sns.displot(data=df,x='Total',kind='kde',hue='Traffic Situation',fill=True,height=8,aspect=4)


# In[18]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[44]:


# Dark red color shows the strong positive correlation


# In[19]:


sns.relplot(data=df, x='Total', y='Day of the week', kind='scatter',hue='Traffic Situation')


# In[20]:


sns.boxplot(x="Traffic Situation", y="Total", data=df)


# In[21]:


plt.figure(figsize=(10,5))
sns.histplot(data=df, x="Day of the week", hue="Traffic Situation")


# In[ ]:





# In[22]:


df['Traffic Situation'].value_counts().plot(kind='pie',autopct='%0.1f%%')


# # Extracted from above EDA
# ()Out of 9 column 6 are integer and 3 object column
# 
# ()there is no null value and 0 duplicate 
# 
# ()Number of Cars(CarCount) has the most contribution to Traffic
# 
# ()Thursday and Wednesday are the most busy days for traffic
# 
# ()Peak hours of traffic are between 8:00am-10:00am and 3:00pm-6:00pm
# 
# ()Normal traffic situation counts the most
# 
# ()Heavy Traffic mostly occurs after 9:00pm
# 
# ()Friday seems to be minimum Traffic
# 
# ()Heavy traffic after 150 vehicles

# In[ ]:





# # Feature Engineering

# In[ ]:





# In[23]:


day_mapping = {'Monday': 1,'Tuesday': 2,'Wednesday': 3,'Thursday': 4,'Friday': 5,'Saturday': 6,'Sunday': 7}


df['Day of the week'] = df['Day of the week'].replace(day_mapping)


# In[24]:


traffic_mapping = {'low': 1,'normal': 2,'high': 3,'heavy': 4}

df['Traffic Situation'] = df['Traffic Situation'].replace(traffic_mapping)




# In[ ]:





# In[ ]:





# In[25]:


df['hour'] = pd.to_datetime(df['Time']).dt.hour 
df['minute'] = pd.to_datetime(df['Time']).dt.minute 
#df['seconds'] = pd.to_datetime(df['Time']).dt.second


# In[26]:


def convert_am_pm(time_str):
    if time_str.endswith('AM'):
        return 0
    elif time_str.endswith('PM'):
        return 1
    else:
        return None  


df['AM_PM'] = df['Time'].apply(lambda x: convert_am_pm(x) if isinstance(x, str) else None)


# In[27]:


df


# In[28]:


df.drop(columns=['Time'], inplace=True)


# In[29]:


df


# In[ ]:


# there are 3 object column 2 are string and 1 is datetime with respected to am and pm converted all into numeric form


# In[ ]:





# In[ ]:





# In[ ]:





# # Splitting data

# In[40]:


feature_columns= ['Date','Day of the week','CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'hour', 'minute', 'AM_PM']
target_column= ['Traffic Situation']


# In[41]:


X = df[feature_columns]
y = df[target_column]


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Model Selection and Trainning

# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))


# # Hyperparameter Tunning

# In[35]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
}

# GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

print("Best parameters found: ", grid_search_rf.best_params_)
print("Best cross-validation score: ", grid_search_rf.best_score_)

# Evaluate the best model
best_model_rf = grid_search_rf.best_estimator_
y_pred_rf = best_model_rf.predict(X_test)
print("Random Forest (Tuned) Accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Random Forest (Tuned) Classification Report:\n", classification_report(y_test, y_pred_rf))


# In[ ]:




