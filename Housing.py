# Import Libraries to be used
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import datetime as dt

# Import Dataset and observe it
data = pd.read_csv('/Users/laolu/Documents/DataProjects/kc_house_data.csv')
data.shape # 21,613 rows and 21 Colimns/Variables
data.dtypes
''' Floats and Integers mostly. Except Date and that needs to change.'''
data.isnull().sum() #No missing Values.
desription = data.describe()

#Work on the columns/Features
data.waterfront.nunique()
data.waterfront.unique() # 0 for No waterfront, 1 For waterfront
data.zipcode.nunique() #70 Unique Zip Codes
data = data.drop(['lat','long'], axis = 1) #Zip Code already represented by lat and long

data['date']= pd.to_datetime(data['date']) #Change date from string to datetime
data['Year']= data['date'].dt.year #Create Column called Year. Year of sale may influence Price
data['Month']= data['date'].dt.month #Create Column called Month. Month of sale may influence Price

data['Year'].nunique() #Dataset contains property sold in 2 years. Hence 2014/2015
''' Column showing how many years the house has existed for before sale'''
data['Built Year Diff'] = data['Year'] -  data['yr_built']
data['Built Year Diff'].describe() #Oldest building is 115 Years
data['yr_built'].max()

# Is it possible they sold these houses before they were built?
observe_built = data.loc[data['Built Year Diff'] == -1]
data = data.drop(data[data['Built Year Diff'] == -1].index) #Dropped theses rows as i could not prove it.
data['yr_renovated'].describe()

data['renovate_diff']= data['Built Year Diff']-(data['Year']-data['yr_renovated'] )

#Create a column showing if house has been renovated or not
data['Renov or Not'] = np.where((data['yr_renovated'] > 0) | (data['yr_renovated'] < 0), 1, 0)
data = data.drop(['renovate_diff'], axis = 1)
data = data.drop(['date', 'id'], axis = 1)

data_corr = data.corr()
#Visualize Data
sns.heatmap(data_corr, linewidths=0.5,annot=True)
plt.figure(figsize=(15, 10))
plt.show()

sns.pairplot(data)
plt.figure(figsize=(15, 10))
plt.show()

#Create dummies firstly, change categorical integers to strings
data['yr_built'] = data['yr_built'].astype(str)
data['zipcode'] = data['zipcode'].astype(str)
data['Year'] = data['Year'].astype(str)
data['Month'] = data['Month'].astype(str
data['Renov or Not'] = data['Renov or Not'].astype(str)
data['waterfront'] = data['waterfront'].astype(str)
data['view'] = data['view'].astype(str)
data['condition'] = data['condition'].astype(str)
data['grade'] = data['grade'].astype(str)

#get dummy for all categorical strings
data = pd.get_dummies(data)

#Select Features and target variable
X = data.loc[:,'bedrooms':]
y=data['price'].values

#Split into test and training dataset
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=0)

#Try Linear regression
ln = LinearRegression()
ln.fit(X_train,y_train)
ln.score(X_train,y_train)
ln.score(X_test,y_test)

preditcion = ln.predict(X_test)

# there doesnt seem to be overfitting. But try Lasso
rd = Lasso(alpha=0.2)

rd.fit(X_train,y_train)
rd.score(X_train,y_train)
rd.score(X_test,y_test)

preditcion2 = rd.predict(X_test)
