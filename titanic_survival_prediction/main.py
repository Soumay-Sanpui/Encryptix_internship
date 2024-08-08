# -*- coding: utf-8 -*-
"""Encryptix

Original file is located at
    https://colab.research.google.com/drive/1BH05_7bZ7qA0WXnHH1Pa9I1rycnTaPJX

# Titanic Survival Prediction

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings

titanic_data = pd.read_csv("./Titanic-Dataset.csv")
titanic_data.head()
titanic_data.info()
titanic_data.isnull().sum()

#removing null values
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

#replacing missing values with mean
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)

titanic_data.info()
titanic_data.isnull().sum()

#fixing emabarked
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])

#replacing missing with mode
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)
titanic_data.isnull().sum()

"""Analysing the data"""
titanic_data.describe()
#Survived Passengers
titanic_data['Survived'].value_counts()

#visualizing data
sns.set()
sns.countplot(x=titanic_data['Survived'])

#count of survivals wrt Gender
sns.countplot(x='Sex', hue='Survived', data=titanic_data)

#count of survivals wrt Pclass
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

# survival rate by SEX
titanic_data.groupby('Sex')[['Survived']].mean()
titanic_data['Sex'].unique()

# data labeling

labelencoder = LabelEncoder()
titanic_data['Sex']=labelencoder.fit_transform(titanic_data['Sex'])
titanic_data.head()

titanic_data['Sex'], titanic_data['Survived']

sns.countplot(x='Sex', hue='Survived', data=titanic_data)
titanic_data.isnull().sum()

df_final= titanic_data
df_final.head()

"""TRAINING REGRESSION MODEL"""
X=titanic_data[['Pclass','Sex']]
Y=titanic_data['Survived']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

log = LogisticRegression(random_state=0)
log.fit(X_train,Y_train)

pred = print(log.predict(X_test))
print(Y_test)


warnings.filterwarnings("ignore")
res = log.predict([[2,0]])
if(res==0):
  print("So Sorry! Not Survived")
else:
  print("Survived")