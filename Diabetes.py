## import required Library

import pandas as pd
import pickle

## Loading the dataset
df = pd.read_csv("E:\Exposys\Diabetes_Prediction\diabetes.csv")
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

## copying the dataset
df_copy = df.copy(deep = True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure', 'SkinThickness','Insulin','BMI']]

## Replacing by NaN values with mean and median
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace = True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace = True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace = True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace = True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace = True)

## Divide into Independent and Dependent features
x = df.drop('Outcome', axis=1)
y = df['Outcome']

## split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=7)

## selecting  the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(x_train, y_train)

## creating pickle file
Pkl_Filename = "dp.pkl"
with open(Pkl_Filename,'wb') as file:
    pickle.dump(classifier, file)
