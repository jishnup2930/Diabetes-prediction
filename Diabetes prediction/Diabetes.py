import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('diabetes.csv')
x = data.drop('Outcome',axis=1)
y = data.Outcome
scaler = StandardScaler()
scaled_x= scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(scaled_x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)