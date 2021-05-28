import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

uber=pd.read_csv("taxi.csv")
#print(uber.head())
print(uber.columns)

## so Numberofweeklyriders is dependent variable and rest all are independent variable

X = uber.iloc[:,0:-1].values
y = uber.iloc[:,-1].values

#print(X)
#print(y)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lr=LinearRegression()

lr.fit(x_train,y_train)
print("Training Score : ",lr.score(x_train,y_train))
print("Test Score : ",lr.score(x_test,y_test))

pickle.dump(lr,open('uber.pkl','wb'))

#to test the model
model=pickle.load(open('uber.pkl','rb'))
print(model.predict([[30, 1250000, 6000, 85]]))