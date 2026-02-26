# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Uses Stochastic Gradient Descent SGD Regressor updates model weights using one training sample at a time instead of the entire dataset.
2️. Fast and Efficient It works well for large datasets because it requires less memory and computation.

3️. Needs Feature Scaling The algorithm performs better when input features are scaled using methods like StandardScaler.

4️. Supports Regularization It can apply L1, L2, or ElasticNet regularization to prevent overfitting.

5️. Iterative Learning Process It improves the model step-by-step through multiple iterations until it reaches minimum error or tolerance leve


## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Varoodhini.M
RegisterNumber: 212225220118
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print('Name:Varoodhini M')
print('Reg No:212225220118')
print("Mean Squared Error:",mse)
print("R-squared Score:",r2)
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()


```

## Output:
<img width="405" height="101" alt="image" src="https://github.com/user-attachments/assets/52c53654-d3fe-4b26-a64b-535ba349ce68" />

<img width="727" height="579" alt="image" src="https://github.com/user-attachments/assets/0aa7b693-f245-4269-9a5c-228ce49ba298" />




## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
