# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop


## Program & Output:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
## Developed by: VAMSI KRISHNA G
## RegisterNumber:  212223220120

```
import pandas as pd
data=pd.read_csv('Placement_data.csv')
data.head(5)
```
![image](https://github.com/user-attachments/assets/1c2d7421-1e8c-4115-b97d-8722818fad8a)



```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```

![image](https://github.com/user-attachments/assets/fa9cdd8e-5889-4261-b3f3-2787b5576fe9)



```
data1.isnull().sum()
```

![image](https://github.com/user-attachments/assets/18d28d27-c0e7-4777-8c2d-4a2591745d9b)



```
data1.duplicated().sum()
```

![image](https://github.com/user-attachments/assets/f2944f0e-c736-4536-8ee4-979f727adb6a)



```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
![image](https://github.com/user-attachments/assets/560004ff-9c30-4e05-967b-d4dec9d03a4f)


```
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/617edbc2-1dca-489c-b2e4-d4c609a132bf)


```
y=data1["status"]
y
```

![image](https://github.com/user-attachments/assets/28b056e0-ac92-4add-8490-13b4cd8cecf6)


```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/c40dbd01-48ab-4d1f-9cb7-f12de1db80c3)


```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy=",accuracy)
```
![image](https://github.com/user-attachments/assets/1150de4d-85f6-48be-abe5-7dbdbd70dd71)



```
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix
```
![image](https://github.com/user-attachments/assets/33e2f7f3-4d0e-4cd2-951b-f6c2f12c0ddc)



```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![ex5_op11](https://github.com/user-attachments/assets/e4a467d2-0551-46ff-8d14-2d9ab50b5c3c)


```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/e83f0c83-9dca-48eb-afea-0071d17dbed8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
