# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values
## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Srihariharan S A
RegisterNumber:  212221040160
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
Placement Data:
![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/fe519b85-936b-4cc8-92d1-eb9615b59c4d)

Salary Data: 

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/222cd724-e4c9-4af7-adfb-992cbcc9d735)


Checking the null() function:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/9f59838e-ed4c-49c3-8204-bf2ea11b7a83)


Data Duplicate:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/07abe305-8e03-4c3d-ad4f-7d9804c47a27)



Print Data: 

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/a4b8ca14-1ca7-4767-9a22-39981099f5cd)

Data-status:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/32233e9f-3c9e-456d-9260-a91b18a77a34)



y_prediction array:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/a66e4595-af8d-45cd-b0a1-1e27d8c6a4e2)



Accuracy value:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/aa4bbd52-7cab-4fc1-9050-b02eb9bd3434)



Confusion array:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/47b6d69b-8c17-4994-b806-e02bbcfea572)


Classification report:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/c601ec40-ac22-4769-a425-48ef3b3f4fa5)


Prediction of LR:

![image](https://github.com/Kishore2o/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679883/d70a332d-7b75-4e3d-9ea0-aeb8c7a05d62)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
