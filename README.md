# EXP-05: Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LOKNAATH P
RegisterNumber:  212223240080
```

```python
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

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

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
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
### Placement Data:

![data](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/cba641d7-4b64-474a-9df3-f8047b4ddc21)

### Salary Data:

![salarydata](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/b69592e3-fb46-446d-87a4-60e8dabf45a1)

### Checking the null() function:


![null](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/196a08f0-0571-40f2-bfdf-b6e1d2b4fa8f)

### Data Duplicate:


![duplicate](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/3efb2a8c-6c60-4466-99b2-2c3c7b7a39b4)

### Print Data:

![dataprint](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/37d05f23-2187-49d2-a871-7dbf5d7baca9)


### Data-Status:

![datastatus](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/d0b24ebb-4d7a-4956-b6e5-b87f65ccbeeb)

### Y_prediction array:
![ypredict](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/81a5cd80-1fa0-48d8-a838-567b6e7a6676)

### Accuracy value:
![accuracy](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/1ca21819-8baa-4312-aae8-1b094fe75ea6)

### Confusion array:

![confusion](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/675efabe-006d-463a-b5f0-0cc4354ca37a)

### Classification Report:


![cr](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/be3ab929-d71c-492a-8adc-9a054cf08983)


### Prediction of LR:
![lrpredict](https://github.com/charumathiramesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120204455/295b82c5-385c-4832-9d92-282a651946cb)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
