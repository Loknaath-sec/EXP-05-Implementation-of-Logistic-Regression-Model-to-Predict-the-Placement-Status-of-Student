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

![image](https://github.com/user-attachments/assets/d53219be-85d2-407c-a534-e54ff3844129)

### Salary Data:

![image](https://github.com/user-attachments/assets/ee997ec0-f6ce-4f95-9f6c-ce1ec4e5908e)

### Checking the null() function:

![image](https://github.com/user-attachments/assets/3c2e07c3-946a-47ba-82dd-0bb343804274)

### Data Duplicate:

![image](https://github.com/user-attachments/assets/8c0fce2c-fae6-42ae-b30e-2379db7f296e)

### Print Data:

![image](https://github.com/user-attachments/assets/2d38399c-afba-43bc-a8af-457c5c56d28a)

### Data-Status:

![image](https://github.com/user-attachments/assets/3cf4526d-f812-4576-9978-1775edef3fbc)

### Y_prediction array:

![image](https://github.com/user-attachments/assets/3e70f566-d947-4623-96d0-84e648952863)

### Accuracy value:

![image](https://github.com/user-attachments/assets/66f77467-ff63-4469-80f5-51998637b44a)

### Confusion array:

![image](https://github.com/user-attachments/assets/4c2cb564-903d-4e46-9f6f-f3de9775cdb1)

### Classification Report:

![image](https://github.com/user-attachments/assets/edc7279a-632f-495a-b267-cc43cac83c5b)


### Prediction of LR:

![image](https://github.com/user-attachments/assets/4260ac0f-1fa0-4b80-a5fb-b2e08a3c45dd)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
