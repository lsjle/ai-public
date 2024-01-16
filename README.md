# ai-public
This repositry contain my work during 110 first semester. 
This naming rule is monthyear#.py
Different file might have to run in different environment, some might conflict due to unknown reason and others might have problem, all code should be treated as untested!
This is not the original repository, the original contain copyright-restricted content. It's illgeal to contain them, so missing file are expected.
# 11/16
```python
import seaborn as sns
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
```
this code demostrate the package we need to import for our data analysis
```python
df=pd.read_csv("dataset/train_data_titanic.csv")
```
this code show us how to import a dataset as Dataframe in pandas
# 11/23
```python
df.drop(['Name','Ticket'],axis=1,inplace=
        True)
```
this code will drop a column which we dont want, usually, contain text type data or too many missing value. 
```python
betterdf= pd.DataFrame(KNNImputer().fit_transform(kdf))
```
this code show us to filled missing value using KNN.
```python
model = LogisticRegression(max_iter=1000)
```
this code is setting the model to logistic regression and the maxium iteration value is 1000. If we set max_iter too little, it's performance might weaken, if we set it too high, it might be a waste of time. 
```python
accuracy = accuracy_score(y_test, y_pred)
```
This is the function which we evaluate our model's performance
# 11/30
```python
k=10000
lines=np.random.rand(k)*100
grades=np.zeros(k)
for i in range(k):
    grades[i]=math.floor(lines[i]/2+2*np.random.normal())
df=pd.DataFrame(data={'line':lines,'grade':grades})
```
This code is generating a set of data that are simllar to what has happened for our school english eassay writing exam.
```python
plt.scatter(sx, sy, color='black',marker='.')
plt.plot(sx, syy, color='blue', linewidth=3)
plt.xlabel('Line')
plt.ylabel('Grade')
plt.title('Best Fit Line in Linear Regression')
plt.show()
```
This is the code we use to show the figure and evaluate using our own vision.
```python
sumbitdf=pd.DataFrame(columns=['PassengerId','Survived'])
sumbitdf['PassengerId']=range(892,1310)
sumbitdf['Survived']=pred
for i in range(0,418):
  sumbitdf['Survived'][i]=round(sumbitdf['Survived'][i])
sumbitdf.to_csv('for_submission_2023113004.csv', index=False)
```
This is the code we use to submit our data to kaggle.
# 12/07
```python
table = df.pivot_table(
    values="LoanAmount", index="Self_Employed", 
    columns="Education", aggfunc=np.median
)
```
This code will generate a table to account different types of value.