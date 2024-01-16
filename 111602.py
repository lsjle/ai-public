from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import numpy as np
df=pd.read_csv("dataset/train_data_titanic.csv")
print(df.info())
sns.pairplot(df)
x=df[:2]+df[:,5]+df[:,6]+df[:,7]+df[:,9]
#error handle here
y=df['Survived']
train_x,train_y,test_x,test_y=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=0)
reg=LinearRegression()
reg.fit(train_x,train_y)
y_pred=reg.predict(test_x)
print(r2_score(test_y,y_pred))