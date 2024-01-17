# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 120701.py
#  * ----------------------------------------------------------------------------
#  */
#

import pandas as pd
import seaborn as sns
import numpy as np
import joblib
from sklearn.metrics import recall_score,accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("dataset/loan_prediction_training_data.csv")
df.head()
df["Self_Employed"].fillna("No", inplace=True)
#consider change this or not
table = df.pivot_table(
    values="LoanAmount", index="Self_Employed", 
    columns="Education", aggfunc=np.median
)
table.loc["Yes", "Graduate"]
table.loc["No", "Not Graduate"]
##something wrong here
def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]

df['LoanAmount'].fillna(df.apply(fage, axis=1), inplace=True)
#df.dropna(subset="LoanAmount",inplace=True)
#seems better to just not keep the data
#since this can be vary and is highly depends on it
#and for other missing value?
#Loan_Amount_Term use random in normal distribution between -1~1 std
df.describe()
ka = df[df.Loan_Amount_Term.isnull()].index
value = np.random.normal(loc=df.Loan_Amount_Term.mean(), scale=df.Loan_Amount_Term.std(), size=df.Loan_Amount_Term.isnull().sum())
df.Loan_Amount_Term.fillna(pd.Series(value,index=ka), inplace=True)
df.Gender.fillna(1)
df['Dependents'].replace(['3+'],[3],inplace=True)
ck=df[df['Dependents'].isnull()].index
#15 below should be =int(df['Dependents'].isnull().value_counts())
value1=np.random.choice([0,1,2,3],15,p=(0.575962,0.170283,0.168614,0.085141))
df['Dependents'].fillna(pd.Series(value1,index=ck),inplace=True)
df['Dependents'].replace(['0','1','2','3'],[0,1,2,3],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Gender'].fillna(df['Gender'].mode([0]))
df['Gender']=df['Gender'].replace(['Male', 'Female'],[1, 0])

df['Married'].replace(['Yes','No'],[1,0],inplace=True)
df['Education'].replace(['Graduate','Not Graduate'],[1,0],inplace=True)
df['Self_Employed'].replace(['Yes','No'],[1,0],inplace=True)
df['Loan_Status'].replace(['Y','N'],[1,0],inplace=True)
#not sure the following fix is true or not
#finished
#encode label
var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le=LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i])
# temp=pd.crosstab(df["Credit_History"],df["Loan_Status"])
# temp.plot(kind='bar',stacked=True)
#wrong cuz it always use 1.0
def modelfun(model,data,predc,outcome,tsize=0.3,randomnum=65):
    x=data[predc]
    y=data[outcome]
    xt,xs,yt,ys=train_test_split(x,y,test_size=tsize,random_state=randomnum)
    model.fit(xt,yt)
    pred=model.predict(xs)
    return [model,recall_score(ys,pred),accuracy_score(ys,pred),f1_score(ys,pred)]
modelk, r,a,f1=modelfun(RandomForestClassifier(n_estimators=20),df,var_mod,'Loan_Status')
print(r," ",a," ",f1)
joblib.dump(modelk, "2023121401.pkl")