# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 121405.py
#  * ----------------------------------------------------------------------------
#  */
#

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,precision_score,f1_score

print('in')
df=pd.read_csv("dataset/kaggle/credit-default-prediction-ai-big-data/train.csv")
df.head()
df.describe()
df['Annual Income'].isna().value_counts()
#fix annual income problem
#with normal distribution
index = df[df['Annual Income'].isna()].index
value = np.random.normal(loc=df['Annual Income'].mean(), scale=df['Annual Income'].std(), size=df['Annual Income'].isna().sum())

df['Annual Income'].fillna(pd.Series(value, index=index), inplace=True)
#note that somehow the income will become negative at some where
df['Months since last delinquent'].isna().value_counts()
#more missing value than true value drop
df.drop(['Months since last delinquent'],axis=1,inplace=True)
#easy just fill with no
df['Bankruptcies'].fillna(0,inplace=True)
#fix Credit Score with normal distribution
index = df[df['Credit Score'].isna()].index
value = np.random.normal(loc=df['Credit Score'].mean(), scale=df['Credit Score'].std(), size=df['Credit Score'].isna().sum())

df['Credit Score'].fillna(pd.Series(value, index=index), inplace=True)
#fix long/short term problem
df['Term'].replace(['Short Term','Long Term'],[0,1])
df['Years in current job'].replace(['\syears','\syear','<\s','\+','NaN'],['','','','',''],regex=True,inplace=True)
df['Years in current job']=df['Years in current job'].apply(pd.to_numeric)
df['Years in current job'].fillna(10,inplace=True)
#done with missing value fixing

#now start generator dataset
X=df.drop(['Id','Credit Default','Term','Purpose','Home Ownership'],axis=1)
#currently also drop object data
y=df['Credit Default']
tx,sx,ty,sy=train_test_split(X,y,test_size=0.3,random_state=43)
model=RandomForestClassifier(n_estimators=100)
model.fit(tx,ty)
pred=model.predict(sx)
print("done")
print(f"f1score: {precision_score(sy,pred)}")

#end of module training start testing below

tdf=pd.read_csv("dataset/kaggle/credit-default-prediction-ai-big-data/test.csv")
tdf.describe()
tdf['Bankruptcies'].fillna(0,inplace=True)
tdf.drop(['Months since last delinquent'],axis=1,inplace=True)
index = tdf[tdf['Annual Income'].isna()].index
value = np.random.normal(loc=tdf['Annual Income'].mean(), scale=tdf['Annual Income'].std(), size=tdf['Annual Income'].isna().sum())

tdf['Annual Income'].fillna(pd.Series(value, index=index), inplace=True)
index = tdf[tdf['Credit Score'].isna()].index
value = np.random.normal(loc=tdf['Credit Score'].mean(), scale=tdf['Credit Score'].std(), size=tdf['Credit Score'].isna().sum())

tdf['Credit Score'].fillna(pd.Series(value, index=index), inplace=True)
tdf['Term'].replace(['Short Term','Long Term'],[0,1])
tdf['Years in current job'].replace(['\syears','\syear','<\s','\+','NaN'],['','','','',''],regex=True,inplace=True)
tdf['Years in current job']=tdf['Years in current job'].apply(pd.to_numeric)
tdf['Years in current job'].fillna(10,inplace=True)
#finish fixing missing value
#start pred
testdata=tdf.drop(['Id','Term','Purpose','Home Ownership'],axis=1)
pred=model.predict(testdata)
sumbitdf=pd.DataFrame(columns=['Id','Credit Default'])
sumbitdf['Id']=range(7500,10000)
sumbitdf['Credit Default']=pred
sumbitdf.to_csv('for_submission_2023121402.csv', index=False)