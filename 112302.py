# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 112302.py
#  * ----------------------------------------------------------------------------
#  */
#

#error!
#score 0
import joblib
import pandas as pd
from sklearn.impute import KNNImputer
pretrain=joblib.load('2023113001.pkl')
df=pd.read_csv('dataset/kaggle/titanic/test.csv')
print(df['Age'].isnull().value_counts())
df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
adf=df['Embarked']
df['Sex']=df['Sex'].replace(['male','female'],[0,1])
knn=KNNImputer()
betterdf=pd.DataFrame(knn.fit_transform(df.drop(['Embarked'],axis=1)))
adf=df['Embarked']
ldf=pd.get_dummies(data=adf,dtype=int,columns=['Embarked'])
ldf[7]=ldf['C']
ldf[8]=ldf['Q']
ldf[9]=ldf['S']
ldf.drop(['C','Q','S'],inplace=True,axis=1)

betterdf=pd.concat([betterdf,ldf],axis=1)
print(betterdf.head())
pred=pretrain.predict(betterdf.drop([0],axis=1))
sumbitdf=pd.DataFrame(columns=['PassengerId','Survived'])
sumbitdf['PassengerId']=range(892,1310)
sumbitdf['Survived']=pred
sumbitdf.to_csv('for_submission_20231130.csv', index=False)