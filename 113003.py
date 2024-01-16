#use KNN to fit missing value in age
#use S fit missing value in Embarked
#use RandomForestClassifier(n_estimators=100, random_state=42 to fit the model
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
df = pd.read_csv('dataset/kaggle/titanic/train.csv')
df['Embarked'].value_counts()
#sns.pairplot(df)
df.drop(['Name','Ticket'],axis=1,inplace=
        True)
#df.groupby('Survived').mean(numeric_only=True)
#temp drop 

df['Sex']=df['Sex'].replace(['male', 'female'],[0, 1])
df.drop(['Cabin'],axis=1,inplace=True)

#FIX Cabin with with random value
kdf=df.drop(['Embarked'],axis=1)

betterdf= pd.DataFrame(KNNImputer().fit_transform(kdf))
df['Embarked'] = df['Embarked'].fillna('S')
adf=df['Embarked']
ldf=pd.get_dummies(data=adf, dtype=int, columns=['Embarked'])
print(ldf.head())
ldf[8]=ldf['C']
ldf[9]=ldf['Q']
ldf[10]=ldf['S']
ldf.drop(['C','Q','S'],inplace=True,axis=1)
betterdf=pd.concat([betterdf,ldf],axis=1)
#using KNN to handle missing value
#above should be trained again

#lgtm
x=betterdf.drop([0,1],axis=1)
y=betterdf[1]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=64)
# Create a logistic regression model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
#as far as good
joblib.dump(model,'2023113001.pkl',compress=3)
#new as append to merge two together 20231130
gdf=pd.read_csv('dataset/kaggle/titanic/test.csv')
print(df['Age'].isnull().value_counts())
gdf.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
hdf=gdf['Embarked']
gdf['Sex']=gdf['Sex'].replace(['male','female'],[0,1])
knn=KNNImputer()
bettergdf=pd.DataFrame(knn.fit_transform(gdf.drop(['Embarked'],axis=1)))
gldf=pd.get_dummies(data=hdf,dtype=int,columns=['Embarked'])
gldf[7]=gldf['C']
gldf[8]=gldf['Q']
gldf[9]=gldf['S']
gldf.drop(['C','Q','S'],inplace=True,axis=1)
bettergggdf=pd.concat([bettergdf,gldf],axis=1)
pred=model.predict(bettergggdf.drop([0],axis=1))
sumbitdf=pd.DataFrame(columns=['PassengerId','Survived'])
sumbitdf['PassengerId']=range(892,1310)
sumbitdf['Survived']=pred
for i in range(0,418):
  sumbitdf['Survived'][i]=round(sumbitdf['Survived'][i])
sumbitdf.to_csv('for_submission_2023113004.csv', index=False)
from sklearn.metrics import accuracy_score
rdf=pd.read_csv("gender_submission.csv")
print(accuracy_score(rdf['Survived'],sumbitdf['Survived']))