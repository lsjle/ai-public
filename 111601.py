# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 111601.py
#  * ----------------------------------------------------------------------------
#  */
#

import seaborn as sns
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
#done with import
df=pd.read_csv("dataset/Housing_Dataset_Sample.csv")
print(df.info())
#start working 
df.head()
df.describe().T
#somehow it do print
print("clear")
sns.displot(df['Price'])
sns.pairplot(df)
#print("clear")
#plt.plot(df['Price'])
#why it's shows like normal distribution
# rand=np.zeros(10000)
# for i in range(10000):
#     rand[i]=np.random.normal()
# plt.plot(rand)
X=df.iloc[:,:5]
#only ":" means all
# 0-4 5,price:y 6,address useless


Y=df['Price']
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
# 訓練模型
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

y_pred = linearModel.predict(X_test)
# 21.894831181729202
print('MSE:', mean_squared_error(y_test, y_pred)/pow(10,10))
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
