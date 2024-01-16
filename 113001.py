#this is just a test file which generate the problem of hsnu english test
import pandas as pd
import random
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
k=10000
lines=np.random.rand(k)*100
grades=np.zeros(k)
for i in range(k):
    grades[i]=math.floor(lines[i]/2+2*np.random.normal())
df=pd.DataFrame(data={'line':lines,'grade':grades})
df.corr()
x=df['line'].values.reshape(-1, 1)
y=df['grade']
tx,sx,ty,sy=train_test_split(x,y,test_size=0.3,random_state=87)
model=LinearRegression()
model.fit(tx,ty)
syy=model.predict(sx)
print(r2_score(sy,syy))
# Plot the best fit line
plt.scatter(sx, sy, color='black',marker='.')
plt.plot(sx, syy, color='blue', linewidth=3)
plt.xlabel('Line')
plt.ylabel('Grade')
plt.title('Best Fit Line in Linear Regression')
plt.show()