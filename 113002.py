import pandas as pd
import numpy as np
sumbitdf=pd.DataFrame(columns=['PassengerId','Survived'])
sumbitdf['PassengerId']=range(892,1310)
sumbitdf['Survived']=np.random.rand(418)
for i in range(418):
    sumbitdf['Survived'][i]=1#round(sumbitdf['Survived'][i])
sumbitdf.to_csv('random_2023113002.csv', index=False)