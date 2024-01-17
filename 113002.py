# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 113002.py
#  * ----------------------------------------------------------------------------
#  */
#

import pandas as pd
import numpy as np
sumbitdf=pd.DataFrame(columns=['PassengerId','Survived'])
sumbitdf['PassengerId']=range(892,1310)
sumbitdf['Survived']=np.random.rand(418)
for i in range(418):
    sumbitdf['Survived'][i]=1#round(sumbitdf['Survived'][i])
sumbitdf.to_csv('random_2023113002.csv', index=False)