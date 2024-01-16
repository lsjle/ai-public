from sklearn import datasets
x,y=datasets.load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
maax=0
maxc=0
tx,ax,ty,ay=train_test_split(x,y,test_size=0.2,random_state=0)
reg=Ridge(alpha=.5)
reg.fit(tx,ty)
py=reg.predict(ax)
# for i in range(1000):
#     tx,ax,ty,ay=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=i)
#     reg=LinearRegression()
#     reg.fit(tx,ty)
#     py=reg.predict(ax)
#     if(r2_score(ay,py)>maax):
#         maxc=i
#         maax=r2_score(ay,py)
print(r2_score(ay,py))