# ai-public
This repositry contain my work during 110 first semester. 
This naming rule is monthyear#.py
Different file might have to run in different environment, some might conflict due to unknown reason and others might have problem, all code should be treated as untested!
This is not the original repository, the original contain copyright-restricted content. It's illgeal to contain them, so missing file are expected.
# 11/16
```python
import seaborn as sns
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
```
this code demostrate the package we need to import for our data analysis
```python
df=pd.read_csv("dataset/train_data_titanic.csv")
```
this code show us how to import a dataset as Dataframe in pandas
# 11/23
```python
df.drop(['Name','Ticket'],axis=1,inplace=
        True)
```
this code will drop a column which we dont want, usually, contain text type data or too many missing value. 
```python
betterdf= pd.DataFrame(KNNImputer().fit_transform(kdf))
```
this code show us to filled missing value using KNN.
```python
model = LogisticRegression(max_iter=1000)
```
this code is setting the model to logistic regression and the maxium iteration value is 1000. If we set max_iter too little, it's performance might weaken, if we set it too high, it might be a waste of time. 
```python
accuracy = accuracy_score(y_test, y_pred)
```
This is the function which we evaluate our model's performance
# 11/30
```python
k=10000
lines=np.random.rand(k)*100
grades=np.zeros(k)
for i in range(k):
    grades[i]=math.floor(lines[i]/2+2*np.random.normal())
df=pd.DataFrame(data={'line':lines,'grade':grades})
```
This code is generating a set of data that are simllar to what has happened for our school english eassay writing exam.
```python
plt.scatter(sx, sy, color='black',marker='.')
plt.plot(sx, syy, color='blue', linewidth=3)
plt.xlabel('Line')
plt.ylabel('Grade')
plt.title('Best Fit Line in Linear Regression')
plt.show()
```
This is the code we use to show the figure and evaluate using our own vision.
```python
sumbitdf=pd.DataFrame(columns=['PassengerId','Survived'])
sumbitdf['PassengerId']=range(892,1310)
sumbitdf['Survived']=pred
for i in range(0,418):
  sumbitdf['Survived'][i]=round(sumbitdf['Survived'][i])
sumbitdf.to_csv('for_submission_2023113004.csv', index=False)
```
This is the code we use to submit our data to kaggle.
# 12/07
```python
table = df.pivot_table(
    values="LoanAmount", index="Self_Employed", 
    columns="Education", aggfunc=np.median
)
```
This code will generate a table to account different types of value.
```python
joblib.dump(modelk, "2023121401.pkl")
```
This code is used to output our trained model.
```python
modelk, r,a,f1=modelfun(RandomForestClassifier(n_estimators=20),df,var_mod,'Loan_Status')
```
This code will call the function we wrote and use random forest classifier to classify the data we give which is df.
# 12/14
```python
import joblib

# Load the model
model = joblib.load("2023121401.pkl")

# Collect input from the user
gender = input("Enter gender (0 for Male, 1 for Female): ")
married = input("Are you married? (0 for No, 1 for Yes): ")
dependents = input("Number of dependents: ")
education = input("Education level (0 for Not Graduate, 1 for Graduate): ")
self_employed = input("Are you self-employed? (0 for No, 1 for Yes): ")
property_area = input("Enter property area (0 for Urban, 1 for Semiurban, 2 for Rural): ")

# Convert input to numerical format
input_data = [int(gender), int(married), int(dependents), int(education), int(self_employed), int(property_area)]

# Make predictions
pred = model.predict([input_data])
print(pred)
```
This is the cli interface for our model.(CLI, command-line interface)
```python
@app.route("/mainpage")
def mainpage():
    return render_template("121403.html")
```
This code is the main web interface code.
```python
@app.route("/proceed", methods=['GET', 'POST'])
def proceed():
    if(request.method=='GET'):
        return 'f'
    else:
        model=joblib.load("2023121401.pkl")
        inputdata=[int(request.values['gender']),int(request.values['married']),int(request.values['dependents']),int(request.values['education']),int(request.values['selfemp']),int(request.values['pa'])]
        return (str(model.predict([inputdata]))+str(model.predict_proba([inputdata])))
```
This is the backend of our web interface. 
RefL 121403.html, this is the awesome html code i wrote.
```python
import tensorflow as tf
print(tf.__file__)
```
This will print the location of tensorflow we used. It's useful for debugging env problem.
```python
from keras.datasets import mnist
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
```
This is how we import our pre-processed data.
# 12/21
```python
df=pd.read_csv("dataset/kaggle/train.csv")
xtrain,xtest,ytrain,ytest=train_test_split(df['filepaths'],df['Font'],test_size=0.3,random_state=83)
xtrainimg =[]
for filepath in xtrain:
    # Open the image file
    img = image_utils.load_img(filepath,color_mode="grayscale",target_size=(128,128))
    # Convert the image to a NumPy array
    img_array = np.array(img)
    # Append the array to the list
    xtrainimg.append(img_array)
```
This is the code we used to convert the image to the type where our model can processed
```python
def replace_letters_with_numbers(input_list):
    result_list = []

    for original_string in input_list:
        modified_string = ""
        for char in original_string:
            if 'a' <= char <= 'z':
                modified_string += str(ord(char) - ord('a') + 11)
            elif 'A' <= char <= 'Z':
                modified_string += str(ord(char) - ord('A') + 11)
            else:
                modified_string += char

        result_list.append(modified_string)

    return result_list
```
This code can categorize the letter we will be used in the future.