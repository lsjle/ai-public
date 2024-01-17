# /*
#  * ----------------------------------------------------------------------------
#  * "THE TEA-WARE LICENSE" (Revision 42-1):
#  * <210114@gs.hs.ntnu.edu.tw> wrote this file.  As long as you retain this notice you
#  * can do whatever you want with this stuff. If we meet some day, and you think
#  * this stuff is worth it, you can buy me a tea (tea or beer both fine) in return.   Lapsang Souchong
#  * Please notice that some part of the code might not be written by me.
#  * File: 120702.py
#  * ----------------------------------------------------------------------------
#  */
#

#testing cpcp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "https://raw.githubusercontent.com/ryanchung403/dataset/main/loan_prediction_training_data.csv"
)

df.describe()
df.info()

df["ApplicantIncome"].hist(bins=50)

df.boxplot(column="ApplicantIncome", by="Education")

#df.boxplot(column="Loan_Status", by="Education")

df.boxplot(column="LoanAmount", by="Self_Employed")

df.boxplot(column="ApplicantIncome", by="Self_Employed")

temp1 = df["Credit_History"].value_counts(ascending=True)

temp2 = df.pivot_table(
    values="Loan_Status",
    index=["Credit_History"],
    aggfunc=lambda x: x.map({"Y": 1, "N": 0}).mean(),
)

temp1.plot(kind="bar")
temp2.plot(kind="bar")

temp3 = df.pivot_table(
    values="Loan_Status",
    index=["Property_Area"],
    aggfunc=lambda x: x.map({"Y": 1, "N": 0}).mean(),
)
temp3


temp4 = df.pivot_table(
    values="Loan_Status",
    index=["Self_Employed"],
    aggfunc=lambda x: x.map({"Y": 1, "N": 0}).mean(),
)
temp4

temp5 = pd.crosstab(df["Credit_History"], df["Loan_Status"])
temp5.plot(kind="bar", stacked=True)

temp6 = pd.crosstab([df["Credit_History"], df["Gender"]], df["Loan_Status"])
temp6.plot(kind="bar", stacked=True)

pd.crosstab([df["Gender"], df["Credit_History"]], df["Loan_Status"])

df.isnull().sum().sort_values(ascending=False)

# Deal with missing values
df["Self_Employed"].value_counts()
df["Self_Employed"].fillna("No", inplace=True)

table = df.pivot_table(
    values="LoanAmount", index="Self_Employed", 
    columns="Education", aggfunc=np.median
)

table.loc["Yes", "Graduate"]
table.loc["No", "Not Graduate"]


def fage(x):
    return table.loc[x['Self_Employed'], x['Education']]

df['LoanAmount'].fillna(df.apply(fage, axis=1), inplace=True)