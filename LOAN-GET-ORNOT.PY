#----------------------------------about data set
#any bank before loan approvel it will check many thinks realted to your bank accound and other parameters like your job salary etc.
#Among all industries, insurance domain has the largest use of analytics & data science methods. 
# This data set would provide you enough taste of working on data sets from insurance companies,what challenges are faced, what strategies are used, which variables influence the outcome etc.
#  This is a classification problem. The data has 615 rows and 13 columns.
#Problem-----
#Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. 
# These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. 
# To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. 
# Here they have provided a partial data set.
#-------------------------------------------work flow
#work flow
#data collection
#data pre-processing
#train-test-split data
#support vector machine model (classification model ) 
#we can use logistic model also 
#-----------------------------------import your necssary labrary
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler(no need to use it because alredy data is specific)
#----------------------------------------data analysis
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/train_u6lujuX_CVtuZ9i (1).csv")
print(data.shape)
print(data.columns)
print(data.info())
print(data.isnull().sum())
print(data.head(5))
print("-----------------")
data_new = data.dropna()
print(data_new.isnull().sum())
print(data_new.shape)
#data.replace()
data_new.replace({"Gender":{"Male":0, "Female":1}, "Married":{"Yes":0, "No":1}, "Education":{"Graduate":0, "Not Graduate":1}, "Self_Employed":{"Yes":0, "No":1}, "Property_Area":{"Urban":0, "Rural":1, "Semiurban":3}, "Loan_Status":{"Y":1, "N":0}, "Dependents":{"3+":3}}, inplace=True)
print(data_new.head(5))
#seperate 
x = data_new.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
print(x.head(5))
y = data_new["Loan_Status"]
print(y)
#-----------------------train_test_split data 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
print(x.shape, x_test.shape, x_train.shape)
print(y.shape, y_test.shape, y_train.shape)

#--------------------------use this logistic model 
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)
#-------------------------------predict our output 

y_values_tr = model.predict(x_train)
print(y_values_tr, "this is our prediction", np.array(y_train), "this is true values")
accur = accuracy_score(y_values_tr, np.array(y_train))
print(accur)

#--------------------------------------preict our test data 
y_value_test = model.predict(x_test)
print(y_value_test, "this is prediction values", np.array(y_test), "this is true values")
accur = accuracy_score(np.array(y_test), y_value_test)
print(accur, "this is our preiction accurency score of test data")

#-----------------------------------plot data graph of test data

sns.categorical(x=y_value_test, y=y_test)
plt.xlabel("THIS IS OUR PREDICTION", fontsize=(15))
plt.ylabel("THIS IS TRUE VALUE", fontsize=(15))
plt.title("COMPARISION OF TEST DATA", fontsize=(25))
plt.show()

