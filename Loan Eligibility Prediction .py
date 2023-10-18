#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv("train.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[6]:


dataset.info()


# In[7]:


dataset.describe()


# In[8]:


pd.crosstab(dataset["Credit_History"],dataset["Loan_Status"],margins=True)


# In[9]:


dataset.boxplot(column="ApplicantIncome")


# In[10]:


dataset["ApplicantIncome"].hist(bins=20)


# In[11]:


dataset["CoapplicantIncome"].hist(bins=20)


# In[12]:


dataset.boxplot(column="ApplicantIncome",by="Education")


# In[13]:


dataset.boxplot(column="LoanAmount")


# In[14]:


dataset["LoanAmount"].hist(bins=20)


# In[15]:


dataset["LoanAmount_log"]=np.log(dataset["LoanAmount"])
dataset["LoanAmount_log"].hist(bins=20)


# In[16]:


dataset.isnull().sum()


# In[26]:


dataset["Gender"].fillna(dataset["Gender"].mode()[0],inplace=True)


# In[27]:


dataset["Married"].fillna(dataset["Married"].mode()[0],inplace=True)


# In[28]:


dataset["Dependents"].fillna(dataset["Dependents"].mode()[0],inplace=True)


# In[29]:


dataset["Self_Employed"].fillna(dataset["Self_Employed"].mode()[0],inplace=True)


# In[22]:


dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[30]:


dataset["Loan_Amount_Term"].fillna(dataset["Loan_Amount_Term"].mode()[0],inplace=True)


# In[31]:


dataset["Credit_History"].fillna(dataset["Credit_History"].mode()[0],inplace=True)


# In[32]:


dataset.isnull().sum()


# In[35]:


dataset["TotalIncome"]=dataset["ApplicantIncome"] + dataset["CoapplicantIncome"]
dataset["TotalIncome_log"]=np.log(dataset["TotalIncome"])


# In[37]:


dataset["TotalIncome_log"].hist(bins=20)


# In[38]:


dataset.head()


# In[40]:


x= dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y= dataset.iloc[:,12].values


# In[41]:


x


# In[42]:


y


# In[44]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[45]:


print(x_train)


# In[47]:


from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()


# In[56]:


for i in range(0,5):
    x_train[:,i] = labelencoder_x.fit_transform(x_train[:,i])


# In[50]:


x_train[:,7]= labelencoder_x.fit_transform(x_train[:,7])


# In[51]:


x_train


# In[52]:


labelencoder_y= LabelEncoder()
y_train= labelencoder_y.fit_transform(y_train)


# In[53]:


y_train


# In[57]:


for i in range(0,5):
    x_test[:,i] = labelencoder_x.fit_transform(x_test[:,i])


# In[58]:


x_test[:,7]= labelencoder_x.fit_transform(x_test[:,7])


# In[59]:


labelencoder_y= LabelEncoder()
y_test= labelencoder_y.fit_transform(y_test)


# In[60]:


x_test


# In[61]:


y_test


# In[64]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[65]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier= DecisionTreeClassifier(criterion="entropy",random_state=0)
DTClassifier.fit(x_train,y_train)


# In[66]:


y_pred= DTClassifier.predict(x_test)
y_pred


# In[69]:


from sklearn import metrics
print("The accuracy of decision tree is: ",metrics.accuracy_score(y_pred,y_test))


# In[70]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(x_train,y_train)


# In[71]:


y_pred= NBClassifier.predict(x_test)


# In[72]:


y_pred


# In[73]:


print("The accuracy of Naive Bayes is : ",metrics.accuracy_score(y_pred,y_test))


# In[77]:


import os
os.listdir()


# In[82]:


testdata= pd.read_csv("C:\\Users\\Ankit\\Downloads\\Testing.csv")


# In[83]:


testdata.head()


# In[84]:


testdata.info()


# In[85]:


testdata.isnull().sum()


# In[90]:


testdata["Gender"].fillna(testdata["Gender"].mode()[0],inplace=True)
testdata["Dependents"].fillna(testdata["Dependents"].mode()[0],inplace=True)
testdata["Self_Employed"].fillna(testdata["Self_Employed"].mode()[0],inplace=True)
testdata["Loan_Amount_Term"].fillna(testdata["Loan_Amount_Term"].mode()[0],inplace=True)
testdata["Credit_History"].fillna(testdata["Credit_History"].mode()[0],inplace=True)


# In[92]:


testdata.isnull().sum()


# In[93]:


testdata.boxplot(column="LoanAmount")


# In[94]:


testdata.boxplot(column="ApplicantIncome")


# In[95]:


testdata.LoanAmount= testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[96]:


testdata["LoanAmount_log"]=np.log(testdata["LoanAmount"])


# In[97]:


testdata.isnull().sum()


# In[98]:


testdata["TotalIncome"]= testdata["ApplicantIncome"]+testdata["CoapplicantIncome"]
testdata["TotalIncome_log"]= np.log(testdata["TotalIncome"])


# In[99]:


testdata.head()


# In[100]:


test= testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[101]:


for i in range(0,5):
    test[:,i]=labelencoder_x.fit_transform(test[:,i])


# In[102]:


test[:,7]= labelencoder_x.fit_transform(test[:,7])


# In[103]:


test


# In[104]:


test= ss.fit_transform(test)


# In[105]:


pred= NBClassifier.predict(test)


# In[106]:


pred

