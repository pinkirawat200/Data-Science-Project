#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[2]:


df= pd.read_csv("loan.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df["LoanAmount_log"]=np.log(df["LoanAmount"])
df["LoanAmount_log"].hist(bins=20)


# In[7]:


df.isnull().sum()


# In[8]:


df["TotalIncome"]=df["ApplicantIncome"]+df["CoapplicantIncome"]
df["TotalIncome_log"]=np.log(df["TotalIncome"])
df["TotalIncome_log"].hist(bins=20)


# In[12]:


df["Gender"].fillna(df["Gender"].mode()[0],inplace=True)
df["Married"].fillna(df["Married"].mode()[0],inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0],inplace=True)

df.LoanAmount= df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log= df.LoanAmount_log.fillna(df.LoanAmount_log.mean())

df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


x= df.iloc[:,np.r_[1:5,9:11,13:15]].values
y= df.iloc[:,12].values


# In[15]:


x


# In[16]:


y


# In[17]:


print("per of missing gender is %2f%%" %((df["Gender"].isnull().sum()/df.shape[0])*100))


# In[19]:


print("Number of people who take loan as group by gender:  ")
print(df["Gender"].value_counts())
sns.countplot(x="Gender",data=df,palette="Set1")


# In[20]:


print("Number of people who take loan as group by marital status:  ")
print(df["Married"].value_counts())
sns.countplot(x="Married",data=df,palette="Set1")


# In[21]:


print("Number of people who take loan as group by Dependents:  ")
print(df["Dependents"].value_counts())
sns.countplot(x="Dependents",data=df,palette="Set1")


# In[22]:


print("Number of people who take loan as group by self employed:  ")
print(df["Self_Employed"].value_counts())
sns.countplot(x="Self_Employed",data=df,palette="Set1")


# In[30]:


print("Number of people who take loan as group by LoanAmount:  ")
print(df["LoanAmount"].value_counts())
sns.countplot(x="LoanAmount",data=df,palette="Set1")


# In[31]:


print("Number of people who take loan as group by Credit History:  ")
print(df["Credit_History"].value_counts())
sns.countplot(x="Credit_History",data=df,palette="Set1")


# In[37]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()


# In[40]:


for i in range(0,5):
    X_train[:,i] = LabelEncoder_x.fit_transform(X_train[:,i])
    X_train[:,7] = LabelEncoder_x.fit_transform(X_train[:,7])
    
X_train


# In[41]:


LabelEncoder_y = LabelEncoder()
y_train = LabelEncoder_y.fit_transform(y_train)
y_train


# In[42]:


for i in range(0,5):
    X_test[:,i] = LabelEncoder_x.fit_transform(X_test[:,i])
    X_test[:,7] = LabelEncoder_x.fit_transform(X_test[:,7])

X_test


# In[43]:


LabelEncoder_y = LabelEncoder()
y_test = LabelEncoder_y.fit_transform(y_test)

y_test


# In[44]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
x_test =  ss.fit_transform(X_test)


# In[45]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)


# In[47]:


from sklearn import metrics
y_pred = rf_clf.predict(x_test)

print("accuracy of random forest clf is",metrics.accuracy_score(y_pred,y_test))

y_pred


# In[55]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,y_train)


# In[56]:


y_pred = nb_clf.predict(X_test)
print("accuracy of gaussianNB is %.",metrics.accuracy_score(y_pred,y_test))


# In[50]:


y_pred


# In[51]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)


# In[52]:


y_pred = dt_clf.predict(X_test)
print("accuracy of DT is",metrics.accuracy_score(y_pred,y_test))


# In[53]:


y_pred


# In[ ]:




