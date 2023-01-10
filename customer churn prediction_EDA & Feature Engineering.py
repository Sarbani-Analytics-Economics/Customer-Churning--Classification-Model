#!/usr/bin/env python
# coding: utf-8

# ### Goal
# 
# This is a banking data, where we will be predicting if the customer will churn in the next quarter or not.(Average balance of customer falls below minimum balance in the next quarter (1/0))
# 
# 

# ###  Data Dictionary
# #### Demographic information about customers
# 
# customer_id - Customer id
# vintage - Vintage of the customer with the bank in number of days
# age - Age of customer
# gender - Gender of customer
# dependents - Number of dependents
# occupation - Occupation of the customer
# city - City of customer (anonymised)
# 
# #### Customer Bank Relationship customer_nw_category - Net worth of customer (3:Low 2:Medium 1:High)
# 
# branch_code - Branch Code for customer account
# days_since_last_transaction - No of Days Since Last Credit in Last 1 year
# 
# #### Transactional Information
# current_balance - Balance as of today
# previous_month_end_balance - End of Month Balance of previous month
# average_monthly_balance_prevQ - Average monthly balances (AMB) in Previous Quarter
# average_monthly_balance_prevQ2 - Average monthly balances (AMB) in previous to previous quarter
# current_month_credit - Total Credit Amount current month
# previous_month_credit - Total Credit Amount previous month
# current_month_debit - Total Debit Amount current month
# previous_month_debit - Total Debit Amount previous month
# current_month_balance - Average Balance of current month
# previous_month_balance - Average Balance of previous month
# churn - Average balance of customer falls below minimum balance in the next quarter (1/0)

# Now, that we understand the dataset in detail. It is time to build a classifiaction model to predict the churn.
# Steps to be follwed are:
# 
# * Load Data & Packages for model building & preprocessing
# * Missing value imputation
# * Feature Engineering
# * Exploratory Data Analysis
# * Preprocessing
# * Select features on the basis of EDA Conclusions & finalise columns for model
# * Decide Evaluation Metric on the basis of business problem
# * Build model -Logistic,Random Forest,XGBoost using Cross Validation
# * Hyperparameter Tuning using Grid Search CV
# * Use Reverse Feature Elimination to find the top features and build model using the top 10 features & compare

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_colwidth", 200)
pd.set_option('display.max_columns', 200)


# In[2]:


customer=pd.read_csv(r'D:\Machine Learning\Customer_Churn\churn_prediction.csv')
customer.head()                  


# In[3]:


customer.shape


# The dataset has 28328  observations with 21 coulmns
# 

# In[4]:


customer.info()


# Gender & Occupation are in object form in the data, rest are of float and interger category.
# 
# Next lets see the distribution of target variable.

# In[5]:


plt.figure(figsize=(4,4))
ax = sns.countplot(x="churn", data=customer)

for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))

plt.show()


# The number of churning customer(1) is generally much lower than non churning customer(0) in a bank.As of now we will keep it as it is, and run the model. 

# ### Missing value imputation

# In[6]:


customer.isnull().sum()


# In[7]:


customer['gender'].value_counts(dropna=False)


# So there is a good mix of males and females and arguably missing values cannot be filled with any one of them. We could create a seperate category by assigning the value -1 for all missing values in this column.
# 
# Before that, first we will convert the gender into 0/1 and then replace missing values with -1

# In[8]:


dict_gender = {'Male': 1, 'Female':0}
customer.replace({'gender': dict_gender}, inplace = True)

customer['gender'] = customer['gender'].fillna(-1)


# In[9]:


customer['dependents'].value_counts(dropna=False)


# In[10]:


#we are filling the null values with mode
# also the frequesncy of dependents>7 is very less, we are assimilating all those frequencies to 8
customer['dependents'] = customer['dependents'].fillna(0)
customer.loc[customer['dependents']>7,'dependents']=8


# In[11]:


customer['dependents'].value_counts(dropna=False)


# In[12]:


customer['occupation'].value_counts(dropna=False)


# In[13]:


#replacing null values with mode
customer['occupation'] = customer['occupation'].fillna('self_employed')


# In[14]:


customer['city'].value_counts(dropna=False)


# In[15]:


# since in city there is  a high null value, we are asigining a unique value to it
customer['city'] = customer['city'].fillna(0)


# In[16]:


customer['days_since_last_transaction'].describe()


# In[17]:


#days_since_last_transaction - No of Days Since Last Credit in Last 1 year,A fair assumption can be made on this column as this is number of days since last transaction in 1 year, we can substitute missing values with a value greater than 1 year say (365+30)
customer['days_since_last_transaction'] = customer['days_since_last_transaction'].fillna(395)


# In[18]:


customer.isnull().sum()


# ### Feature Engineering (Creating New Features)

# ###### Categorical Features

# In[19]:


customer['seniority']=customer['age'].apply(lambda x:'senior citizen' if x>=60 else 'non senior citizen')


# In[20]:


customer['dependency_status']=customer['dependents'].apply(lambda x:'no dependents' if x==0 else 'with dependents')


# In[21]:


customer['vintage_yr']=round(customer['vintage']/365)


# In[22]:


customer['vintage_yr'].describe()


# In[23]:


customer['last_trans_month']=round(customer['days_since_last_transaction']/30)


# In[24]:


customer['last_trans_month'].value_counts()


# ### Exploratory Data Analysis

# In[25]:


categorical_var=['gender','occupation','customer_nw_category','seniority','dependency_status']


# In[26]:


def graph_plot(feature):
    crosstab=pd.crosstab(index=customer[feature],columns=customer['churn'])
    total_count=pd.DataFrame(customer[feature].value_counts())
    crosstable=pd.concat([crosstab,total_count],axis=1)
    crosstable['nchurn']=(crosstable[0]/crosstable[feature])*100
    crosstable['churn']=(crosstable[1]/crosstable[feature])*100
    percentage_table=crosstable[['nchurn','churn']]
    plt.figure(figsize=(8,5))
    ax = percentage_table.plot.bar(rot=0)
    ax.set_title(feature)
    for container in ax.containers:
           ax.bar_label(container,fmt='%.2f')


# In[27]:


categorical_var=['gender','occupation','customer_nw_category','seniority','dependency_status']
for i in categorical_var:
    graph_plot(i)


# ##### Observations
# 
# * Male customers have higher rate of churning than female customers
# * Self Employed people has highest rate of churning
# * High & low net worth individuals have higher rate of churning than middle nw individuals
# * Non Senior Citizens (age>60) have higher tendency of churning
# * People with dependency have higher rate of churning

# In[28]:


sns.kdeplot(data=customer,x='vintage_yr',hue='churn')


# In[29]:


sns.kdeplot(data=customer,x='last_trans_month',hue='churn')


# ###### Observation
# 
# * People who have relationship with the bank for less or around 1 year is more likely to churn,fpr them who have a relationship with the bank for almost 5 years have the least change of churning out.
# * Transactions in last 1/2 months or transaction more than 12 months ago have higher rate of churning- its not shwing any particular pattern at this point.

# In[30]:


len(pd.unique(customer['branch_code']))


# In[31]:


len(pd.unique(customer['city']))


# In[32]:


plt.figure(figsize=(16,14))
corrplot = sns.heatmap(customer.corr(), cmap="YlGnBu", annot=True)
plt.show()


# There is large number of unique categories in both city and branch code, and also the correlations between the target feature and these two features are very low, these two features wont be able to give us much relevant information.
# 
# * Although there is a corelation between customer net worth & branch code, it is safe to assume like every bank there are some premium branches which serves high net worth customer and so on. 
# * current_balance & current_month_balance has high corelation, we can take only current balance for further analysis
# * average_monthly_balance_prevQ & previous_month_balance is highly corelated,we can take only average_monthly_balance_prevQ
# * the credit & debit features are also highly corelated with each other, since debit features have slightly higher corelation with target variable we will take the debit coulmns for further analysis

# ##### Numerical Features

# In[33]:


numerical_col=['current_balance','previous_month_balance', 'current_month_debit','previous_month_debit']
for i in numerical_col:
    sns.kdeplot(data=customer,x=i,hue='churn')
    plt.show()


# ### Scaling Numerical Features 
# Now, we can see that there are a lot of outliers in the dataset especially when it comes to previous and current balance features. Also, the distributions are skewed for these features. We will take 2 steps to deal with that here:
# 
# Log Transformation
# Standard Scaler
# Standard scaling is anyways a necessity when it comes to linear models and we have done that here after doing log transformation on all balance features.

# In[34]:


num_cols = ['customer_nw_category', 'current_balance',
            'previous_month_end_balance', 'average_monthly_balance_prevQ2', 'average_monthly_balance_prevQ',
            'current_month_credit','previous_month_credit', 'current_month_debit', 
            'previous_month_debit','current_month_balance', 'previous_month_balance']
for i in num_cols:
    customer[i] = np.log(customer[i] + 17000)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
scaled = std.fit_transform(customer[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)


# In[35]:


customer_org = customer.copy()
customer = customer.drop(columns = num_cols,axis = 1)
customer = customer.merge(scaled,left_index=True,right_index=True,how = "left")


# In[36]:


numerical_col=['current_balance','previous_month_balance', 'current_month_debit','previous_month_debit']
for i in numerical_col:
    sns.kdeplot(data=customer,x=i,hue='churn')
    plt.show()


# In[37]:


customer.head()


# In[38]:


import os
os.chdir('D:\Machine Learning\Customer_Churn')


# In[40]:


customer.to_csv('cust.csv',index=False)


# In[ ]:





# In[ ]:




