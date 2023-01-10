#!/usr/bin/env python
# coding: utf-8

# In the previous notebook we have done 
# * Missing Value Imputation
# * EDA of a bank customer churning data,
# * Created some new features
# * Scaled the numerical features
# 
# Next in this notebook, we will use that processed data to:
# * Preprocessing
# * Select features on the basis of EDA Conclusions & finalise columns for model
# * Decide Evaluation Metric on the basis of business problem
# * Build model -Logistic,Random Forest,XGBoost using Cross Validation
# * Hyperparameter Tuning using Grid Search CV
# * Use Reverse Feature Elimination to find the top features and build model using the top 10 features & compare
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_colwidth", 200)
pd.set_option('display.max_columns', 200)


# In[2]:


cust=pd.read_csv('D:\Machine Learning\Customer_Churn\cust.csv')
cust.head()


# In[3]:


cust.shape


# In[4]:


cust.isnull().sum()


# There is no missing value in the data currently.
# 
# During EDA we have created some new features, next we will do one hot encoding and check how they are affecting the target variables

# In[5]:


categorical_col=['occupation','seniority','dependency_status']
cust = pd.concat([cust,pd.get_dummies(cust[categorical_col])],axis = 1)


# In[6]:


cust.head()


# In[7]:


cust.drop(columns=['customer_id','occupation','seniority','dependency_status'],inplace=True)


# In[8]:


cust.head()


# In[9]:


corr_matrix=cust.corr()*100
corr_matrix["churn"].sort_values(ascending=False)


# In[10]:


-10.027668>-10.031528


# In[11]:


corr_matrix[['customer_nw_category', 'current_balance',
            'previous_month_end_balance', 'average_monthly_balance_prevQ2', 'average_monthly_balance_prevQ',
            'current_month_credit','previous_month_credit', 'current_month_debit', 
            'previous_month_debit','current_month_balance', 'previous_month_balance']]


# In[12]:


X=cust[['current_balance','previous_month_balance','average_monthly_balance_prevQ','average_monthly_balance_prevQ2', 'current_month_credit','previous_month_credit', 'current_month_debit', 
            'previous_month_debit','branch_code','gender','customer_nw_category','city','occupation_self_employed','occupation_company','occupation_student','occupation_salaried','occupation_retired','seniority_senior citizen','seniority_non senior citizen','dependency_status_no dependents','dependency_status_with dependents','vintage_yr','last_trans_month']]


# In[13]:


X.head()


# In[14]:


X.shape


# In[15]:


y_all=cust[['churn']]


# ### Train Test Split to create a validation set

# In[16]:


xtrain, xtest, ytrain, ytest = train_test_split(X,y_all,test_size=1/4, random_state=11, stratify = y_all)


# ## Model Building and Evaluation Metrics
# Since this is a binary classification problem, we could use the following 2 popular metrics:
# 
# 1. Recall
# 2. Area under the Receiver operating characteristic curve
# 
# Now, we are looking at the recall value here because a customer falsely marked as churn would not be as bad as a customer who was not detected as a churning customer and appropriate measures were not taken by the bank to stop him/her from churning
# 
# The ROC AUC is the area under the curve when plotting the (normalized) true positive rate (x-axis) and the false positive rate (y-axis).
# 

# def cv_score(ml_model, rstate = 12, thres = 0.5, cols = df.columns):
#     i = 1
#     cv_scores = []
#     df1 = df.copy()
#     df1 = df[cols]
#     
#     # 5 Fold cross validation stratified on the basis of target
#     kf = StratifiedKFold(n_splits=5,random_state=rstate,shuffle=True)
#     for df_index,test_index in kf.split(df1,y_all):
#         print('\n{} of kfold {}'.format(i,kf.n_splits))
#         xtr,xvl = df1.loc[df_index],df1.loc[test_index]
#         ytr,yvl = y_all.loc[df_index],y_all.loc[test_index]
#             
#         # Define model for fitting on the training set for each fold
#         model = ml_model
#         model.fit(xtr, ytr)
#         pred_probs = model.predict_proba(xvl)
#         pp = []
#          
#         # Use threshold to define the classes based on probability values
#         for j in pred_probs[:,1]:
#             if j>thres:
#                 pp.append(1)
#             else:
#                 pp.append(0)
#          
#         # Calculate scores for each fold and print
#         pred_val = pp
#         roc_score = roc_auc_score(yvl,pred_probs[:,1])
#         recall = recall_score(yvl,pred_val)
#         precision = precision_score(yvl,pred_val)
#         sufix = ""
#         msg = ""
#         msg += "ROC AUC Score: {}, Recall Score: {:.4f}, Precision Score: {:.4f} ".format(roc_score, recall,precision)
#         print("{}".format(msg))
#          
#          # Save scores
#         cv_scores.append(roc_score)
#         i+=1
#     return cv_scores

# #####  Creating a funtion that will give Recall ,Accuracy ,ROC_AUC Score for  each ML model under each validation segment.

# In[17]:


def get_score(ml_model):
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    model = ml_model()
    score = cross_validate(model, xtrain, ytrain, scoring=['recall','accuracy','roc_auc'], cv=cv, n_jobs=-1)
    return score
    


# In[18]:


get_score(LogisticRegression)


# In[19]:


get_score(SVC)


# In[20]:


get_score(GaussianNB)


# In[21]:


get_score(DecisionTreeClassifier)


# In[22]:


get_score(RandomForestClassifier)


# * We are getting the best recall value from Decesion Tree, but the accuracy & ROC-AUC of DT is lower than Random Forest or Logistic regression
# * Random Forest is giving us highest ROC-AUC & Accuarcy Score, also the recall score is better than most of the ML model, hence we will take Random Forest as our final Model to do the prediction

# #### Random Forest Implimentation as Final Model

# In[23]:


clf = RandomForestClassifier(n_estimators = 100)
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)


# In[24]:


print('Parameters currently in use:\n')
print(clf.get_params())


# In[25]:


from sklearn.metrics import ConfusionMatrixDisplay
conf_mat = confusion_matrix(ytest, ypred)
ConfusionMatrixDisplay(conf_mat).plot()


# In[26]:


recall_score(ytest,ypred)


# In[27]:


accuracy_score(ytest,ypred)


# In[28]:


roc_auc_score(ytest,ypred)


# #### Hyperparameter Tuning in Random Forest

# In[29]:


param_rf={'criterion':['gini', 'entropy', 'log_loss'],'max_depth':range(0,10),
          'ccp_alpha': [0.0,0.1, .01, .001],'max_features':['sqrt', 'log2', 'None']}
          
rf = RandomForestClassifier()
grid_rf=GridSearchCV(estimator=rf,param_grid=param_rf,scoring='roc_auc',cv=5)
gridresult_rf=grid_rf.fit(xtrain, ytrain.values.ravel())
score_rf=gridresult_rf.best_score_
score_rf


# In[30]:


gridresult_rf.best_params_


# In[31]:


clf_pm = RandomForestClassifier(ccp_alpha=0.0,
criterion='entropy',
max_depth=9,
max_features='log2',n_estimators=100)
clf_pm.fit(xtrain, ytrain)
ypred_pm = clf_pm.predict(xtest)


# In[32]:


print('Parameters currently in use:\n')
print(clf_pm.get_params())


# {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

# In[33]:


roc_auc_score(ytest,ypred_pm)


# In[34]:


accuracy_score(ytest,ypred_pm)


# In[35]:


recall_score(ytest,ypred_pm)


# ### Reverse Feature Elimination or Backward Selection
# 
# We have already built a model using our selected features and hyper tuned it. We can try using backward feature elimination to check if we can do better. Let's do that next.

# In[36]:


from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Create the RFE object and rank each feature
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=1, step=1)
rfe.fit(xtrain, ytrain)


# In[37]:


ranking_df = pd.DataFrame()
ranking_df['Feature_name'] = xtrain.columns
ranking_df['Rank'] = rfe.ranking_


# In[38]:


ranking_df=ranking_df.sort_values(by=['Rank'],ascending=True)


# In[39]:


ranking_df


# In[43]:


model=RandomForestClassifier()
mean_accuracy=[]
feature_number=[]
for i in range(1,24):
    mean_score=cross_val_score(model,xtrain[ranking_df.iloc[:i,0]],ytrain,scoring='roc_auc',cv=5).mean()
    feature_number.append(i)
    mean_accuracy.append(mean_score)


# In[44]:


mean_score


# In[52]:


d={'mean_accuracy':mean_accuracy,'feature_number':feature_number}
roc_df=pd.DataFrame(d)
roc_df


# In[57]:


sns.lineplot(x=feature_number,y=mean_accuracy)


# The balance features are proving to be very important as can be seen from the table. The RFE function can also be used to select features. Lets select the top 8 features from this table and check score.

# In[63]:


model_top=RandomForestClassifier()
model_top.fit(xtrain[ranking_df.iloc[:8,0]],ytrain)
ypred_top= clf.predict(xtest)
print(recall_score(ytest,ypred_top))
print(accuracy_score(ytest,ypred_top))
print(roc_auc_score(ytest,ypred_top))


# We can with the help of feature selection our Recall & ROC AUC values have improved than hyper parameter tuning.

# In this Project,
# * I have done exploratory data analysis to understand how different variable effect customer churning
# * Prepare the data for modelling
# * Select the best model (Random Forest) using Cross Validation
# * Have done Hyper parameter tuning on the model, using Grid Search
# * Using Backward Feature Elimination and calcualting mean AUC ROC score ,selected the best 8 features for model
# 
# After the final model our
# * Accuracy Score is 86%
# * Recall Score is 40%
# * AUC ROC Score is 68%

# In[ ]:




