#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # import Datasets

# In[3]:


data=pd.read_csv("C:\\Users\\byral\\OneDrive\\Desktop\\heart.csv")


# # Knowing information about dataset(with method .info())

# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.nunique()


# # check for missing value

# In[7]:


data.isnull().sum()


# # Cleaning Datasets

# In[8]:


data.dropna(inplace=True)
data.isnull().sum()


# In[9]:


data.shape


# # See the Categorical Values

# In[10]:


data.head(10)


# # visualising dataset

# In[11]:


data.plot()
plt.show()


# In[168]:


data['HeartDisease'].value_counts().plot(kind='bar')


# # Extracting Independent & dependent Variable

# In[12]:


X=data.iloc[:,:-1]
y=data.iloc[:,-1]


# # encode the Categorical Variable

# In[13]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[14]:


ct1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,6,8,10])], remainder='passthrough')


# In[15]:


X=ct1.fit_transform(X)


# In[16]:


X=pd.DataFrame(X)


# In[17]:


X.hist(figsize=(15,15))
plt.show()


# In[18]:


X.plot()
plt.show()


# # Using Feature selection technique [RFE]

# In[19]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[20]:


# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 5)
fit = rfe.fit(X, y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


# In[21]:


fit


# In[22]:


fit.n_features_


# In[23]:


features = fit.transform(X)


# In[24]:


d=pd.DataFrame(features)


# In[25]:


d.hist(figsize=(10,10))
plt.show()


# # Splitting the data-set into Training and Test Set

# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.2, random_state = 4)


# In[27]:


X_train_d=pd.DataFrame(X_train)


# In[221]:


X_train_d.head()


# # Feature Scaling

# In[28]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)


# In[29]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[30]:


X_train


# # Trainig Decision Tree Classifier

# In[31]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[32]:


y_pred=classifier.predict(X_test)


# # comparing Results

# In[33]:


y_test=np.array(y_test)
y_pred=np.array(y_pred)


# In[34]:


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[35]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# # Checking for Accurecy

# In[36]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[37]:


import seaborn as sa
sa.heatmap(cm, annot = True)
plt.show()


# In[227]:


print('DecisionTreeClassifierModel Train Score is : ' , classifier.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is : ' , classifier.score(X_test, y_test))


# In[195]:


recall_score(y_test,y_pred)


# In[196]:


precision_score(y_test,y_pred)


# In[197]:


f1_score(y_test,y_pred)


# In[268]:


import matplotlib.pyplot as plt
importance = classifier.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# # checking for overfitting

# In[231]:


from sklearn import linear_model 
Lasso_reg= linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)


# In[232]:


Lasso_reg.fit(X_train,y_train)


# In[233]:


Lasso_reg.score(X_train,y_train)


# In[234]:


Lasso_reg.score(X_test,y_test)


# # Ending our Project

# In[ ]:




