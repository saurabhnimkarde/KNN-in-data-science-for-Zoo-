#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[2]:


zoo=pd.read_csv('Zoo.csv')


# In[3]:


zoo.head()


# In[4]:


zoo.shape


# In[5]:


zoo = zoo.rename({'animal name':'animal'},axis = 1)


# In[6]:


zoo.info()


# In[7]:


y=zoo['type'].values
X=zoo.drop(['type','animal'],axis=1).values


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


# In[9]:


knn.score(X_train,y_train)


# In[10]:


knn.score(X_test,y_test)


# In[21]:


#We Obtained 80 % accuracy by using KNN


# In[11]:


preds=knn.predict(X_test)


# In[12]:


preds


# In[13]:


from sklearn.metrics import classification_report


# In[14]:


print(classification_report(preds,y_test))


# In[ ]:




