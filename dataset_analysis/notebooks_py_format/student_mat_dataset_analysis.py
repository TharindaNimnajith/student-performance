#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_mat = pd.read_csv('../dataset/original/student-mat.csv', sep=';')


# In[3]:


df_mat


# In[4]:


df_mat.head()


# In[5]:


df_mat.tail()


# In[6]:


df_mat.shape


# In[7]:


df_mat.shape[0]


# In[8]:


df_mat.info()


# In[9]:


df_mat_null = df_mat[df_mat.isnull().any(axis=1)]


# In[10]:


df_mat_null


# In[11]:


df_mat_null.shape


# In[12]:


df_mat_null.shape[0]


# In[13]:


df_mat_not_null = df_mat[df_mat.notna().all(axis=1)]


# In[14]:


df_mat_not_null


# In[15]:


df_mat_not_null.shape


# In[16]:


df_mat_not_null.shape[0]


# In[17]:


df_mat = df_mat_not_null


# In[18]:


df_mat.head()


# In[19]:


df_mat.tail()


# In[20]:


df_mat.shape


# In[21]:


df_mat.shape[0]


# In[22]:


df_mat.info()


# In[23]:


df_mat.columns


# In[24]:


df_mat.describe()


# In[24]:




