#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_por = pd.read_csv('../dataset/original/student-por.csv', sep=';')


# In[3]:


df_por


# In[4]:


df_por.head()


# In[5]:


df_por.tail()


# In[6]:


df_por.shape


# In[7]:


df_por.shape[0]


# In[8]:


df_por.info()


# In[9]:


df_por_null = df_por[df_por.isnull().any(axis=1)]


# In[10]:


df_por_null


# In[11]:


df_por_null.shape


# In[12]:


df_por_null.shape[0]


# In[13]:


df_por_not_null = df_por[df_por.notna().all(axis=1)]


# In[14]:


df_por_not_null


# In[15]:


df_por_not_null.shape


# In[16]:


df_por_not_null.shape[0]


# In[17]:


df_por = df_por_not_null


# In[18]:


df_por.head()


# In[19]:


df_por.tail()


# In[20]:


df_por.shape


# In[21]:


df_por.shape[0]


# In[22]:


df_por.info()


# In[23]:


df_por.columns


# In[24]:


df_por.describe()


# In[24]:




