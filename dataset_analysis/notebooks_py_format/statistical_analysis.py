#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../dataset/processed/student-performance-dataset.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df_null = df[df.isnull().any(axis=1)]


# In[6]:


df_null


# In[7]:


df.info()


# In[8]:


df.columns


# In[9]:


df.describe()


# In[10]:


df.describe().round(2)


# In[11]:


df['age'].describe().round(2)


# In[12]:


df['failures'].describe().round(2)


# In[13]:


df['age'].describe().round(2)


# In[14]:


df['failures'].describe().round(2)


# In[15]:


df['famrel'].describe().round(2)


# In[16]:


df['freetime'].describe().round(2)


# In[17]:


df['goout'].describe().round(2)


# In[18]:


df['Dalc'].describe().round(2)


# In[19]:


df['Walc'].describe().round(2)


# In[20]:


df['health'].describe().round(2)


# In[21]:


df['absences'].describe().round(2)


# In[22]:


df['G1'].describe().round(2)


# In[23]:


df['G2'].describe().round(2)


# In[24]:


df['G3'].describe().round(2)


# In[24]:




