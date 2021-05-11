#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_mat = pd.read_csv('../dataset/original/student-mat.csv', sep=';')


# In[3]:


df_mat


# In[4]:


df_mat.shape


# In[5]:


df_mat.info()


# In[6]:


df_por = pd.read_csv('../dataset/original/student-por.csv', sep=';')


# In[7]:


df_por


# In[8]:


df_por.shape


# In[9]:


df_por.info()


# In[10]:


merged_df = pd.concat([df_mat, df_por])


# In[11]:


merged_df


# In[12]:


merged_df.head()


# In[13]:


merged_df.tail()


# In[14]:


merged_df.shape


# In[15]:


merged_df.shape[0]


# In[16]:


merged_df.info()


# In[17]:


merged_df_null = merged_df[merged_df.isnull().any(axis=1)]


# In[18]:


merged_df_null


# In[19]:


merged_df_null.shape


# In[20]:


merged_df_null.shape[0]


# In[21]:


merged_df_not_null = merged_df[merged_df.notna().all(axis=1)]


# In[22]:


merged_df_not_null


# In[23]:


merged_df_not_null.shape


# In[24]:


merged_df_not_null.shape[0]


# In[25]:


merged_df = merged_df_not_null


# In[26]:


merged_df.head()


# In[27]:


merged_df.tail()


# In[28]:


merged_df.shape


# In[29]:


merged_df.shape[0]


# In[30]:


merged_df.info()


# In[31]:


merged_df.columns


# In[32]:


merged_df.describe()


# In[33]:


file_name = '../dataset/processed/student-performance-dataset.csv'


# In[34]:


merged_df.to_csv(file_name, encoding='utf-8', index=False)


# In[35]:


df = pd.read_csv(file_name)


# In[36]:


df


# In[37]:


df.head()


# In[38]:


df.tail()


# In[39]:


df.shape


# In[40]:


df_null = df[df.isnull().any(axis=1)]


# In[41]:


df_null


# In[42]:


df_null.shape


# In[43]:


df_null.shape[0]


# In[44]:


df.duplicated().sum()


# In[45]:


df_drop_duplicate = df.drop_duplicates()


# In[46]:


df_drop_duplicate


# In[47]:


df.duplicated().sum()


# In[48]:


df.info()


# In[49]:


df.columns


# In[50]:


df.describe()


# In[50]:




