#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../dataset/processed/student-performance-dataset.csv')


# In[3]:


df


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


def draw_plot_basic(attr):
    x = df[attr]
    y = df['G3']
    plt.scatter(x, y)
    plt.title(attr)
    plt.savefig(f'data_relationships_plots/basic/{attr}.png', facecolor='white')
    plt.show()


# In[6]:


def draw_plot_median(attr):
    x = df[attr].unique()
    x.sort()
    y = df.groupby(attr)['G3'].median()
    plt.scatter(x, y)
    plt.title(attr)
    plt.savefig(f'data_relationships_plots/median/{attr}.png', facecolor='white')
    plt.show()


# In[7]:


def draw_plot_mean(attr):
    x = df[attr].unique()
    x.sort()
    y = df.groupby(attr)['G3'].mean()
    plt.scatter(x, y)
    plt.title(attr)
    plt.savefig(f'data_relationships_plots/mean/{attr}.png', facecolor='white')
    plt.show()


# In[8]:


def draw_plots(attr):
    draw_plot_basic(attr)
    draw_plot_mean(attr)
    draw_plot_median(attr)


# In[9]:


for column in df.columns:
    draw_plots(column)


# In[9]:




