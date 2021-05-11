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


df['sex'].value_counts(sort=False)


# In[6]:


fig = plt.figure()
ax = df['sex'].value_counts(sort=False).plot(kind='pie',
                                             labels=['female', 'male'],
                                             autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Gender')
plt.legend()
fig.savefig('column_data_analysis_plots/gender.png', facecolor='white')
plt.show()


# In[7]:


df['age'].value_counts(sort=False)


# In[8]:


fig = plt.figure()
df['age'].value_counts(sort=False).plot(kind='bar')
plt.title('Age')
plt.xlabel('age')
plt.ylabel('student count')
fig.savefig('column_data_analysis_plots/age.png', facecolor='white')
plt.show()


# In[9]:


df['address'].value_counts(sort=False)


# In[10]:


fig = plt.figure()
ax = df['address'].value_counts(sort=False).plot(kind='pie',
                                                 labels=['rural', 'urban'],
                                                 autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Address')
plt.legend()
fig.savefig('column_data_analysis_plots/address.png', facecolor='white')
plt.show()


# In[11]:


df['famsize'].value_counts(sort=False)


# In[12]:


fig = plt.figure()
ax = df['famsize'].value_counts(sort=False).plot(kind='pie',
                                                 labels=['3 or less', 'more than 3'],
                                                 autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Family Size')
plt.legend()
fig.savefig('column_data_analysis_plots/family_size.png', facecolor='white')
plt.show()


# In[13]:


df['Mjob'].value_counts(sort=False)


# In[14]:


fig = plt.figure()
ax = df['Mjob'].value_counts(sort=False).plot(kind='pie',
                                              autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Mother\'s Job')
plt.legend()
fig.savefig('column_data_analysis_plots/mothers_job.png', facecolor='white')
plt.show()


# In[15]:


df['Fjob'].value_counts(sort=False)


# In[16]:


fig = plt.figure()
ax = df['Fjob'].value_counts(sort=False).plot(kind='pie',
                                              autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Father\'s Job')
plt.legend()
fig.savefig('column_data_analysis_plots/fathers_job.png', facecolor='white')
plt.show()


# In[17]:


df['reason'].value_counts(sort=False)


# In[18]:


fig = plt.figure()
ax = df['reason'].value_counts(sort=False).plot(kind='pie',
                                                autopct='%1.0f%%',
                                                labels=['course preference', 'other', 'school reputation',
                                                        'close to home'])
ax.yaxis.set_visible(False)
plt.title('Reason for School Selection')
plt.legend()
fig.savefig('column_data_analysis_plots/reason_for_school_selection.png', facecolor='white')
plt.show()


# In[19]:


df['guardian'].value_counts(sort=False)


# In[20]:


fig = plt.figure()
ax = df['guardian'].value_counts(sort=False).plot(kind='pie',
                                                  autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Guardian')
plt.legend()
fig.savefig('column_data_analysis_plots/guardian.png', facecolor='white')
plt.show()


# In[21]:


df['Pstatus'].value_counts(sort=False)


# In[22]:


fig = plt.figure()
ax = df['Pstatus'].value_counts(sort=False).plot(kind='pie',
                                                 labels=['apart', 'living together'],
                                                 autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Parent\'s Cohabitation Status')
plt.legend()
fig.savefig('column_data_analysis_plots/parents_cohabitation_status.png', facecolor='white')
plt.show()


# In[23]:


df['Medu'].value_counts(sort=False)


# In[24]:


fig = plt.figure(figsize=(12, 18))
ax = df['Medu'].value_counts(sort=False).plot(kind='bar')
plt.title('Mother\'s Education')
plt.xlabel('education level')
plt.ylabel('student count')
ax.set_xticklabels(['none', 'primary education', '5th to 9th grade', 'secondary education', 'higher education'])
fig.savefig('column_data_analysis_plots/mothers_education.png', facecolor='white')
plt.show()


# In[25]:


df['Fedu'].value_counts(sort=False)


# In[26]:


fig = plt.figure(figsize=(12, 18))
ax = df['Fedu'].value_counts(sort=False).plot(kind='bar')
plt.title('Father\'s Education')
plt.xlabel('education level')
plt.ylabel('student count')
ax.set_xticklabels(['none', 'primary education', '5th to 9th grade', 'secondary education', 'higher education'])
fig.savefig('column_data_analysis_plots/fathers_education.png', facecolor='white')
plt.show()


# In[27]:


df['traveltime'].value_counts(sort=False)


# In[28]:


fig = plt.figure(figsize=(8, 16))
ax = df['traveltime'].value_counts(sort=False).plot(kind='bar')
plt.title('Travel Time')
plt.xlabel('travel time')
plt.ylabel('student count')
ax.set_xticklabels(['less than 15 min', '15 to 30 min', '30 min to 1 hour', 'more than 1 hour'])
fig.savefig('column_data_analysis_plots/travel_time.png', facecolor='white')
plt.show()


# In[29]:


df['studytime'].value_counts(sort=False)


# In[30]:


fig = plt.figure(figsize=(8, 16))
ax = df['studytime'].value_counts(sort=False).plot(kind='bar')
plt.title('Study Time')
plt.xlabel('study time')
plt.ylabel('student count')
ax.set_xticklabels(['less than 2 hrs', '2 to 5 hours', '5 to 10 hours', 'more than 10 hours'])
fig.savefig('column_data_analysis_plots/study_time.png', facecolor='white')
plt.show()


# In[31]:


df['failures'].value_counts(sort=False)


# In[32]:


fig = plt.figure(figsize=(10, 16))
ax = df['failures'].value_counts(sort=False).plot(kind='bar')
plt.title('Failures')
plt.xlabel('failures')
plt.ylabel('student count')
ax.set_xticklabels(['0', '1', '2', '3 or more'])
fig.savefig('column_data_analysis_plots/failures.png', facecolor='white')
plt.show()


# In[33]:


df['schoolsup'].value_counts(sort=False)


# In[34]:


fig = plt.figure()
ax = df['schoolsup'].value_counts(sort=False).plot(kind='pie',
                                                   autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Extra Educational Support')
plt.legend()
fig.savefig('column_data_analysis_plots/extra_educational_support.png', facecolor='white')
plt.show()


# In[35]:


df['famsup'].value_counts(sort=False)


# In[36]:


fig = plt.figure()
ax = df['famsup'].value_counts(sort=False).plot(kind='pie',
                                                autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Family Educational Support')
plt.legend()
fig.savefig('column_data_analysis_plots/family_educational_support.png', facecolor='white')
plt.show()


# In[37]:


df['paid'].value_counts(sort=False)


# In[38]:


fig = plt.figure()
ax = df['paid'].value_counts(sort=False).plot(kind='pie',
                                              autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Extra Paid Classes')
plt.legend()
fig.savefig('column_data_analysis_plots/extra_paid_classes.png', facecolor='white')
plt.show()


# In[39]:


df['activities'].value_counts(sort=False)


# In[40]:


fig = plt.figure()
ax = df['activities'].value_counts(sort=False).plot(kind='pie',
                                                    autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Extra Curricular Activities')
plt.legend()
fig.savefig('column_data_analysis_plots/extra_curricular_activities.png', facecolor='white')
plt.show()


# In[41]:


df['nursery'].value_counts(sort=False)


# In[42]:


fig = plt.figure()
ax = df['nursery'].value_counts(sort=False).plot(kind='pie',
                                                 autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Attended Nursery School')
plt.legend()
fig.savefig('column_data_analysis_plots/attended_nursery_school.png', facecolor='white')
plt.show()


# In[43]:


df['higher'].value_counts(sort=False)


# In[44]:


fig = plt.figure()
ax = df['higher'].value_counts(sort=False).plot(kind='pie',
                                                autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Want Higher Education')
plt.legend()
fig.savefig('column_data_analysis_plots/want_higher_education.png', facecolor='white')
plt.show()


# In[45]:


df['internet'].value_counts(sort=False)


# In[46]:


fig = plt.figure()
ax = df['internet'].value_counts(sort=False).plot(kind='pie',
                                                  autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Internet Access')
plt.legend()
fig.savefig('column_data_analysis_plots/internet_access.png', facecolor='white')
plt.show()


# In[47]:


df['romantic'].value_counts(sort=False)


# In[48]:


fig = plt.figure()
ax = df['romantic'].value_counts(sort=False).plot(kind='pie',
                                                  autopct='%1.0f%%')
ax.yaxis.set_visible(False)
plt.title('Romantic Relationship')
plt.legend()
fig.savefig('column_data_analysis_plots/romantic_relationship.png', facecolor='white')
plt.show()


# In[49]:


df['famrel'].value_counts(sort=False)


# In[50]:


fig = plt.figure()
df['famrel'].value_counts(sort=False).plot(kind='bar')
plt.title('Family Relationships Quality')
plt.xlabel('family relationships quality')
plt.ylabel('student count')
fig.savefig('column_data_analysis_plots/family_relationships_quality.png', facecolor='white')
plt.show()


# In[51]:


df['freetime'].value_counts(sort=False)


# In[52]:


fig = plt.figure()
df['freetime'].value_counts(sort=False).plot(kind='bar')
plt.title('Free Time After School')
plt.xlabel('free time after school')
plt.ylabel('student count')
fig.savefig('column_data_analysis_plots/free_time_after_school.png', facecolor='white')
plt.show()


# In[53]:


df['goout'].value_counts(sort=False)


# In[54]:


fig = plt.figure()
df['goout'].value_counts(sort=False).plot(kind='bar')
plt.title('Outings with Friends')
plt.xlabel('outings with friends')
plt.ylabel('student count')
fig.savefig('column_data_analysis_plots/outings_with_friends.png', facecolor='white')
plt.show()


# In[55]:


df['Dalc'].value_counts(sort=False)


# In[56]:


fig = plt.figure()
df['Dalc'].value_counts(sort=False).plot(kind='bar')
plt.title('Workday Alcohol Consumption')
plt.xlabel('workday alcohol consumption')
plt.ylabel('student count')
fig.savefig('column_data_analysis_plots/workday_alcohol_consumption.png', facecolor='white')
plt.show()


# In[57]:


df['Walc'].value_counts(sort=False)


# In[58]:


fig = plt.figure()
df['Walc'].value_counts(sort=False).plot(kind='bar')
plt.title('Weekend Alcohol Consumption')
plt.xlabel('weekend alcohol consumption')
plt.ylabel('student count')
fig.savefig('column_data_analysis_plots/weekend_alcohol_consumption.png', facecolor='white')
plt.show()


# In[59]:


df['health'].value_counts(sort=False)


# In[60]:


fig = plt.figure()
df['health'].value_counts(sort=False).plot(kind='bar')
plt.title('Heath Status')
plt.xlabel('health status')
plt.ylabel('student count')
fig.savefig('column_data_analysis_plots/health_status.png', facecolor='white')
plt.show()


# In[61]:


import numpy as np


# In[62]:


df['absences'].value_counts(sort=False)


# In[63]:


fig = plt.figure()
df['absences'].value_counts(sort=False).plot(kind='line')
plt.title('Absences')
plt.xlabel('absences')
plt.ylabel('student count')
plt.xticks([i for i in range(0, 80, 10)])
fig.savefig('column_data_analysis_plots/absences.png', facecolor='white')
plt.show()


# In[64]:


df['G1'].value_counts(sort=False)


# In[65]:


fig = plt.figure()
df['G1'].value_counts(sort=False).plot(kind='line')
plt.title('Grade 1')
plt.xlabel('grade 1')
plt.ylabel('student count')
plt.xticks([i for i in range(0, 21, 2)])
fig.savefig('column_data_analysis_plots/g1.png', facecolor='white')
plt.show()


# In[66]:


df['G2'].value_counts(sort=False)


# In[67]:


fig = plt.figure()
df['G2'].value_counts(sort=False).plot(kind='line')
plt.title('Grade 2')
plt.xlabel('grade 2')
plt.ylabel('student count')
plt.xticks([i for i in range(0, 21, 2)])
fig.savefig('column_data_analysis_plots/g2.png', facecolor='white')
plt.show()


# In[68]:


df['G3'].value_counts(sort=False)


# In[69]:


fig = plt.figure()
df['G3'].value_counts(sort=False).plot(kind='line')
plt.title('Grade 3')
plt.xlabel('grade 3')
plt.ylabel('student count')
plt.xticks([i for i in range(0, 21, 2)])
fig.savefig('column_data_analysis_plots/g3.png', facecolor='white')
plt.show()


# In[69]:




