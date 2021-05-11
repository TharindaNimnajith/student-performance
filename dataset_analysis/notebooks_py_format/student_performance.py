#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing pandas library to perform data manipulation and analysis
import pandas as pd

pd.options.display.max_columns = None


# In[2]:


# importing the student-mat.csv dataset to a pandas dataframe
# dataset includes the student performances for mathematics subject
df_mat = pd.read_csv('../dataset/original/student-mat.csv', sep=';')
df_mat.insert(0, 'Subject', 0)
df_mat


# In[3]:


# importing the student-por.csv dataset to a pandas dataframe
# dataset includes the student performances for portuguese subject
df_por = pd.read_csv('../dataset/original/student-por.csv', sep=';')
df_por.insert(0, 'Subject', 1)
df_por


# In[4]:


# merging the two dataframes into one
merged_df = pd.concat([df_mat, df_por])
merged_df = merged_df.reset_index(drop=True)
merged_df


# In[5]:


# dropping irrelevant attribute columns from the dataframe
merged_df.drop(['Subject', 'school', 'age'], axis='columns', inplace=True)
merged_df


# In[6]:


# adding a new column called status to the dataframe based on the G3 value
# categorizing the students into three classes based on their final examination grades
merged_df['status'] = merged_df.apply(lambda x: 0 if x['G3'] <= 7 else (1 if x['G3'] <= 14 else 2), axis=1)
merged_df


# In[7]:


# removing the missing or null values from the dataframe if exist
merged_df = merged_df[merged_df.notna().all(axis=1)]

# checking for missing or null values in the dataframe
merged_df_null = merged_df[merged_df.isnull().any(axis=1)]
merged_df_null


# In[8]:


# importing pyplot from matplotlib library to create interactive visualizations
import matplotlib.pyplot as plt

# importing seaborn library which is built on top of matplotlib to create statistical graphics
import seaborn as sns

# plotting the heatmap for missing or null values in the dataframe
fig = plt.figure(figsize=(9, 8))
sns.heatmap(merged_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Null Values Detection Heat Map')
fig.savefig('preprocessing_plots/null_detection_heat_map.png', facecolor='white')
plt.show()


# In[9]:


# removing the duplicate rows from the dataframe if exist
merged_df = merged_df.drop_duplicates()

# checking the number of duplicate rows exist in the dataframe
merged_df.duplicated().sum()


# In[10]:


merged_df


# In[11]:


# displaying the first 5 rows of the dataframe
merged_df.head()


# In[12]:


# displaying the last 5 rows of the dataframe
merged_df.tail()


# In[13]:


# displaying the dimensionality of the dataframe
merged_df.shape


# In[14]:


# printing a concise summary of the dataframe
# information such as index, data type, columns, non-null values, and memory usage
merged_df.info()


# In[15]:


# copying the dataset without the status attribute
df_without_status = merged_df.loc[:, merged_df.columns != 'status']
df_without_status


# In[16]:


# generating descriptive statistics of the dataframe
df_without_status.describe().round(2)


# In[17]:


# displaying the unbiased variance of the numeric columns of the dataframe
variance_vector = df_without_status.var()
variance_vector


# In[18]:


# computing and displaying pair-wise correlation of columns of the dataframe
correlation_matrix = df_without_status.corr().abs()
correlation_matrix


# In[19]:


# sorting the attribute correlation values with G3 in descending order
correlation_matrix['G3'].sort_values(ascending=False)


# In[20]:


# importing numpy library to perform fast mathematical operations over arrays and matrices
import numpy as np

upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
upper_tri


# In[21]:


# identifying one attribute out of each highly correlated column pair
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.6)]
to_drop


# In[22]:


# identifying highly correlated column pairs

to_drop_pairs = []

for i in range(len(upper_tri.columns)):
    if any(upper_tri[upper_tri.columns[i]] > 0.6):
        to_drop_pairs.append([upper_tri.columns[i], upper_tri.columns[i - 1]])

to_drop_pairs


# In[23]:


# plotting the heatmap for correlation matrix
fig = plt.figure(figsize=(13, 12))
sns.heatmap(df_without_status[df_without_status.columns].corr().abs(), annot=True)
plt.title('Correlation Matrix')
fig.savefig('preprocessing_plots/correlation_matrix.png', facecolor='white')
plt.show()


# In[24]:


from pandas.api import types

# identifying the columns with numeric values
numeric_columns = [i for i in merged_df.columns if types.is_numeric_dtype(merged_df[i])]
numeric_columns


# In[25]:


# identifying the columns with string values
string_columns = [i for i in merged_df.columns if types.is_string_dtype(merged_df[i])]
string_columns


# In[26]:


# dropping G3 column
# status column represents the target variable now
# G3 is just a duplicate representation of the status
merged_df.drop(['G3'], axis='columns', inplace=True)
merged_df


# In[27]:


# assigning all attributes except status (features) to X
X = merged_df.loc[:, merged_df.columns != 'status']
X


# In[28]:


# assigning status (target) to y
y = merged_df.loc[:, 'status']
y


# In[29]:


# importing LabelEncoder from scikit-learn library
# converting every categorical value in a column to a number
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for column in string_columns:
    X.loc[:, column] = label_encoder.fit_transform(X.loc[:, column])

X


# In[30]:


# importing OneHotEncoder from scikit-learn library
# solving the hierarchical issue of label encoding
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse=False)

# since Mjob contains 5 categories, 5 columns will be created
X['mother_job_at_home'] = one_hot_encoder.fit_transform(X[['Mjob']])[:, 0]
X['mother_job_health'] = one_hot_encoder.fit_transform(X[['Mjob']])[:, 1]
X['mother_job_other'] = one_hot_encoder.fit_transform(X[['Mjob']])[:, 2]
X['mother_job_services'] = one_hot_encoder.fit_transform(X[['Mjob']])[:, 3]
X['mother_job_teacher'] = one_hot_encoder.fit_transform(X[['Mjob']])[:, 4]

# since Fjob contains 5 categories, 5 columns will be created
X['father_job_at_home'] = one_hot_encoder.fit_transform(X[['Fjob']])[:, 0]
X['father_job_health'] = one_hot_encoder.fit_transform(X[['Fjob']])[:, 1]
X['father_job_other'] = one_hot_encoder.fit_transform(X[['Fjob']])[:, 2]
X['father_job_services'] = one_hot_encoder.fit_transform(X[['Fjob']])[:, 3]
X['father_job_teacher'] = one_hot_encoder.fit_transform(X[['Fjob']])[:, 4]

# since reason contains 4 categories, 4 columns will be created
X['reason_course'] = one_hot_encoder.fit_transform(X[['reason']])[:, 0]
X['reason_home'] = one_hot_encoder.fit_transform(X[['reason']])[:, 1]
X['reason_other'] = one_hot_encoder.fit_transform(X[['reason']])[:, 2]
X['reason_reputation'] = one_hot_encoder.fit_transform(X[['reason']])[:, 3]

# since guardian contains 3 categories, 3 columns will be created
X['guardian_father'] = one_hot_encoder.fit_transform(X[['guardian']])[:, 0]
X['guardian_mother'] = one_hot_encoder.fit_transform(X[['guardian']])[:, 1]
X['guardian_other'] = one_hot_encoder.fit_transform(X[['guardian']])[:, 2]

X


# In[31]:


# dropping the initial four attributes from the dataframe
X.drop(['Mjob', 'Fjob', 'reason', 'guardian'], axis='columns', inplace=True)
X


# In[32]:


# displaying the unbiased variance of the numeric columns of the dataframe
variance_vector = X.var()
variance_vector


# In[33]:


# computing and displaying pair-wise correlation of columns of the dataframe
correlation_matrix = X.corr().abs()
correlation_matrix


# In[34]:


# plotting the heatmap for correlation matrix
fig = plt.figure(figsize=(40, 40))
sns.heatmap(X[X.columns].corr().abs(), annot=True)
plt.title('Correlation Matrix for Features')
fig.savefig('preprocessing_plots/correlation_matrix_for_features.png', facecolor='white')
plt.show()


# In[35]:


upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
upper_tri


# In[36]:


# identifying one attribute out of each highly correlated column pair
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.6)]
to_drop


# In[37]:


# identifying highly correlated column pairs

to_drop_pairs = []

for i in range(len(upper_tri.columns)):
    if any(upper_tri[upper_tri.columns[i]] > 0.6):
        to_drop_pairs.append([upper_tri.columns[i], upper_tri.columns[i - 1]])

to_drop_pairs


# In[38]:


# dropping one attribute out of each highly correlated column pair
X.drop(['Fedu', 'Walc', 'G1'], axis='columns', inplace=True)
X


# In[39]:


# importing XGBClassifier from xgboost library
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X, y)

# detecting feature importance of the training attributes
importances = model.feature_importances_

importance_list = []

for i, v in enumerate(importances):
    importance_list.append([i, X.columns[i], v])

importance_list = sorted(importance_list, reverse=True, key=lambda x: x[2])
importance_list


# In[40]:


# plotting the bar chart for feature importance
plt.bar([x for x in range(len(importances))], importances)
plt.title('Feature Importance')
plt.savefig('preprocessing_plots/feature_importance.png', facecolor='white')
plt.show()


# In[41]:


# getting indices of the top 12 important features
indices = np.argsort(importances)[-1:-13:-1]
indices


# In[42]:


# assigning columns of only the top 12 important features to X
X = X.iloc[:, indices]
X


# In[43]:


# importing train_test_split from scikit-learn library
from sklearn.model_selection import train_test_split

# splitting data into random train and test subsets
# train set - 80%, test set - 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[44]:


X_train
# shape - (835, 12)


# In[45]:


X_test
# shape - (209, 12)


# In[46]:


y_train
# shape - (835, 1)


# In[47]:


y_test
# shape - (209, 1)


# In[48]:


# importing KNeighborsClassifier from scikit-learn library
# KNN is a simple machine learning algorithm based on supervised learning
# it can be used for both regression and classification problems
from sklearn.neighbors import KNeighborsClassifier

# finding the best value for k with the minimum error

error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

best_k = error.index(min(error)) + 1
best_k


# In[49]:


# plotting the k values with the error rate
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('knn - error rate for k value')
plt.xlabel('k value')
plt.ylabel('mean error')
plt.savefig('preprocessing_plots/knn_mean_error_for_k.png', facecolor='white')
plt.show()


# In[50]:


# training the KNN model
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)


# In[51]:


# returning parameters of the trained model
knn_model.get_params()


# In[52]:


# importing pickle module
# used for serializing and deserializing a python object structure
import pickle

# saving the trained model to a pickle file
with open('models/knn.pkl', 'wb') as model_file:
    pickle.dump(knn_model, model_file)


# In[53]:


# loading the trained model from the pickle file
with open('models/knn.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)


# In[54]:


# predicting labels of X_test data values on the basis of the trained model
y_pred_knn = knn_model.predict(X_test)
y_pred_knn


# In[55]:


# importing mean_squared_error from scikit-learn library
from sklearn.metrics import mean_squared_error

# mean squared error (MSE)
print(mean_squared_error(y_test, y_pred_knn))

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted and actual value of variable
print(mean_squared_error(y_test, y_pred_knn, squared=False))


# In[56]:


# importing mean_absolute_error from scikit-learn library
from sklearn.metrics import mean_absolute_error

# mean absolute error (MAE)
mean_absolute_error(y_test, y_pred_knn)


# In[57]:


# importing accuracy_score from scikit-learn library
from sklearn.metrics import accuracy_score

# accuracy
# ratio of the number of correct predictions to the total number of input samples
accuracy_score(y_test, y_pred_knn)


# In[58]:


# importing precision_recall_fscore_support from scikit-learn library
from sklearn.metrics import precision_recall_fscore_support

# computing precision, recall, f-measure and support for each class
print(precision_recall_fscore_support(y_test, y_pred_knn, average='micro'))
print(precision_recall_fscore_support(y_test, y_pred_knn, average='macro'))
print(precision_recall_fscore_support(y_test, y_pred_knn, average='weighted'))


# In[59]:


# importing classification_report from scikit-learn library
# used to measure the quality of predictions from a classification algorithm
from sklearn.metrics import classification_report

target_names = list(map(str, sorted(y.unique().tolist())))

# report shows the main classification metrics precision, recall and f1-score on a per-class basis
print(classification_report(y_test, y_pred_knn, target_names=target_names))


# In[60]:


# importing confusion_matrix from scikit-learn library
from sklearn.metrics import confusion_matrix

# confusion matrix is a summarized table used to assess the performance of a classification model
# number of correct and incorrect predictions are summarized with their count according to each class
print(confusion_matrix(y_test, y_pred_knn))


# In[61]:


# importing plot_confusion_matrix from scikit-learn library
from sklearn.metrics import plot_confusion_matrix

titles_options = [('knn_confusion_matrix_without_normalization', None),
                  ('knn_normalized_confusion_matrix', 'true')]

# plotting the confusion matrix
for title, normalize in titles_options:
    cm_plot = plot_confusion_matrix(knn_model, X_test, y_test, display_labels=target_names, cmap=plt.cm.Blues,
                                    normalize=normalize)
    cm_plot.ax_.set_title(title)
    plt.savefig(f'preprocessing_plots/{title}.png', facecolor='white')

plt.show()


# In[62]:


# plotting scatter plot to visualize overlapping of predicted and test target data points
plt.scatter(range(len(y_pred_knn)), y_pred_knn, color='red')
plt.scatter(range(len(y_test)), y_test, color='green')
plt.title('knn - predicted status vs. real status')
plt.xlabel('students')
plt.ylabel('status')
plt.savefig('preprocessing_plots/knn_predicted_vs_real.png', facecolor='white')
plt.show()


# In[63]:


# importing DecisionTreeClassifier from scikit-learn library
# decision tree algorithm uses a decision tree as a predictive model to go from observations
# about an item represented in the branches to conclusions about the item's target value
# represented in the leaves
from sklearn.tree import DecisionTreeClassifier

# importing GridSearchCV from scikit-learn library
# used to loop through predefined hyper-parameters and fit the model on the training set
from sklearn.model_selection import GridSearchCV

# importing warnings library to handle exceptions, errors, and warning of the program
import warnings

# ignoring potential warnings of the program
warnings.filterwarnings('ignore')

param_grid = {
    'criterion': ['gini', 'entropy']
}

dt_model = DecisionTreeClassifier()

# looping through criterion hyper-parameters (gini and entropy) of decision tree algorithm
# and fit the model on the training set
grid_search_cv = GridSearchCV(dt_model, param_grid, scoring='f1', cv=5)
grid_search_cv.fit(X_train, y_train)

# printing the best value for the parameters
grid_search_cv.best_params_


# In[64]:


# training the decision tree model
# default criterion hyper-parameter value is gini
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[65]:


# returning parameters of the trained model
dt_model.get_params()


# In[66]:


# getting depth of the decision tree
print(dt_model.get_depth())

# getting number of leaves of the decision tree
print(dt_model.get_n_leaves())


# In[67]:


# saving the trained model to a pickle file
with open('models/dt.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)


# In[68]:


# loading the trained model from the pickle file
with open('models/dt.pkl', 'rb') as model_file:
    dt_model = pickle.load(model_file)


# In[69]:


# predicting labels of X_test data values on the basis of the trained model
y_pred_dt = dt_model.predict(X_test)
y_pred_dt


# In[70]:


# mean squared error (MSE)
print(mean_squared_error(y_test, y_pred_dt))

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted and actual value of variable
print(mean_squared_error(y_test, y_pred_dt, squared=False))


# In[71]:


# mean absolute error (MAE)
mean_absolute_error(y_test, y_pred_dt)


# In[72]:


# accuracy
# ratio of the number of correct predictions to the total number of input samples
accuracy_score(y_test, y_pred_dt)


# In[73]:


# computing precision, recall, f-measure and support for each class
print(precision_recall_fscore_support(y_test, y_pred_dt, average='micro'))
print(precision_recall_fscore_support(y_test, y_pred_dt, average='macro'))
print(precision_recall_fscore_support(y_test, y_pred_dt, average='weighted'))


# In[74]:


target_names = list(map(str, sorted(y.unique().tolist())))

# report shows the main classification metrics precision, recall and f1-score on a per-class basis
print(classification_report(y_test, y_pred_dt, target_names=target_names))


# In[75]:


# confusion matrix is a summarized table used to assess the performance of a classification model
# number of correct and incorrect predictions are summarized with their count according to each class
print(confusion_matrix(y_test, y_pred_dt))


# In[76]:


titles_options = [('dt_confusion_matrix_without_normalization', None),
                  ('dt_normalized_confusion_matrix', 'true')]

# plotting the confusion matrix
for title, normalize in titles_options:
    cm_plot = plot_confusion_matrix(dt_model, X_test, y_test, display_labels=target_names, cmap=plt.cm.Blues,
                                    normalize=normalize)
    cm_plot.ax_.set_title(title)
    plt.savefig(f'preprocessing_plots/{title}.png', facecolor='white')

plt.show()


# In[77]:


# plotting scatter plot to visualize overlapping of predicted and test target data points
plt.scatter(range(len(y_pred_dt)), y_pred_dt, color='red')
plt.scatter(range(len(y_test)), y_test, color='green')
plt.title('dt - predicted status vs. real status')
plt.xlabel('students')
plt.ylabel('status')
plt.savefig('preprocessing_plots/dt_predicted_vs_real.png', facecolor='white')
plt.show()


# In[78]:


# importing graphviz library
# used to create graph objects which can be completed using different nodes and edges
import graphviz

# importing export_graphviz from scikit-learn library
# used to export a decision tree in DOT format to work with the graphviz library
from sklearn.tree import export_graphviz

# plotting the decision tree graph
graphviz_plot = export_graphviz(dt_model, feature_names=X.columns)
graph = graphviz.Source(graphviz_plot)
graph.render(filename='preprocessing_plots/dt_graphviz_plot', format='png', cleanup=True)
graph.view()


# In[79]:


# displaying the decision tree graph
graph


# In[80]:


# importing RandomForestClassifier from scikit-learn library
# random forest is a supervised learning algorithm
# it is used for both classification and regression
# it is comprised of multiple decision trees
from sklearn.ensemble import RandomForestClassifier

# n_estimators - number of trees to be built before taking maximum voting or averages of predictions
n_estimators = list(range(1, 150, 3))

param_grid = {
    'n_estimators': n_estimators
}

rf_model = RandomForestClassifier(oob_score=True)

# looping through n_estimators hyper-parameters (sequence of numbers from 1 to 149, increment by 3)
# of random forest algorithm and fit the model on the training set
grid_search_cv = GridSearchCV(rf_model, param_grid, cv=5)
grid_search_cv.fit(X, y)

# printing the best value for the parameters
grid_search_cv.best_params_


# In[81]:


# printing the mean_test_score of cv_results_
grid_search_cv.cv_results_['mean_test_score']


# In[82]:


scores = grid_search_cv.cv_results_['mean_test_score']

# plotting the accuracy against n_estimators value for random forest classifier model
plt.plot(n_estimators, scores)
plt.title('rf - accuracy and n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.savefig('preprocessing_plots/rf_accuracy_n_estimators.png', facecolor='white')
plt.show()


# In[83]:


# training the random forest model
rf_model = RandomForestClassifier(n_estimators=grid_search_cv.best_params_['n_estimators'], oob_score=True)
rf_model.fit(X_train, y_train)


# In[84]:


# returning parameters of the trained model
rf_model.get_params()


# In[85]:


# getting n_estimators value of the trained model
rf_model.n_estimators


# In[86]:


# getting oob_score_ value of the trained model
# oob_score_ - out-of-bag score
rf_model.oob_score_


# In[87]:


# saving the trained model to a pickle file
with open('models/rf.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)


# In[88]:


# loading the trained model from the pickle file
with open('models/rf.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)


# In[89]:


# predicting labels of X_test data values on the basis of the trained model
y_pred_rf = rf_model.predict(X_test)
y_pred_rf


# In[90]:


# mean squared error (MSE)
print(mean_squared_error(y_test, y_pred_rf))

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted and actual value of variable
print(mean_squared_error(y_test, y_pred_rf, squared=False))


# In[91]:


# mean absolute error (MAE)
mean_absolute_error(y_test, y_pred_rf)


# In[92]:


# accuracy
# ratio of the number of correct predictions to the total number of input samples
accuracy_score(y_test, y_pred_rf)


# In[93]:


# computing precision, recall, f-measure and support for each class
print(precision_recall_fscore_support(y_test, y_pred_rf, average='micro'))
print(precision_recall_fscore_support(y_test, y_pred_rf, average='macro'))
print(precision_recall_fscore_support(y_test, y_pred_rf, average='weighted'))


# In[94]:


target_names = list(map(str, sorted(y.unique().tolist())))

# report shows the main classification metrics precision, recall and f1-score on a per-class basis
print(classification_report(y_test, y_pred_rf, target_names=target_names))


# In[95]:


# confusion matrix is a summarized table used to assess the performance of a classification model
# number of correct and incorrect predictions are summarized with their count according to each class
print(confusion_matrix(y_test, y_pred_rf))


# In[96]:


titles_options = [('rf_confusion_matrix_without_normalization', None),
                  ('rf_normalized_confusion_matrix', 'true')]

# plotting the confusion matrix
for title, normalize in titles_options:
    cm_plot = plot_confusion_matrix(rf_model, X_test, y_test, display_labels=target_names, cmap=plt.cm.Blues,
                                    normalize=normalize)
    cm_plot.ax_.set_title(title)
    plt.savefig(f'preprocessing_plots/{title}.png', facecolor='white')

plt.show()


# In[97]:


# plotting scatter plot to visualize overlapping of predicted and test target data points
plt.scatter(range(len(y_pred_rf)), y_pred_rf, color='red')
plt.scatter(range(len(y_test)), y_test, color='green')
plt.title('rf - predicted status vs. real status')
plt.xlabel('students')
plt.ylabel('status')
plt.savefig('preprocessing_plots/rf_predicted_vs_real.png', facecolor='white')
plt.show()


# In[98]:


# there are mainly three types of Naive Bayes models
# they are GaussianNB, MultinomialNB, and BernoulliNB
# importing GaussianNB from scikit-learn library
from sklearn.naive_bayes import GaussianNB

# training the gaussian naive bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


# In[99]:


# returning parameters of the trained model
nb_model.get_params()


# In[100]:


# saving the trained model to a pickle file
with open('models/nb.pkl', 'wb') as model_file:
    pickle.dump(nb_model, model_file)


# In[101]:


# loading the trained model from the pickle file
with open('models/nb.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)


# In[102]:


# predicting labels of X_test data values on the basis of the trained model
y_pred_nb = nb_model.predict(X_test)
y_pred_nb


# In[103]:


# mean squared error (MSE)
print(mean_squared_error(y_test, y_pred_nb))

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted and actual value of variable
print(mean_squared_error(y_test, y_pred_nb, squared=False))


# In[104]:


# mean absolute error (MAE)
mean_absolute_error(y_test, y_pred_nb)


# In[105]:


# accuracy
# ratio of the number of correct predictions to the total number of input samples
accuracy_score(y_test, y_pred_nb)


# In[106]:


# computing precision, recall, f-measure and support for each class
print(precision_recall_fscore_support(y_test, y_pred_nb, average='micro'))
print(precision_recall_fscore_support(y_test, y_pred_nb, average='macro'))
print(precision_recall_fscore_support(y_test, y_pred_nb, average='weighted'))


# In[107]:


target_names = list(map(str, sorted(y.unique().tolist())))

# report shows the main classification metrics precision, recall and f1-score on a per-class basis
print(classification_report(y_test, y_pred_nb, target_names=target_names))


# In[108]:


# confusion matrix is a summarized table used to assess the performance of a classification model
# number of correct and incorrect predictions are summarized with their count according to each class
print(confusion_matrix(y_test, y_pred_nb))


# In[109]:


titles_options = [('nb_confusion_matrix_without_normalization', None),
                  ('nb_normalized_confusion_matrix', 'true')]

# plotting the confusion matrix
for title, normalize in titles_options:
    cm_plot = plot_confusion_matrix(nb_model, X_test, y_test, display_labels=target_names, cmap=plt.cm.Blues,
                                    normalize=normalize)
    cm_plot.ax_.set_title(title)
    plt.savefig(f'preprocessing_plots/{title}.png', facecolor='white')

plt.show()


# In[110]:


# plotting scatter plot to visualize overlapping of predicted and test target data points
plt.scatter(range(len(y_pred_nb)), y_pred_nb, color='red')
plt.scatter(range(len(y_test)), y_test, color='green')
plt.title('nb - predicted status vs. real status')
plt.xlabel('students')
plt.ylabel('status')
plt.savefig('preprocessing_plots/nb_predicted_vs_real.png', facecolor='white')
plt.show()


# In[111]:


# importing recall_score, precision_score, f1_score from scikit-learn library
from sklearn.metrics import recall_score, precision_score, f1_score

accuracy = {}
recall = {}
precision = {}
f1 = {}
rmse = {}
mae = {}

y_pred_dict = {
    'KNN': y_pred_knn,
    'DT': y_pred_dt,
    'RF': y_pred_rf,
    'NB': y_pred_nb,
}

for y_pred in y_pred_dict:
    accuracy[y_pred] = accuracy_score(y_test, y_pred_dict[y_pred])
    recall[y_pred] = recall_score(y_test, y_pred_dict[y_pred], average='weighted')
    precision[y_pred] = precision_score(y_test, y_pred_dict[y_pred], average='weighted')
    f1[y_pred] = f1_score(y_test, y_pred_dict[y_pred], average='weighted')
    rmse[y_pred] = mean_squared_error(y_test, y_pred_dict[y_pred], squared=False)
    mae[y_pred] = mean_absolute_error(y_test, y_pred_dict[y_pred])

# ratio of the number of correct predictions to the total number of input samples
print(accuracy)

# recall is the ratio tp / (tp + fn)
# tp is the number of true positives
# fn the number of false negatives
print(recall)

# precision is the ratio tp / (tp + fp)
# tp is the number of true positives
# fp the number of false positives
print(precision)

# F1 score is also known as balanced F-score or F-measure
# F1 score can be interpreted as a weighted average of the precision and recall
# F1 = 2 * (precision * recall) / (precision + recall)
print(f1)

# root mean squared error (RMSE)
# square root of the average of squared differences between predicted and actual value of variable
print(rmse)

# mean absolute error (MAE)
print(mae)


# In[112]:


# sorting the accuracy scores of four models in descending order
sorted(accuracy.items(), key=lambda kv: kv[1], reverse=True)


# In[113]:


# plotting the accuracy comparison bar chart
plt.bar(y_pred_dict.keys(), accuracy.values(), color='rgbc')
plt.title('Accuracy Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.savefig('preprocessing_plots/accuracy_comparison.png', facecolor='white')
plt.show()


# In[114]:


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars
    bars = []
    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
        bars.append(bar[0])
    if legend:
        ax.legend(bars, data.keys())


# In[115]:


data = {}

for key in y_pred_dict.keys():
    data[key] = [accuracy[key], recall[key], precision[key], f1[key], rmse[key], mae[key]]

fig, ax = plt.subplots()
bar_plot(ax, data, total_width=.9, single_width=.9)

# plotting the algorithm comparison chart for all evaluation metrics
plt.title('Algorithm Comparison')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.xticks(range(len(data[key])), ['accuracy', 'recall', 'precision', 'f1', 'rmse', 'mae'])
plt.savefig('preprocessing_plots/algorithm_comparison.png', facecolor='white')
plt.show()


# In[115]:




