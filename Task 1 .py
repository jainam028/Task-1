#!/usr/bin/env python
# coding: utf-8

# ## Task1 - Predict the percentage of marks of an student based on the number of study hours
# 

# Prediction using supervised ML with the help of linear Regression

# Name : Jainam Shah

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading the CSV file

# In[2]:


student_df = pd.read_csv('student.csv')


# In[3]:


student_df


# In[4]:


# Lets see few top rows with help of head()
student_df.head()


# In[5]:


# With the help of tail method we can see bottom rows..
student_df.head()


# In[6]:


student_df.dtypes


# In[7]:


#with the help of shape method we can see number of rows and columns present in the csv file
student_df.shape


# In[8]:


# checking the null values
student_df.isna().sum()


# From above we can say there are no NAN values ..

# In[9]:


student_df.info()


# In[10]:


#describe() is used to view some basic statistical details like percentile, mean, std etc.
student_df.describe()


# In[11]:


# Plotting the distribution of scores
student_df.plot(x = 'Hours' , y = 'Scores' , style = 'o' , c = 'blue')
plt.title('Hours vs Scores')
plt.xlabel('No_of_hours_studied')
plt.ylabel('Percentage score')
plt.show()


# We can say that from above graph that as number of hours increase then gradually percentage also increases

# ## Data Preparation

# In[12]:


x = student_df.drop(['Scores'],axis = 1)
y = student_df['Scores']


# In[13]:


x.head()


# In[14]:


y.head()


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5,random_state = 0)


# In[16]:


x_train.shape


# In[17]:


x_test.shape


# ## Training the model

# In[18]:


from sklearn.linear_model import LinearRegression
regres = LinearRegression()
regres.fit(x_train , y_train)


# ## Plotting the Regression Line as well as scatter plot

# In[19]:


line = regres.coef_*x + regres.intercept_
plt.scatter(x,y,c='red')
plt.plot(x,line,c = 'green')
plt.show()


# ## Making Predictions :)

# In[20]:


# Testing Data in hours
print(x_test)


# In[21]:


# Predict the scores
y_pred = regres.predict(x_test)


# In[22]:


y


# In[23]:


student_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})


# In[24]:


student_df


# We have to predict for 9.25 hours

# In[25]:


hours = 9.25
our_prediction = regres.predict([[hours]])
print("Number of hours = {}".format(hours))
print("Predicted Score = {}".format(our_prediction))


# ## Evaluating the model

# In[26]:


from sklearn import metrics
print('Mean absolute error: ',metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




