#!/usr/bin/env python
# coding: utf-8

# # SPAM MAIL DETECTION

# In[1]:


#importing neccesary libraries
import numpy as np # numercial python, arrays
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


#reading the dataset using pandas dataframe
data=pd.read_csv('spam.csv',encoding="latin1")


# In[5]:


data.head(10)


# In[6]:


data.tail(10)


# In[7]:


data.columns


# In[8]:


data.info


# In[10]:


data.describe()


# In[12]:


#check the number of rows and columns present in df
print('Rows:',data.shape[0])
print('Columns:',data.shape[1])


# In[14]:


#Lets see null value count in df
data.isnull().sum()


# In[16]:


data.isnull().mean()*100  #check the percentage of null value


# In[18]:


data.drop(columns=data[['Unnamed: 2','Unnamed: 3','Unnamed: 4']],axis=1,inplace=True)
data


# In[20]:


data.shape


# In[22]:


data.columns=['spam/ham','sms']


# In[25]:


#Convert the text data into numerical form
data.loc[data['spam/ham'] == 'spam', 'spam/ham',] = 0
data.loc[data['spam/ham'] == 'ham', 'spam/ham',] = 1


# In[27]:


data


# In[29]:


x=data.sms
x


# In[31]:


y =data['spam/ham']
y


# In[32]:


#models
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[33]:


#splitting into testing and training
train, test = train_test_split(data, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)


# In[35]:


print(x.shape)
print(x_train.shape)
print(x_test.shape)


# In[36]:


print(y.shape)
print(y_train.shape)
print(y_test.shape)


# In[38]:


x_train,x_test


# In[39]:


y_train,y_test


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
feat_vect=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
feat_vect


# In[42]:


y_train=y_train.astype('int')
y_test=y_test.astype('int')


# In[45]:


x_train_vec =feat_vect.fit_transform(x_train)


# In[47]:


x_test_vec =feat_vect.transform(x_test)


# In[48]:


print(x_train)


# In[50]:


x_train_vec


# In[52]:


print(x_train_vec)


# In[53]:


print(x_test_vec)


# In[54]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(x_train_vec, y_train)
prediction = model.predict(x_test_vec)
print('Accuracy Score:',metrics.accuracy_score(prediction,y_test)*100)


# In[55]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(y_test,prediction))


# In[56]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(x_train_vec,y_train)

pred = model1.predict(x_test_vec)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,pred)*100)


# In[57]:


#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train_vec,y_train)
predict = model2.predict(x_test_vec)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,predict)*100)


# In[58]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model3.fit(x_train_vec,y_train)
y_pred = model3.predict(x_test_vec)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)


# In[60]:


#Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model4=RandomForestClassifier(n_estimators=100)
model4.fit(x_train_vec,y_train)
pred_y=model4.predict(x_test_vec)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)


# In[63]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines','KNN' ,'Decision Tree', 'Random Forest'],
    'Score': [96.23,98.29,97.36,90.58,96.59]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:




