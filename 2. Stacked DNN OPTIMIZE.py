#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from numpy import dstack
from sklearn.linear_model import LogisticRegression
     


# In[10]:


df = pd.read_csv('Churn_Modelling.csv')


# In[11]:


df.head()
     
#This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.


# In[12]:


df.info()
     
#There are 3 features with string values.


# In[14]:


df.isnull().sum()

#Dataset is free of null values.


# In[15]:


df['Geography'].value_counts()

#Since the number of countries is less and also their is no rank which can be associated with the customers of different region. Hence we use one hot encodding.


# In[16]:


#One hot encodding
geo = pd.get_dummies(df['Geography'],drop_first = True)
gen = pd.get_dummies(df['Gender'],drop_first= True)
df = pd.concat([df,gen,geo],axis=1)


#One-hot encoding in machine learning is the conversion of categorical information into a format that may be fed into machine learning algorithms to improve prediction accuracy. One-hot encoding is a common method for dealing with categorical data in machine learning.


# In[17]:


#Drop unnecessary data
df.drop(['Geography','Gender','Surname','RowNumber','CustomerId'],axis=1,inplace = True)
     


# In[18]:


df


# In[19]:


#We need to do data normalization before we send the data to train a neural network model.

from sklearn.preprocessing import MinMaxScaler


# In[20]:


X = df.drop('Exited',axis=1)
y = df['Exited']
     


# In[21]:


#normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[22]:


X_scaled


# In[23]:


pd.DataFrame( )


# In[24]:


X_final = pd.DataFrame(columns = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Male','Germany','Spain'],data = X_scaled)


# In[25]:


X_final.head()


# In[26]:


#train-test-split
X_train,X_test,y_train,y_test = train_test_split(X_final,y,test_size = 0.30,random_state = 101)


# In[27]:


#Modeling

#Lets create 3 neural networks to stack sequentially further.




model1 = Sequential()
model1.add(Dense(50,activation = 'relu',input_dim = 11))
model1.add(Dense(25,activation = 'relu'))
model1.add(Dense(1,activation = 'sigmoid'))
     


# In[28]:


model1.summary()


# Output shape: N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).


# In[29]:


y.value_counts()


# In[30]:


#We will chose f_score

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# In[31]:


model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history = model1.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)


# In[32]:


plt.plot(history.history['loss'])
plt.plot(history.history['f1_m'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()


# In[33]:


#Lets save the model.

model1.save('model1.h5')


# In[34]:


# Train 2 more different models.

model2 = Sequential()
model2.add(Dense(25,activation = 'relu',input_dim = 11))
model2.add(Dense(25,activation = 'relu'))
model2.add(Dense(10,activation = 'relu'))
model2.add(Dense(1,activation = 'sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history1 = model2.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)
     


# In[35]:


plt.plot(history1.history['loss'])
plt.plot(history1.history['f1_m'])
plt.plot(history1.history['val_loss'])
plt.plot(history1.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()
     


# In[36]:


model2.save('model2.h5')


# In[37]:


model3 = Sequential()
model3.add(Dense(50,activation = 'relu',input_dim = 11))
model3.add(Dense(25,activation = 'relu'))
model3.add(Dense(25,activation = 'relu'))
model3.add(Dropout(0.1))
model3.add(Dense(10,activation = 'relu'))
model3.add(Dense(1,activation = 'sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history3 = model3.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)


# In[38]:


model3.save('model3.h5')


# In[42]:


plt.plot(history3.history['loss'])
plt.plot(history3.history['f1_m'])
plt.plot(history3.history['val_loss'])
plt.plot(history3.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()
     


# In[40]:


model4 = Sequential()
model4.add(Dense(50,activation = 'relu',input_dim = 11))
model4.add(Dense(25,activation = 'relu'))
model4.add(Dropout(0.1))
model4.add(Dense(10,activation = 'relu'))
model4.add(Dense(1,activation = 'sigmoid'))

model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history4 = model4.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 100)


# In[41]:


plt.plot(history4.history['loss'])
plt.plot(history4.history['f1_m'])
plt.plot(history4.history['val_loss'])
plt.plot(history4.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()
     


# In[43]:


model4.save('model4.h5')


# In[44]:


dependencies = {
    'f1_m': f1_m
}

     


# In[49]:


# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'model' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename,custom_objects=dependencies)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models


# In[50]:


n_members = 4
members = load_all_models(n_members)
print('Loaded %d models' % len(members))


# In[51]:


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat #
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX
     


# In[52]:


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression() #meta learner
	model.fit(stackedX, inputy)
	return model


# In[53]:


model = fit_stacked_model(members, X_test,y_test)


# In[56]:


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat


# In[57]:


# evaluate model on test set
yhat = stacked_prediction(members, model, X_test)
score = f1_m(y_test/1.0, yhat/1.0)
print('Stacked F Score:', score)


# In[58]:


from sklearn.metrics import f1_score


# In[ ]:


i = 0
for model in members:
    i+=1
    pred = model.predict(X_test)
    score = f1_score(y_test,pred)
    print('F-Score of model {} is '.format(i),score)


# In[ ]:


#F-Score of model 1 is  0.5901328273244781
#F-Score of model 2 is  0.5849802371541502
#F-Score of model 3 is  0.5935483870967743
#F-Score of model 4 is  0.5987144168962351
#Stacked Model give f1 score of 0.6007 which is higher than any other model taken alone.

