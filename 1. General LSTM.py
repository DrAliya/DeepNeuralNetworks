#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


#Create a 2-D feature NumPy array with random integers
features = (np.random.randint(10, size=(100, 1)))
print(features.shape)


# In[3]:


#Split the dataset into 75/25 for train and test.
training_dataset_length = math.ceil(len(features) * .75)
print(training_dataset_length)


# In[4]:


#Scale all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)


# In[5]:


#Here we predict the 11th value using [1,2,….,10]. Here N = 100, and the sliding window size is l = 10. So x_train will contain values of sliding windows of l = 10, and y_train will contain values of every l+1 value we want to predict.
train_data = scaled_data[0:training_dataset_length , : ]
#Splitting the data
x_train=[]
y_train = []
for i in range(10, len(train_data)):
    x_train.append(train_data[i-10:i,0])
    y_train.append(train_data[i,0])


# In[6]:


#Then converting the x_train and y_train into NumPy array values and reshaping it into a 3-D array, shape accepted by the LSTM model.
#Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data into 3-D array
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[7]:


from keras.layers import Dropout
# Initialising the RNN
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and Dropout layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and Dropout layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and and Dropout layer
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
# For Full connection layer we use dense
# As the output is 1D so we use unit=1
model.add(Dense(units = 1))


# In[8]:


#compile and fit the model on 30 epochs
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 30, batch_size = 50)


# In[9]:


#Crete test data similar to train data, convert to NumPy array and reshape the array to 3-D shape.

#Test data set
test_data = scaled_data[training_dataset_length - 10: , : ]
#splitting the x_test and y_test data sets
x_test = []
y_test = features[training_dataset_length : , : ]
for i in range(10,len(test_data)):
    x_test.append(test_data[i-10:i,0])
#Convert x_test to a numpy array
x_test = np.array(x_test)
#Reshape the data into 3-D array
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# In[10]:


#Making the predictions and calculating the rmse score(smaller the rmse score, better the model has performed).

#check predicted values
predictions = model.predict(x_test)

#Undo scaling

predictions = scaler.inverse_transform(predictions)

#Calculate RMSE score

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

rmse


# In[11]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# define input sequence
input_seq = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])

# define model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(3, 1)))
model.add(Dense(1))

# compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# reshape input data for LSTM
input_seq = np.reshape(input_seq, (input_seq.shape[0], 3, 1))

# train model
model.fit(input_seq, np.array([0.7, 0.8, 0.9, 1.0, 1.1]), epochs=100)

# make prediction
test_seq = np.array([[1.6, 1.7, 1.8], [1.9, 2.0, 2.1]])
test_seq = np.reshape(test_seq, (test_seq.shape[0], 3, 1))
predictions = model.predict(test_seq)

# print predictions
print(predictions)


# In[2]:





# In[ ]:




