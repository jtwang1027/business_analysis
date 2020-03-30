import pandas as pd
import numpy as np
import calendar
from datetime import datetime, date
from weather_dict import wdict
import os

data=pd.read_csv(r'C:\Users\jwang\Desktop\quickly\items_transactions\items-2019-01-01-2020-02-15.csv')


wdata= pd.read_excel(r'C:\Users\jwang\Desktop\quickly\weather_data.xlsx')
cdata= pd.read_excel(r'C:\Users\jwang\Desktop\quickly\condition_data-april_2019.xlsx')

wc= pd.merge(cdata,wdata, on='Date') # does intersection of data, not UNION
wc['Date']=pd.to_datetime(wc.Date)


#imported as module
#wdict={'Partly Sunny':0, 'Rain':-2, 'Cloudy':-2, 'Mostly Sunny':1, 'Mostly Cloudy':-1,
#       'Partly Cloudy':0, 'Clear':2 , 'Hazy': -2, 'PM Thunderstorms':-3, 'AM Clouds/PM Sun': 1}

#removes soft/grand opening days
#datetime conversions
def preprocess(dat):
    dat.dropna(subset=['Device Name'],inplace= True)
    dat=dat[dat['Qty']<15] #remove bulk orders
    #device name = nan is invoice/bulk orders
    dat['Date']=pd.to_datetime(dat.Date)
    dat=dat[dat.Date>'2018-12-01'] # remove grand opening and earlier dates
    dat['Day']=dat.Date.apply( lambda x: calendar.day_name[x.weekday()])

    def timesplit(t):
        '''returns time of day given in minutes'''
        (h, m, s) = t.split(':')
        return(int(h) * 60 + int(m) + int(s)/60)  
    
    dat['Ltime']=dat.Time.apply(timesplit)

    def hours(t):
        '''returns hour of the day'''
        (h, m, s) = t.split(':')
        return(int(h))
    dat['Hour']=dat.Time.apply(hours)
    
    
    def dollar(y):
        '''remove dollar sign by removing 1st element'''
        return([float(x[1:]) for x in y] )
    
    if '$' in data['Gross Sales'][0]: #so it's not done more than once
        dat[['Gross Sales', 'Discounts', 'Net Sales', 'Tax']]=dat[['Gross Sales', 'Discounts', 'Net Sales', 'Tax']].apply(dollar)
    
    return(dat)

data1=preprocess(data)
data1=pd.merge(data1,wc, on='Date') # extra columns? # row num is correct

keep=['Date', 'Time', 'Category', 'Item', 'Qty','Hour',
 'Net Sales',  'Customer ID','Day', 'Ltime', 'Cond', 'Temp', 'Rain']
data1=data1[keep]


#conditions conversion
data1['qcond']=data1.Cond.map(wdict)

#one-hot encoding for weekdays, Day column auto removed
data1=pd.get_dummies(data1, prefix=[None],columns=['Day'])

data1['sum']=data1.groupby(['Date','Hour'])['Net Sales'].transform('sum')

keep2= ['Temp', 'Rain', 'Friday',
       'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
       'Wednesday','Hour' ,'sum', 'qcond'] # 'Ltime' removed
data2= data1[keep2]


labels=data2['sum']
features= data2.drop(['sum'],axis=1)

data2.to_excel('april_2019-dataset.xlsx')

#feature_list=list(features.columns)
#RF
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size = 0.25, random_state = 42)



from sklearn.ensemble import RandomForestRegressor


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators =5000, random_state = 42, oob_score= True)
rf.fit(features,labels);
print(rf.oob_score_)
'''
# Train the model on training data
rf.fit(train_features, train_labels);

#rf.fit(train_features, train_labels);
print(trees, rf.oob_score_, rf.score(test_features, test_labels))
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
'''

'''
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

rf.score(test_features, test_labels) # use this intead of the bullshit above
test

#variable importance
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
'''


# FC NEURAL NETWORK 
# using data2 as input
import tensorflow as tf
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks  import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

NAME= '10x10x10x1-{}'.format(int(time.time()))

tensorboard=TensorBoard( log_dir= 'logs/{}'.format(NAME))
earlystop=  EarlyStopping(monitor= 'val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
#checkpoint= ModelCheckpoint(filepath='logs/best_model)


features1= preprocessing.scale(features)
x_train1, x_test1, y_train1, y_test1 = train_test_split(features1, labels, test_size=0.2)


#x_scale= preprocessing.MinMaxScaler(data2)
k=len(features1.columns)
try:
    x_train = x_train1.values.reshape(len(x_train), k, )
    x_test = x_test1.values.reshape(len(x_test), k, )
except:
    pass

y_train= y_train1.values
y_test= y_test1.values


model= keras.models.Sequential()
model.add(keras.layers.Dense( k, activation=tf.nn.relu))
model.add(keras.layers.Dense( k, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])


#model.summary()

epochs=100
batch_size=40
#fit model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, callbacks=[tensorboard],
          validation_data=(x_test, y_test))

val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)

model.save(r'C:\Users\jwang\Desktop\quickly\10x2-ANN.model')


'''
use the command below in the directoy C:/Users/jwang/ folder where "logs" is
py -m tensorboard.main --logdir=logs/
copy url into browser
'''


#save model
from sklearn.externals import joblib
joblib.dump(rf,r'C:\Users\jwang\Desktop\quickly\models\transactions-model-190408-RF')
