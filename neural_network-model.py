import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb

data1=pd.read_csv(r'C:\Users\jwang\PycharmProjects\business_analysis\feb_2020-dataset.csv')


y=data1['Gross Sales']
X= data1.drop(['Gross Sales'],axis=1)

# features1= preprocessing.scale(X)
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

k=len(X.columns)
try:
    x_train = x_train.values.reshape(len(x_train), k, )
    x_test = x_test.values.reshape(len(x_test), k, )
except:
    pass

y_train= y_train.values
y_test= y_test.values




# FC NEURAL NETWORK
import tensorflow as tf
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.callbacks  import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

NAME= '10x10x1-{}'.format(int(time.time()))

tensorboard=TensorBoard( log_dir= r'logs\{}'.format(NAME))
earlystop=  EarlyStopping(monitor= 'val_loss', mode='min',verbose=1, patience=5)
checkpoint= ModelCheckpoint(filepath='logs/best_model')


model= keras.models.Sequential()
model.add(keras.layers.Dense( k, activation=tf.nn.relu))
model.add(keras.layers.Dense( k, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])
#model.summary()

epochs=20
batch_size=40
#fit model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, callbacks=[tensorboard, checkpoint],
          validation_data=(x_test, y_test))

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save(r'C:\Users\jwang\PycharmProjects\business_analysis\10x10x1-NN.model') #also saved as best model 2-15-20

'''
use the command below in the directoy C:/Users/jwang/ folder where "logs" is
py -m tensorboard.main --logdir=logs/
copy url into browser
'''


#save model
from sklearn.externals import joblib
joblib.dump(rf,r'C:\Users\jwang\Desktop\quickly\models\transactions-model-190408-RF')
